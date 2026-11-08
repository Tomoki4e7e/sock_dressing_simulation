#!/usr/bin/env python3
"""Continuous pose/opening search; only the unchanged A hard checks can accept."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np,yaml,fcl
from scipy.optimize import differential_evolution
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from shapely.errors import GEOSException
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import URDF,write,geometric_check
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import projected_cloth_area,material_distances
from sock_dressing_simulation.reaching_dataset import load_endpoint_samples


def run(identifier,seed_index,seconds,out,seed):
    row=next(r for r in inventory(ROOT) if r['id']==identifier);state=load_state(ROOT,row);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());criteria_sha=sha256(ROOT/'config/geometric_closure.yaml');hand=RightHand(URDF);model=type('HandOnly',(),dict(hand=hand))();skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);points=np.array(state['particles_world']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();bank=hand.bank(.001);fingers=[v for n,v in hand.vertices(.001).items() if n.endswith(('/left_finger_link','/right_finger_link'))]
    seeds=[]
    if row['stage']=='B':
        for path in sorted((ROOT/'artifacts/diverse-left-reaching/b-natural-toe/grasp-search').glob(f'*probe/pose-{row["number"]:02d}-B.json')):
            d=json.loads(path.read_text())
            if not d.get('promising_poses'):continue
            samples=load_endpoint_samples(d,state,ROOT)
            for r in d['promising_poses']:
                R=np.array(r['rotation']);sample=samples[r['sample_index']];seeds.append(dict(sample=sample,position=(np.array(sample['position'])+R@r['offset_local_m']).tolist(),rotation=R.tolist(),area_mm2=r['connected_area_mm2']))
    for root in [ROOT/'artifacts/common-grasp-start-search',ROOT/'artifacts/common-grasp-start-search/transport']:
        path=root/'geometry'/f'{identifier}.json'
        if path.exists():
            d=json.loads(path.read_text());seeds.extend(d.get('open_promising',[])+d.get('promising',[]))
    if not seeds:raise ValueError('No geometry seed: '+identifier)
    seeds.sort(key=lambda r:-r['area_mm2']);candidate=seeds[seed_index%len(seeds)];sample=candidate['sample'];p0=np.array(candidate['position']);R0=np.array(candidate['rotation']);dist=material_distances(rest,faces,faces[sample['face']],sample['weights'])
    edges=np.unique(np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0);cloud=np.r_[points,points[edges].mean(1),points[faces].mean(1)];parts=[]
    for part in bank:
        v=part.vertices;dy=0 if part.label.endswith('/gripper_base') else (-.5 if '/left_' in part.label else .5);parts.append((part,ConvexHull(v).equations,v.min(0),v.max(0),dy))
    rng=np.random.default_rng(seed);accepted=[];counts=dict(evaluations=0,skin_rejections=0,hard_rejections=0);best=None;began=time.monotonic();last_saved=began
    path=out/f'{identifier}-seed-{seed_index:02d}.json'
    def save():
        write(path,dict(condition=row,criteria_sha256=criteria_sha,seed=seed,source_seed=candidate,seconds=seconds,elapsed_s=time.monotonic()-began,counts=counts,best=best,accepted=accepted,scope='DE uses only a point-cloud penetration surrogate; every accepted pose independently passes the existing full-mesh A geometry and closure rules'))
    def open_skin(R,p,gap):
        minimum=float('inf');centers=[]
        for part,planes,lo,hi,dy in parts:
            shift=np.array([0,dy*(gap-.001),0]);part.obj.setTransform(fcl.Transform(R,p+R@shift));distance=float(fcl.distance(part.obj,skin.obj,fcl.DistanceRequest(),fcl.DistanceResult()))
            if not np.isfinite(distance):raise ValueError('Nonfinite skin distance')
            if distance<.001:return False,max(0,distance)
            minimum=min(minimum,distance);centers.append(p+R@(part.center+shift))
        centers=np.array(centers);inside=((centers>=skin.minimum)&(centers<=skin.maximum)).all(1)
        if inside.any() and (skin.winding(centers[inside])>=.5).any():return False,0.
        return True,minimum
    def objective(x):
        nonlocal best,last_saved
        if time.monotonic()-began>seconds:raise TimeoutError()
        counts['evaluations']+=1;R=R0@Rotation.from_rotvec(np.deg2rad(x[3:6])).as_matrix();p=p0+R0@(x[:3]*.001);gap=float(x[6]*.001)
        clear,skin_distance,_,_=skin.inspect(bank,R,p,.001)
        if not clear:counts['skin_rejections']+=1;return 1e6+(1-skin_distance/.001)*1e4+float(np.sum(x[:6]**2))
        clear,opening_skin=open_skin(R,p,gap)
        if not clear:counts['skin_rejections']+=1;return 1e5+(1-opening_skin/.001)*1e4+float(np.sum(x[:6]**2))
        local=(points-p)@R
        try:area=projected_cloth_area(local,faces,fingers,dist,depth,**{n:settings[n] for n in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
        except GEOSException:return 1e6
        captured=min(area['area_m2'],area['largest_component_area_m2'])*1e6
        local_cloud=(cloud-p)@R;penetration=0.
        for part,planes,lo,hi,dy in parts:
            shift=np.array([0,dy*(gap-.001),0]);keep=((local_cloud>=lo+shift-.00002)&(local_cloud<=hi+shift+.00002)).all(1)
            if keep.any():
                signed=np.max((local_cloud[keep]-shift)@planes[:,:3].T+planes[:,3],axis=1);penetration+=float(np.square(np.maximum(.00002-signed,0)*1000).sum())
        deficit=max(0,4-captured)**2+max(0,.05-area['area_fraction'])**2*100+sum(max(0,.0005-area[n])**2*1e6 for n in ('span_x_m','span_z_m'))
        value=1e4*deficit+penetration-.001*min(captured,8)
        valid=None
        if deficit==0 and penetration<.01:
            if len(hand._banks)>128:hand._banks={k:v for k,v in hand._banks.items() if k[0]==.001}
            try:geometry,reason=geometric_check(model,state,dict(sample,position=p.tolist()),R,skin,cloth,settings,dist,approach_distance=0.,opening_gaps=[gap])
            except (ValueError,GEOSException):geometry=None;reason='numerical_rejected'
            if geometry:
                valid=dict(position=p.tolist(),rotation=R.tolist(),sample=sample,geometry=geometry);accepted.append(valid);save();print('OPTIMIZED GRASP',identifier,seed_index,'area',captured,'gap mm',gap*1000,flush=True);raise StopIteration()
            counts['hard_rejections']+=1;value+=1
        if best is None or value<best['objective']:
            best=dict(objective=value,point_penetration=penetration,capture_area_mm2=captured,opening_gap_m=gap,closed_skin_distance_m=skin_distance,open_skin_distance_m=opening_skin,x=x.tolist(),position=p.tolist(),rotation=R.tolist(),sample=sample)
        if time.monotonic()-last_saved>20:save();last_saved=time.monotonic();print('OPT PROGRESS',identifier,seed_index,counts['evaluations'],'objective',best['objective'],'area',best['capture_area_mm2'],'penetration',best['point_penetration'],flush=True)
        return value
    bounds=[(-8,8),(-2,2),(-8,8)]+[(-35,35)]*3+[(1.001,8)]
    pop=rng.uniform(np.array(bounds)[:,0],np.array(bounds)[:,1],(70,7));pop[:55,:6]=rng.normal(0,[.35,.05,.35,.8,.8,.8],(55,6));pop[:55,6]=np.clip(rng.normal(1.1,.04,55),1.001,8);pop[0]=[0,0,0,0,0,0,1.01]
    try:differential_evolution(objective,bounds,init=pop,rng=rng,mutation=(.5,1.3),recombination=.85,popsize=10,maxiter=10000,polish=False,tol=0,atol=0,callback=lambda *args:bool(accepted))
    except (TimeoutError,StopIteration):pass
    assert sha256(ROOT/row['state'])==row['state_sha256'] and sha256(ROOT/'config/geometric_closure.yaml')==criteria_sha
    save();print('OPT DONE',identifier,seed_index,len(accepted),counts,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--id',required=True);ap.add_argument('--seed-index',type=int,default=0);ap.add_argument('--seconds',type=float,default=600);ap.add_argument('--seed',type=int,default=20261109);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/common-grasp-start-search/optimization');a=ap.parse_args();run(a.id,a.seed_index,a.seconds,a.output.resolve(),a.seed+a.seed_index)
