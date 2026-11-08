#!/usr/bin/env python3
"""Find actual finite-pad grasp endpoints before choosing either arm's start.

Saved cloth, the hand meshes and the reviewed 1 mm / 4 mm2 rule are immutable.
The proposed open width is continuous; it is independently checked by the same
A geometric evaluator, including insertion, cloth and the whole closure sweep.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys,time
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from shapely.errors import GEOSException
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import geometric_check,write,URDF
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,candidate_samples,orientation
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area


def search(payload):
    row,out,budgets,seed,maximum,seed_files,mirror_seeds=payload;began=time.monotonic();limit=budgets[row['stage']];state=load_state(ROOT,row);hand=RightHand(URDF);model=type('HandOnly',(),dict(hand=hand))();skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());settings_sha=sha256(ROOT/'config/geometric_closure.yaml')
    samples=candidate_samples(state,64,np.linspace(0,.030,13));points=np.array(state['particles_world']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();fingers=[v for n,v in hand.vertices(.001).items() if n.endswith(('/left_finger_link','/right_finger_link'))];distances={};rng=np.random.default_rng(seed+row['number']+100*ord(row['stage']));counts=Counter();accepted=[];promising=[];open_promising=[];last_saved=began;trials=0
    def bank(gap,kind='endpoint'):
        if len(hand._banks)>160:hand._banks={k:v for k,v in hand._banks.items() if k[0]==.001}
        return hand.bank(float(gap),kind,pinch_gap=.001)
    def save():
        write(out/'geometry'/f"{row['id']}.json",dict(condition=row,criteria_sha256=settings_sha,hand_urdf_sha256=sha256(URDF),accepted=accepted,counts=dict(counts),trials=trials,elapsed_s=time.monotonic()-began,search_budget_s=limit,seed=seed+row['number']+100*ord(row['stage']),promising=promising[:24],open_promising=open_promising[:24],scope='Hand-only geometry; neither arm IK, route connectivity nor native motion is inferred'))
    old=[]
    if row['stage']=='B':
        from sock_dressing_simulation.reaching_dataset import load_endpoint_samples
        for p in sorted((ROOT/'artifacts/diverse-left-reaching/b-natural-toe/grasp-search').glob(f'*probe/pose-{row["number"]:02d}-B.json')):
            d=json.loads(p.read_text())
            if not d.get('promising_poses'):continue
            old_samples=load_endpoint_samples(d,state,ROOT)
            for candidate in d['promising_poses']:
                sample=old_samples[candidate['sample_index']];R=np.array(candidate['rotation']);p0=np.array(sample['position'])+R@candidate['offset_local_m'];old.append((sample,p0,R))
    # Transport real previously qualified A goals in the local material frame,
    # rather than relying only on an orientation grid about arbitrary points.
    from sock_dressing_simulation.reaching_dataset import load_grasp_samples
    def frame(sample):
        a=np.array(sample['along']);n=np.array(sample['normal']);a=a-n*np.dot(a,n);a/=np.linalg.norm(a)
        return np.column_stack([a,n,np.cross(a,n)])
    for path in sorted((ROOT/'artifacts/diverse-left-reaching/final/routes').glob('*held-A*-trajectory.json')):
        plan=json.loads(path.read_text());gr=plan['geometry_record'];source_sample=load_grasp_samples(gr,ROOT)[gr['candidate']['sample_index']];source_R=np.array(gr['candidate']['rotation_matrix']);source_p=np.array(gr['candidate']['grasp_center_unity_world_m'])
        if row['stage']=='A' and row['number']==plan['number']:
            old.insert(0,(source_sample,source_p,source_R))
        def proximity(sample):
            angular=(sample['theta_deg']-source_sample['theta_deg']+180)%360-180
            return (angular/360*.1)**2+(sample['depth_m']-source_sample['depth_m'])**2
        target=min(samples,key=proximity);change=frame(target)@frame(source_sample).T
        old.append((target,np.array(target['position'])+change@(source_p-source_sample['position']),change@source_R))
    for path in seed_files:
        source=json.loads(Path(path).read_text())
        for accepted_seed in source.get('accepted',[]):
            source_sample=accepted_seed['sample'];source_R=np.array(accepted_seed['rotation']);source_p=np.array(accepted_seed['position'])
            if source['condition']['id']==row['id']:old.insert(0,(source_sample,source_p,source_R));continue
            def proximity(sample):
                angular=(sample['theta_deg']-source_sample['theta_deg']+180)%360-180
                return (angular/360*.1)**2+(sample['depth_m']-source_sample['depth_m'])**2
            target=min(samples,key=proximity);change=frame(target)@frame(source_sample).T
            old.append((target,np.array(target['position'])+change@(source_p-source_sample['position']),change@source_R))
    if mirror_seeds:
        from sock_dressing_simulation.held_ankle_poses import foot_basis
        foot=np.array(state['source']['surfaces'][0]['vertices']);center=(foot.min(0)+foot.max(0))/2;normal=foot_basis(state['source'])[:,2];reflection=np.eye(3)-2*np.outer(normal,normal);mirrored=[]
        sample_positions=np.array([s['position'] for s in samples])
        for source_sample,p0,R0 in old:
            anchor=center+reflection@(np.array(source_sample['position'])-center);target=samples[int(np.argmin(np.linalg.norm(sample_positions-anchor,axis=1)))];mapped=np.array(target['position'])+reflection@(p0-source_sample['position'])
            for flip in (np.diag([1,1,-1]),np.diag([1,-1,1])):
                mirrored.append((target,mapped,reflection@R0@flip))
        old=mirrored+old
    # Budget only the search itself, not material interpolation and seed I/O.
    began=time.monotonic();last_saved=began
    while time.monotonic()-began<limit and len(accepted)<maximum:
        trials+=1
        if trials<=len(old):sample,p,R=old[trials-1]
        elif (open_promising or promising) and rng.random()<.55:
            pool=open_promising if open_promising and rng.random()<.65 else promising
            if not pool:pool=open_promising
            item=pool[int(rng.integers(min(len(pool),16)))];sample=item['sample'];scale=float(rng.choice([.25,.5,1,2]));R=np.array(item['rotation'])@Rotation.from_rotvec(rng.normal(0,np.deg2rad(2)*scale,3)).as_matrix();p=np.array(item['position'])+R@rng.normal(0,[.0005,.0001,.0005],3)*scale
        else:
            sample=samples[int(rng.integers(len(samples)))];tilt=float(rng.uniform(-30,15) if rng.random()<.7 else rng.uniform(-85,85));yaw=float(rng.uniform(0,360));roll=float(rng.normal(0,8) if rng.random()<.8 else rng.uniform(-180,180))
            if rng.random()<.5:roll+=180
            R=orientation(sample,yaw,tilt)@Rotation.from_rotvec([np.deg2rad(roll),0,0]).as_matrix();p=np.array(sample['position'])+R@np.r_[rng.uniform(-.004,.004),rng.normal(0,.00025),rng.uniform(-.004,.004)]
        try:
            if not skin.inspect(bank(.001),R,p,.001)[0]:counts['closed_skin']+=1;continue
            counts['closed_skin_passed']+=1;key=(sample['face'],tuple(sample['weights']))
            if key not in distances:distances[key]=material_distances(rest,faces,faces[sample['face']],sample['weights'])
            projection=projected_cloth_area((points-p)@R,faces,fingers,distances[key],depth,**{n:settings[n] for n in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')});score=min(projection['area_m2'],projection['largest_component_area_m2'])*1e6
            if score<.5:counts['area_small']+=1;continue
            seed_record=dict(sample=sample,position=p.tolist(),rotation=R.tolist(),area_mm2=score)
            promising.append(seed_record);promising.sort(key=lambda r:-r['area_mm2']);del promising[48:]
            # The convex closure sweeps are nested in width. Find the largest
            # skin-safe opening; a discrete 1.25/1.5 mm grid can miss this interval.
            lo=.001;hi=.05
            if skin.inspect(bank(hi,'closure'),R,p,.001)[0]:lo=hi
            else:
                for _ in range(15):
                    mid=(lo+hi)/2
                    if skin.inspect(bank(mid,'closure'),R,p,.001)[0]:lo=mid
                    else:hi=mid
            safe=max(.00100001,lo-2e-7);tested=[]
            for gap in [safe]+list(np.linspace(safe,.00100001,9)[1:]):
                if not skin.inspect(bank(gap,'closure'),R,p,.001)[0]:continue
                hits=cloth.intersections(bank(gap),R,p)
                if hits:
                    counts['open_cloth']+=1;continue
                counts['open_and_closure_clear']+=1;tested.append(float(gap));break
            if not tested:continue
            open_promising.append(dict(seed_record,opening_gap_m=tested[0]));open_promising.sort(key=lambda r:-r['area_mm2']);del open_promising[48:]
            if score<4 or projection['area_fraction']<.05 or min(projection['span_x_m'],projection['span_z_m'])<.0005:counts['capture_insufficient']+=1;continue
            geometry,reason=geometric_check(model,state,dict(sample,position=p.tolist()),R,skin,cloth,settings,distances[key],approach_distance=0.,opening_gaps=tested);counts[reason]+=1
            if geometry:
                if any(np.linalg.norm(p-r['position'])<.002 and Rotation.from_matrix(R@np.array(r['rotation']).T).magnitude()<np.deg2rad(12) for r in accepted):continue
                accepted.append(dict(position=p.tolist(),rotation=R.tolist(),sample=sample,geometry=geometry));print('GRASP GEOMETRY',row['id'],len(accepted),'area mm2',score,'opening mm',tested[0]*1000,flush=True);save()
        except (GEOSException,ValueError) as exc:
            if isinstance(exc,ValueError) and 'Ambiguous solid containment' not in str(exc):raise
            counts['numerically_unresolved_rejected']+=1
        if time.monotonic()-last_saved>20:save();last_saved=time.monotonic();print('GEOMETRY PROGRESS',row['id'],trials,dict(counts),'best',promising[0]['area_mm2'] if promising else 0,flush=True)
    assert sha256(ROOT/row['state'])==row['state_sha256'] and sha256(ROOT/'config/geometric_closure.yaml')==settings_sha
    save();print('GEOMETRY DONE',row['id'],len(accepted),trials,flush=True)
    return dict(id=row['id'],stage=row['stage'],accepted=len(accepted),trials=trials,counts=dict(counts))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/common-grasp-start-search');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--stages',default='ABCDE');ap.add_argument('--ids',nargs='*');ap.add_argument('--seconds',type=float);ap.add_argument('--maximum',type=int,default=8);ap.add_argument('--mirror-seeds',action='store_true');ap.add_argument('--seed-geometries',nargs='*',type=Path,default=[]);ap.add_argument('--seed',type=int,default=20261109);a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);rows=[r for r in inventory(ROOT) if r['stage'] in a.stages and (not a.ids or r['id'] in a.ids)];budgets=dict(A=20,B=240,C=90,D=150,E=120)
    if a.seconds is not None:budgets={s:a.seconds for s in 'ABCDE'}
    write(out/'inputs.json',dict(rows=rows,criteria_sha256=sha256(ROOT/'config/geometric_closure.yaml'),fixed_robot_urdf_sha256=sha256(URDF),approach='Endpoint capture before common-start search; no shape, gripper or criterion changes'))
    done=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(search,(r,out,budgets,a.seed,a.maximum,a.seed_geometries,a.mirror_seeds)) for r in rows]
        for f in as_completed(futures):done.append(f.result());write(out/'geometry-progress.json',dict(completed=len(done),requested=len(rows),with_grasp_geometry=sum(r['accepted']>0 for r in done),rows=done))

if __name__=='__main__':main()
