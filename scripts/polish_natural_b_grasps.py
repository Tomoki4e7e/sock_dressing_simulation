#!/usr/bin/env python3
"""Locally optimize B poses under the unchanged fixed-gap/window rule.

The objective measures cloth area penetrating slightly padded open-hand convex
pieces. It is only a search surrogate. Acceptance always calls the independent
hard geometry check, including the original finite 8 mm capture window.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,time
from pathlib import Path
import numpy as np,yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from shapely.errors import GEOSException
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import geometric_check,write,URDF
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import projected_cloth_area,material_distances,clip_polygon
from sock_dressing_simulation.reaching_dataset import file_sha256,load_endpoint_samples
from sock_dressing_simulation.fitted_b_cuff import PROFILE


def penetration_area(local,faces,hulls):
    area=0.;triangles=local[faces]
    for low,high,planes in hulls:
        eligible=((triangles.max(1)>=low)&(triangles.min(1)<=high)).all(1)
        for triangle in triangles[eligible]:
            polygon=triangle
            for plane in planes:
                polygon=clip_polygon(polygon,plane)
                if len(polygon)<3:break
            if len(polygon)>=3:
                area+=.5*np.linalg.norm(np.cross(polygon[1:-1]-polygon[0],polygon[2:]-polygon[0]),axis=1).sum()
    return float(area*1e6)


def run(number,source,out,seconds):
    probe=json.loads((source/f'pose-{number:02d}-B.json').read_text());state_path=ROOT/probe['frozen_source_file'];state=json.loads(state_path.read_text());sha=file_sha256(state_path)
    assert state['shape_profile']==PROFILE
    samples=load_endpoint_samples(probe,state,ROOT);skin=SkinBody(state['source']['skin_surface']);hand=RightHand(URDF);cloth=ClothScreen(state['particles_world'],state['triangles']);model=type('HandOnly',(),dict(hand=hand))()
    points=np.array(state['particles_world']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text())
    gap=.0011;hulls=[]
    for part in hand.bank(gap):
        equations=ConvexHull(part.vertices).equations.copy();equations[:,3]-=.00002
        hulls.append((part.vertices.min(0)-.00002,part.vertices.max(0)+.00002,equations))
    fingers=[v for name,v in hand.vertices(.001).items() if name.endswith(('/left_finger_link','/right_finger_link'))]
    seeds=probe.get('promising_poses',[]);counts=dict(evaluations=0,solver_trials=0,numerical_rejected=0);accepted=[];trials=[];began=time.monotonic()
    for seed in seeds[:4]:
        if time.monotonic()-began>=seconds:break
        sample=samples[seed['sample_index']];R0=np.array(seed['rotation']);p0=np.array(sample['position'])+R0@np.array(seed['offset_local_m']);dist=material_distances(rest,faces,faces[sample['face']],sample['weights']);cache={}
        def evaluate(x):
            if time.monotonic()-began>=seconds:raise TimeoutError()
            key=tuple(x)
            if key in cache:return cache[key]
            counts['evaluations']+=1;R=Rotation.from_rotvec(np.deg2rad(x[3:])).as_matrix()@R0;p=p0+R0@(np.array(x[:3])*.001);local=(points-p)@R
            try:area=projected_cloth_area(local,faces,fingers,dist,depth,**{k:settings[k] for k in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
            except GEOSException:
                counts['numerical_rejected']+=1;area=dict(area_m2=0,largest_component_area_m2=0,area_fraction=0,span_x_m=0,span_z_m=0)
            clear,distance,_,_=skin.inspect(hand.bank(gap),R,p,.001)
            captured=min(area['area_m2'],area['largest_component_area_m2'])*1e6
            constraints=np.r_[captured-4,area['area_fraction']/.05-1,area['span_x_m']/.0005-1,area['span_z_m']/.0005-1,distance/.001-1]
            objective=penetration_area(local,faces,hulls)
            cache[key]=(objective,constraints,p,R,captured);return cache[key]
        try:
            counts['solver_trials']+=1
            fit=minimize(lambda x:evaluate(x)[0],np.zeros(6),method='SLSQP',bounds=[(-3,3)]*3+[(-12,12)]*3,constraints=[dict(type='ineq',fun=lambda x:evaluate(x)[1])],options=dict(maxiter=80,ftol=1e-8,eps=1e-3))
            objective,constraints,p,R,area=evaluate(fit.x)
            shifted=dict(sample,position=p.tolist())
            try:geometry,reason=geometric_check(model,state,shifted,R,skin,cloth,settings,dist,approach_distance=0.)
            except GEOSException:geometry=None;reason='numerical_geometry_rejected'
            trials.append(dict(sample_index=sample['index'],solver_success=bool(fit.success),message=str(fit.message),penetration_area_mm2=objective,capture_area_mm2=area,constraints=constraints.tolist(),hard_check=reason,position=p.tolist(),rotation=R.tolist()))
            if geometry:
                accepted.append(dict(number=number,sample_index=sample['index'],rotation=R.tolist(),position=p.tolist(),offset_local_m=(R.T@(p-np.array(sample['position']))).tolist(),geometry=geometry))
                print('POLISHED GEOMETRY',number,'PASSED',flush=True)
        except TimeoutError:break
    assert file_sha256(state_path)==sha
    write(out/f'pose-{number:02d}-B.json',dict(number=number,shape_profile=PROFILE,source=str((source/f'pose-{number:02d}-B.json').relative_to(ROOT)),source_state_sha256=sha,elapsed_s=time.monotonic()-began,counts=counts,trials=trials,accepted_geometry=accepted,scope='Local continuous pose optimization only; hard checks unchanged; arm IK/routes not inferred'))
    print('POLISH RESULT',number,len(accepted),counts,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--numbers',type=int,nargs='+',default=[19,38,8]);ap.add_argument('--source',type=Path,default=ROOT/'artifacts/diverse-left-reaching/b-natural-toe/grasp-search/wide-pose-probe');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/b-natural-toe/grasp-search/local-polish');ap.add_argument('--seconds',type=float,default=90);a=ap.parse_args()
    if not np.isfinite(a.seconds) or a.seconds<=0:ap.error('Require a finite positive search budget')
    for n in a.numbers:run(n,a.source.resolve(),a.output.resolve(),a.seconds)
