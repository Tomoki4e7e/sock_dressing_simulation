#!/usr/bin/env python3
"""Adjust a photo-start grasp endpoint without moving the human or torso.

Endpoint proposals retain the 1 mm skin / 4 mm² capture rules. The generic
straight 50 mm tool sweep is replaced by an independently checked robot path.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import geometric_check,write,REGISTERED_MANIFEST
from search_photo_reference_reaching import photo_context
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances
from scipy.optimize import minimize
from plan_fixed_torso_route import rrt


def inputs():
    initial=yaml.safe_load((ROOT/'config/photo_reference_initial_pose.yaml').read_text());n=initial['number']
    state_file=f'artifacts/bulged-cuff/pose-{n:02d}/state.json';state=json.loads((ROOT/state_file).read_text())
    k,c,dl,du,contract,scene=photo_context(n,state,initial['left_foot_floor_allowance_m']);c.park_left(initial['left_q_rad'],initial['left_gap_m'])
    prior=json.loads((ROOT/'artifacts/photo-reference-reaching/summary.json').read_text());plan=json.loads((ROOT/prior['selected_trajectory']).read_text())
    return initial,state_file,state,k,c,contract,plan


def geometry(a):
    initial,state_file,state,k,c,contract,plan=inputs();n=initial['number'];delta=np.array(initial['translation_m']);q=np.array(plan['q_rad'][-1]);p0,R0=k.pose(q);p0-=delta
    samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/samples.json').read_text());positions=np.array([s['position'] for s in samples]);tree=cKDTree(positions)
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);distances={};counts=Counter();accepted=[];started=time.monotonic()
    rotations=[R0]
    for angle in (5,10,15,30,45,60):
        for axis in np.r_[np.eye(3),-np.eye(3)]:rotations.append(R0@Rotation.from_rotvec(np.deg2rad(angle)*axis).as_matrix())
    # Include a full half-circle of jaw assignments about the approach axis.
    for roll in range(15,181,15):rotations.append(R0@Rotation.from_rotvec([np.deg2rad(roll),0,0]).as_matrix())
    seen=set();tested_positions=0
    if a.mode=='local':
        points=[p0+np.array(off) for off in itertools.product(np.arange(-.008,.0081,.002),np.arange(0,.0101,.002),np.arange(-.008,.0081,.002))]
        points.sort(key=lambda p:np.linalg.norm(p-p0))
    else:
        selected=[s for s in samples if s['depth_m']<=.0100001]
        selected.sort(key=lambda s:(-s['position'][1],np.linalg.norm(np.array(s['position'])-p0)))
        points=[]
        for s in selected:
            for radial,axial in itertools.product((0.,.002,.004,-.002), (0.,.002,.004)):
                points.append(np.array(s['position'])+radial*np.array(s['normal'])+axial*np.array(state['opening_normal']))
    for point in points:
        key=tuple(np.round(point,8))
        if key in seen:continue
        seen.add(key);tested_positions+=1;si=int(tree.query(point)[1]);sample=dict(samples[si],position=point.tolist())
        if si not in distances:distances[si]=material_distances(rest,faces,faces[sample['face']],sample['weights'])
        for ri,R in enumerate(rotations):
            value,reason=geometric_check(k,state,sample,R,skin,cloth,settings,distances[si],approach_distance=0.);counts[reason]+=1;counts['tested']+=1
            if value:
                value.update(number=n,state_file=state_file,source=state_file,source_kind='bulged',generator='photo_fixed_layout_closure_adjustment',material_sample_position_m=samples[si]['position'],target_offset_from_material_sample_m=(point-positions[si]).tolist(),approach_distance_m=0.,actual_robot_route_required=True,
                    position_change_from_previous_endpoint_m=(point-p0).tolist(),orientation_change_deg=float(np.rad2deg(Rotation.from_matrix(R@R0.T).magnitude())),rotation_index=ri)
                accepted.append(value);print('CLOSURE GEOMETRY',len(accepted),'sample',si,'d mm',((point-p0)*1000).round(2),'rotation',value['orientation_change_deg'],'gaps',[g['gap_m'] for g in value['valid_openings']],flush=True)
        if tested_positions%10==0 or len(accepted)>=a.maximum:
            write(a.output/f'geometry-{a.mode}.json',dict(mode=a.mode,number=n,counts=dict(counts),tested_positions=tested_positions,accepted=accepted,elapsed_s=time.monotonic()-started,
                initial_configuration=initial,reference_endpoint_source=plan['source_record'],state_sha256=hashlib.sha256((ROOT/state_file).read_bytes()).hexdigest()))
            print('CLOSURE SEARCH',a.mode,tested_positions,dict(counts),flush=True)
        if len(accepted)>=a.maximum:break
    write(a.output/f'geometry-{a.mode}.json',dict(mode=a.mode,number=n,counts=dict(counts),tested_positions=tested_positions,accepted=accepted,elapsed_s=time.monotonic()-started,initial_configuration=initial))


def endpoints(a):
    initial,state_file,state,base,checker,contract,plan=inputs();n=initial['number'];delta=np.array(initial['translation_m'])
    from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene
    k=ControlledArmKinematics(base,a.side,initial['right_q_rad'],initial['right_gap_m']);c=ControlledCollisionScene(checker,a.side,initial['right_q_rad'],initial['right_gap_m'])
    start=np.array(initial[a.side+'_q_rad']);prior=np.array(plan['q_rad'][-1]) if a.side=='right' else start
    lo=k.lower+np.deg2rad(5);hi=k.upper-np.deg2rad(5);comfortable=copy.copy(k);comfortable.lower=lo;comfortable.upper=hi;pool=[]
    for path in sorted(a.output.glob('geometry-*.json')):
        for i,g in enumerate(json.loads(path.read_text())['accepted']):pool.append((str(path.relative_to(ROOT)),i,g))
    refs=json.loads((ROOT/'artifacts/centered-reaching/full-grid/geometry-summary.json').read_text())['accepted']
    for ref in refs:
        if ref['number']!=n:continue
        g=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']];pool.append((ref['file'],ref['candidate_index'],g))
    rng=np.random.default_rng(20261106);counts=Counter();found=[]
    for index,(source,gi,gr) in enumerate(pool):
        output=a.output/('endpoints' if a.side=='right' else 'left-endpoints')/f'target-{index:04d}.json'
        if output.exists():continue
        p=np.array(gr['candidate']['grasp_center_unity_world_m'])+delta;R=np.array(gr['candidate']['rotation_matrix']);attempts=[]
        seeds=[prior,start,np.deg2rad([150,-40,-100,95,140,70,60]),np.deg2rad([180,-60,-80,90,-140,-60,60]),np.deg2rad([150,-50,-100,90,160,-70,70])]
        seeds+=list(rng.uniform(lo,hi,size=(max(0,a.seeds-len(seeds)),7)))
        for si,seed in enumerate(seeds[:a.seeds]):
            sol=comfortable.solve(p,R,np.clip(seed,lo+1e-9,hi-1e-9));q=np.array(sol['q']);attempt=dict(seed=si,**sol,stage='ik_failed');attempts.append(attempt)
            if sol['position_error_m']>1e-4 or sol['orientation_error_deg']>.5:continue
            for opening in sorted(gr['valid_openings'],key=lambda g:(abs(g['gap_m']-.04),-g['gap_m'])):
                gap=opening['gap_m'];check=c.slacks(q,gap,delta,details=True);attempt['stage']='endpoint_collision';attempt['endpoint']=check
                if not check['passed']:
                    cache={}
                    from search_compound_reaching import pose_constraint
                    def eq(angles):return pose_constraint(k,p,R,np.r_[angles,np.zeros(3)])
                    def eqjac(angles):return pose_constraint(k,p,R,np.r_[angles,np.zeros(3)],True)[1][:,:7]
                    def constraints(angles):
                        if 'q' not in cache or not np.array_equal(cache['q'],angles):
                            g,J=c.slacks(angles,gap,delta,with_jacobian=True);cache.update(q=angles.copy(),g=g,J=J[:,:7])
                        return cache['g'],cache['J']
                    opt=minimize(lambda angles:np.sum((angles-q)**2),q,jac=lambda angles:2*(angles-q),method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=eq,jac=eqjac),dict(type='ineq',fun=lambda v:constraints(v)[0]-.0003,jac=lambda v:constraints(v)[1])],options=dict(maxiter=60,ftol=1e-10))
                    e=eq(opt.x)
                    if np.linalg.norm(e[:3])<=1e-4 and np.linalg.norm(e[3:])<=np.deg2rad(.5):q=opt.x;check=c.slacks(q,gap,delta,details=True)
                if not check['passed']:continue
                closure=[c.slacks(q,float(g),delta,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,41)]
                if not all(v['passed'] for v in closure):attempt.update(stage='closure_collision',closure=closure);continue
                attempt.update(stage='endpoint_passed',q=q.tolist(),gap_m=gap,endpoint=check)
                print('ADJUSTED ENDPOINT',index,n,'seed',si,'gap',gap,'q',np.rad2deg(q).round(1),flush=True)
                route=rrt(k,c,start,q,gap,delta,iterations=a.iterations);attempt['trajectory']=route
                print('ADJUSTED ROUTE',index,route['passed'],route['stage'],flush=True)
                if route['passed']:
                    distance=float(np.linalg.norm(k.pose(start)[0]-p));contract.update(minimum_start_tcp_distance_m=distance,minimum_start_skin_cloth_clearance_m=.005)
                    record=dict(number=n,geometry_record=gr,geometry_reference=dict(file=source,index=gi),central_contract=contract,left_gap_m=initial['left_gap_m'],right_gap_m=initial['right_gap_m'],left_foot_floor_allowance_m=initial['left_foot_floor_allowance_m'],
                        moving_side=a.side,right_q_rad=initial['right_q_rad'],left_q_rad=initial['left_q_rad'],initial_opening_gap_m=initial[a.side+'_gap_m'],endpoint_task='geometric_grasp_target',require_virtual_closure=True,minimum_start_tcp_distance_m=distance,minimum_start_skin_cloth_clearance_m=.005,
                        close_after_reach=True,closing_gap_m=.001,attempts=[dict(attempt,left_q_rad=initial['left_q_rad'],translation_m=delta.tolist())],human_and_torso_unchanged=True)
                    write(a.output/'planned'/f'target-{a.side}-{index:04d}.json',record);found.append(index);break
            if found:break
        for v in attempts:counts[v['stage']]+=1
        write(output,dict(number=n,geometry_source=source,geometry_index=gi,geometry_record=gr,attempts=attempts));print('ENDPOINT SEARCH',index,dict(counts),flush=True)
        if found:break
    write(a.output/f'{a.side}-endpoint-summary.json',dict(counts=dict(counts),found_routes=found,initial_configuration=initial))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/photo-grasp-adjustment');ap.add_argument('--mode',choices=['local','rim','endpoints'],default='local');ap.add_argument('--side',choices=['left','right'],default='right');ap.add_argument('--maximum',type=int,default=16);ap.add_argument('--seeds',type=int,default=16);ap.add_argument('--iterations',type=int,default=1500);a=ap.parse_args();a.output=a.output.resolve();(endpoints if a.mode=='endpoints' else geometry)(a)


if __name__=='__main__':main()
