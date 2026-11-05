#!/usr/bin/env python3
"""Reselect both arm starts and centered human translations; torso stays fixed."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,state_for,URDF,ROBOT,FAST_MANIFEST,REGISTERED_MANIFEST
from search_compound_reaching import pose_constraint
from search_translated_reaching import load,good,margin
from apron_reaching_common import overlay_robot
from sock_dressing_simulation.arm_reaching import Kinematics,ARM
from sock_dressing_simulation.centered_reaching import placement_bounds
from sock_dressing_simulation.fixed_torso_reaching import FixedTorsoScene,LEFT_ARM


def context(number,state,manifest=FAST_MANIFEST,left_foot_floor_allowance=0.):
    reference=json.loads(ROBOT.read_text());scene=overlay_robot(load(number),reference);scene['particles_world']=state['particles_world']
    k=Kinematics(URDF,scene);checker=FixedTorsoScene(k,scene,state['triangles'],manifest,ROOT)
    # Bounds use the unchanged reference frames, never the newly selected arms.
    dl,du,contract=placement_bounds(scene,state,edge_margin=0.)
    if left_foot_floor_allowance:
        checker.allow_grounded_left_foot_penetration(left_foot_floor_allowance)
        dl[1]=max(-.3,max(checker.floor_top-y for y in checker.floor_minima.values()))
        contract.update(left_foot_floor_allowance_m=float(left_foot_floor_allowance),translation_lower_m=dl.tolist())
    contract['reference']='Original robot lateral interval, anchored to the unchanged base/torso; independent of selected arm starts'
    return k,checker,dl,du,contract,scene


def parks(args):
    state=json.loads((ROOT/'artifacts/bulged-cuff/pose-10/state.json').read_text());k,checker,dl,du,contract,scene=context(10,state)
    candidates=[];counts=Counter();rng=np.random.default_rng(20261107)
    grid=[]
    for j1 in [-45,0,47,80,125,150,180,210]:
        for j2 in [-110,-80,-50,0,10]:
            for j3 in [-60,0,60]:
                for j4 in [35,65,100,125]:
                    for j5,j6,j7 in [(0,-60,55),(0,60,55),(-90,-45,30),(90,45,30),(0,-80,20),(0,80,20)]:
                        grid.append(np.deg2rad([j1,j2,j3,j4,j5,j6,j7]))
    rng.shuffle(grid)
    for angles in grid:
        checker.park_left(angles);counts['tested']+=1
        # Human/cloth/support are moved remotely only for robot-self screening.
        if not checker.is_clear(k.start,.04,[0,0,-3]):continue
        counts['robot_self_clear']+=1
        frames=k.fk(k.start,only=['left_gripper/grasping_frame']);T=frames['left_gripper/grasping_frame'];p=k.A@T[:3,3]+k.b
        if p[0]<.2:continue
        if any(np.linalg.norm(angles-np.array(c['q_rad']))<.6 for c in candidates):continue
        candidates.append(dict(q_rad=angles.tolist(),q_deg=np.rad2deg(angles).tolist(),grasping_frame_world_m=p.tolist()))
        print('LEFT PARK',len(candidates),np.rad2deg(angles),p,flush=True)
        if len(candidates)>=args.park_count:break
    write(args.output/'left-parks.json',dict(counts=dict(counts),candidates=candidates,left_gap_m=.04,
        torso_joints_rad=checker.reference_torso,collision_policy=checker.srdf_policy,reference_robot_scene_sha256=hashlib.sha256(ROBOT.read_bytes()).hexdigest()))
    if not candidates:raise RuntimeError('No left parked pose passed robot self collision screening')


def retract_route(k,checker,c,q,delta,gap,minimum_start_distance=.15):
    """Find a nontrivial new start by tracing a collision-free Cartesian retreat."""
    comfortable=copy.copy(k);comfortable.lower=k.lower+np.deg2rad(5);comfortable.upper=k.upper-np.deg2rad(5)
    p=np.array(c['grasp_center_unity_world_m'])+delta;R=np.array(c['rotation_matrix']);rows=[np.array(q)];selected=None;failure=None
    for distance in np.arange(.0025,.3001,.0025):
        sol=comfortable.solve(p-R[:,0]*distance,R,rows[-1]);qq=np.asarray(sol['q'])
        if not good(sol) or margin(k,qq)<5-1e-7:failure='retreat_ik';break
        if np.max(abs(qq-rows[-1]))>np.deg2rad(10):failure='retreat_branch_jump';break
        check=checker.slacks(qq,gap,delta,details=True)
        if not check['passed']:failure='retreat_collision';break
        rows.append(qq)
        if distance>=minimum_start_distance-1e-8:
            clearances=checker.gripper_clearances(qq,gap,delta)
            # Report and prefer a separated hand; do not hide the actual margin.
            if all(min(clearances[side]['skin_m'],clearances[side]['cloth_m'])>=.02 and clearances[side]['robot_body_m']>=.03 for side in ('left','right')):
                selected=dict(start_q=qq.tolist(),raw_q=[a.tolist() for a in rows[::-1]],distance_m=float(distance),start_clearances=clearances)
                break
    if selected:
        closure=[checker.slacks(q,float(g),delta,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,31)]
        selected.update(passed=all(v['passed'] for v in closure),stage='retreat_route_passed' if all(v['passed'] for v in closure) else 'closure_collision',virtual_closure_checks=closure)
        return selected
    return dict(passed=False,stage=failure or 'no_separated_start',retreat_samples=len(rows),last_distance_m=float((len(rows)-1)*.0025))


def initials(args):
    """Separately establish whether well-separated, noncolliding starts exist."""
    state=json.loads((ROOT/'artifacts/bulged-cuff/pose-10/state.json').read_text());k,checker,dl,du,contract,scene=context(10,state)
    parks_data=json.loads((args.output/'left-parks.json').read_text());pool=parks_data['candidates'];accepted=[];counts=Counter()
    xs=[float((dl[0]+du[0])/2),float(dl[0]),float(du[0])]
    zs=sorted(np.linspace(-.8,.8,17),key=lambda z:(abs(z),z))
    fixed_names=['base_link','torso/link_3','head/link_3'];reference=k.fk(k.start,only=fixed_names)
    for z in zs:
        for y in [0.,float(dl[1]),.1,.2,.3]:
            for x in xs:
                d=np.array([x,y,z])
                checker.park_left(pool[0]['q_rad']);probe=checker.slacks(np.array(pool[0]['q_rad']),.04,d,details=True)
                counts['layouts_tested']+=1
                if any(not any(name.startswith(('left_arm/','right_arm/','left_gripper/','right_gripper/')) for name in v['pair']) for v in probe['issues']):
                    counts['immutable_layout_rejected']+=1
                    continue
                found=False
                for li,left in enumerate(pool):
                    checker.park_left(left['q_rad'])
                    for ri,right in enumerate(pool):
                        q=np.array(right['q_rad']);counts['tested']+=1
                        if not checker.is_clear(q,.04,d):continue
                        counts['collision_free']+=1;clearance=checker.gripper_clearances(q,.04,d)
                        if not all(min(clearance[s]['skin_m'],clearance[s]['cloth_m'])>=.03 and clearance[s]['robot_body_m']>=.03 for s in ('left','right')):continue
                        frames=k.fk(q,only=fixed_names+['left_gripper/grasping_frame'])
                        fixed_error=max(float(np.linalg.norm(frames[n]-reference[n])) for n in fixed_names)
                        foot=np.array(state['source']['surfaces'][0]['vertices'])+d
                        accepted.append(dict(number=10,right_q_rad=q.tolist(),right_q_deg=np.rad2deg(q).tolist(),left_q_rad=left['q_rad'],left_q_deg=left['q_deg'],
                            translation_m=d.tolist(),right_pool_index=ri,left_pool_index=li,right_gap_m=.04,left_gap_m=.04,
                            foot_x_interval_m=[float(foot[:,0].min()),float(foot[:,0].max())],central_contract=contract,
                            right_tcp_world_m=k.pose(q)[0].tolist(),left_grasping_frame_world_m=(k.A@frames['left_gripper/grasping_frame'][:3,3]+k.b).tolist(),
                            gripper_clearances=clearance,maximum_fixed_frame_matrix_difference=fixed_error,
                            torso_joints_rad=checker.reference_torso,check=checker.slacks(q,.04,d,details=True)))
                        print('INITIAL',len(accepted),'delta',d,'clearances',clearance,flush=True)
                        if len(accepted)>=args.park_count:
                            write(args.output/'initial-configurations.json',dict(counts=dict(counts),candidates=accepted,torso_fixed=True,reachability_verified=False,
                                meaning='Collision-free separated starts only; not a claim of grasp reachability'))
                            return
                        found=True
                        break
                    if found:break
    write(args.output/'initial-configurations.json',dict(counts=dict(counts),candidates=accepted,torso_fixed=True,reachability_verified=False))


def search(args):
    refs=json.loads((ROOT/'artifacts/centered-reaching/full-grid/geometry-summary.json').read_text())['accepted']
    parks_data=json.loads((args.output/'left-parks.json').read_text());left_parks=parks_data['candidates'];contexts={}
    for index,ref in enumerate(refs):
        if index%args.shards!=args.shard_index:continue
        path=args.output/'search'/f'target-{index:04d}.json'
        if path.exists():continue
        r=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']];state=state_for(r);n=r['number']
        if n not in contexts:contexts[n]=context(n,state)
        k,checker,dl,du,contract,scene=contexts[n];checker.park_left(left_parks[0]['q_rad'])
        if args.maximum_delta_z is not None:
            du=du.copy();du[2]=min(du[2],args.maximum_delta_z)
            contract=dict(contract,translation_upper_m=du.tolist(),maximum_delta_z=args.maximum_delta_z)
        c=r['candidate'];p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);pad=np.deg2rad(5);lower=np.r_[k.lower+pad,dl];upper=np.r_[k.upper-pad,du]
        rng=np.random.default_rng(20261108+index);attempts=[];began=time.monotonic()
        initial=[np.deg2rad(v) for v in ([160,-60,0,90,0,0,45],[200,-50,60,70,0,-30,45],[125,-60,-30,90,0,0,45])]+[k.start]
        for seed in range(args.seeds):
            q=initial[seed] if seed<len(initial) else rng.uniform(lower[:7],upper[:7]);q=np.clip(q,lower[:7]+1e-9,upper[:7]-1e-9)
            delta=np.clip(k.pose(q)[0]-p,dl+1e-9,du-1e-9);weights=np.r_[np.full(3,5.),np.full(3,.75)]
            fit=least_squares(lambda x:pose_constraint(k,p,R,x)*weights,np.r_[q,delta],jac=lambda x:pose_constraint(k,p,R,x,True)[1]*weights[:,None],bounds=(lower,upper),max_nfev=180,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            x=fit.x;e=pose_constraint(k,p,R,x);a=dict(seed=seed,q=x[:7].tolist(),translation_m=x[7:].tolist(),position_error_m=float(np.linalg.norm(e[:3])),orientation_error_deg=float(np.rad2deg(np.linalg.norm(e[3:]))),stage='ik_failed');attempts.append(a)
            if not good(a):continue
            a['stage']='endpoint_collision';a['park_checks']=[]
            for opening in sorted(r['valid_openings'],key=lambda v:-v['gap_m']):
                gap=opening['gap_m']
                for park_index,park in enumerate(left_parks):
                    checker.park_left(park['q_rad']);check=checker.slacks(x[:7],gap,x[7:],details=True)
                    a['park_checks'].append(dict(park_index=park_index,gap_m=gap,check=check))
                    if not check['passed']:
                        # Left changes cannot repair a pair containing no left link.
                        if any(not any(v.startswith(('left_arm/','left_gripper/')) for v in issue['pair']) for issue in check['issues']):break
                        continue
                    a.update(stage='endpoint_passed',left_park_index=park_index,left_q_rad=park['q_rad'],gap_m=gap)
                    print('ENDPOINT',index,n,'q1',np.rad2deg(x[0]),'delta',x[7:],'left',park_index,flush=True)
                    route=retract_route(k,checker,c,x[:7],x[7:],gap,args.start_distance)
                    a['trajectory']=route
                    print('ROUTE',index,route['stage'],route.get('distance_m'),flush=True)
                    if route['passed']:break
                if a.get('trajectory',{}).get('passed'):break
            if a.get('trajectory',{}).get('passed'):break
        record=dict(index=index,number=n,geometry_reference=ref,geometry_record=r,central_contract=contract,attempts=attempts,
            torso_joints_rad=checker.reference_torso,base_and_head_joints_rad=checker.reference_fixed_joints,
            all_joint1_branches_searched=True,joint_limit_margin_deg=5,left_gap_m=.04,minimum_start_tcp_distance_m=args.start_distance,
            seeds=args.seeds,elapsed_s=time.monotonic()-began,search_geometry=str(FAST_MANIFEST.relative_to(ROOT)),executed=False)
        write(path,record);print('SEARCH',index,len(refs),n,dict(Counter(v['stage'] for v in attempts)),round(record['elapsed_s'],1),flush=True)
        if any(v.get('trajectory',{}).get('passed') for v in attempts) and args.stop_after_route:break
    rows=[json.loads(p.read_text()) for p in (args.output/'search').glob('target-*.json')]
    write(args.output/f'summary-shard-{args.shard_index}.json',dict(targets=len(rows),trials=sum(len(r['attempts']) for r in rows),routes=sum(v.get('trajectory',{}).get('passed',False) for r in rows for v in r['attempts']),executed=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['parks','initials','search']);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');ap.add_argument('--seeds',type=int,default=16);ap.add_argument('--park-count',type=int,default=12);ap.add_argument('--start-distance',type=float,default=.15);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--stop-after-route',action='store_true');ap.add_argument('--maximum-delta-z',type=float);a=ap.parse_args();a.output=a.output.resolve()
    {'parks':parks,'initials':initials,'search':search}[a.stage](a)


if __name__=='__main__':main()
