#!/usr/bin/env python3
"""Recheck centered routes with the registered collision shapes and export time profiles."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import state_for,write,URDF,ROBOT,REGISTERED_MANIFEST
from search_translated_reaching import load,margin
from sock_dressing_simulation.arm_reaching import Kinematics,ARM,HANDEDNESS
from sock_dressing_simulation.centered_reaching import placement_bounds
from apron_reaching_common import overlay_robot,AuditedScene,start_margin_transition
from validate_compound_trajectory import smooth_profile


def validate(path,output):
    row=json.loads(path.read_text())
    if not any(a.get('trajectory',{}).get('passed') for a in row['attempts']):
        return []
    registration=REGISTERED_MANIFEST.parent
    native=json.loads((registration/'native-validation/summary.json').read_text())
    payload_hash=hashlib.sha256((registration/'payload.json').read_bytes()).hexdigest()
    if not native['passed'] or native['payload_sha256']!=payload_hash:
        raise ValueError('Registered native geometry validation is missing or stale')
    state=state_for(row['geometry_record'])
    scene=overlay_robot(load(row['number']),json.loads(ROBOT.read_text()))
    scene['particles_world']=state['particles_world']
    k=Kinematics(URDF,scene)
    checker=AuditedScene(k,scene,state['triangles'],REGISTERED_MANIFEST,ROOT)
    dl,du,contract=placement_bounds(scene,state,edge_margin=row['central_contract'].get('edge_margin_m',.01))
    candidates=[]
    for trial,a in enumerate(row['attempts']):
        if not a.get('trajectory',{}).get('passed'):
            continue
        d=np.asarray(a['translation_m']);gap=a['gap_m'];route=a['trajectory']
        raw=np.asarray(route['start_route_q']+route['approach_q'][1:])
        profile=smooth_profile(raw,k.velocity,k.acceleration)
        # Test twice the exported sampling density, including curve midpoints.
        u=np.unique(np.r_[profile['u'],(profile['u'][:-1]+profile['u'][1:])/2])
        q=profile['curve'](u);failures=[];minimum=float('inf')
        points=[]
        for i,angles in enumerate(q):
            check=checker.slacks(angles,gap,d,details=True)
            minimum=min(minimum,check['minimum_slack_m'])
            if not check['passed']:
                failures.append(dict(index=i,check=check))
            if i%100==0:
                print('VALIDATING',path.stem,trial,i,len(q),'slack',minimum,flush=True)
        for angles in profile['q']:
            # Independently use tree FK, not the search's fast pose routine.
            T=k.fk(angles,only=['right_gripper/gripper_base'])['right_gripper/gripper_base']
            points.append((k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b).tolist())
        initial_gap=k.q0['right_gripper/finger_joint']+k.hand.gap_offset_m
        opening=[checker.slacks(k.start,float(g),d,details=True) for g in np.linspace(initial_gap,gap,31)]
        closure=[checker.slacks(q[-1],float(g),d,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,61)]
        c=row['geometry_record']['candidate'];target=np.array(c['grasp_center_unity_world_m'])+d
        T=k.fk(q[-1],only=['right_gripper/gripper_base'])['right_gripper/gripper_base']
        rotation=k.A@T[:3,:3]@HANDEDNESS
        angle=float(np.rad2deg(Rotation.from_matrix(rotation@np.array(c['rotation_matrix']).T).magnitude()))
        error=float(np.linalg.norm(np.array(points[-1])-target))
        central=bool(np.all(d>=dl-1e-12) and np.all(d<=du+1e-12))
        policy=start_margin_transition(q,k.lower,k.upper,k.start)
        vf=float(np.max(profile['velocity_bound']/k.velocity));af=float(np.max(profile['acceleration_bound']/k.acceleration))
        foot=np.asarray(state['source']['surfaces'][0]['vertices'])+d
        passed=bool(central and policy['passed'] and not failures and all(v['passed'] for v in opening+closure)
                    and error<=1e-4 and angle<=.5 and vf<=.1+1e-10 and af<=.1+1e-10)
        prefix=output/'routes'/(path.stem+f'-trial-{trial}')
        trajectory=prefix.with_name(prefix.name+'-trajectory.json')
        plan=dict(source=row['source'],search_record=str(path.relative_to(ROOT)),number=row['number'],
            joint_names=ARM,translation_m=d.tolist(),opening_gap_m=gap,initial_gap_m=initial_gap,virtual_closing_gap_m=.001,
            times_s=profile['t'].tolist(),q_rad=profile['q'].tolist(),velocity_rad_s=profile['velocity'].tolist(),
            acceleration_rad_s2=profile['acceleration'].tolist(),tcp_world_m=points,duration_s=profile['duration'],
            central_contract=contract,foot_x_interval_m=[float(foot[:,0].min()),float(foot[:,0].max())],
            geometry_candidate=c,additional_roll_deg=row['geometry_record']['additional_roll_deg'],
            interpolation='PCHIP joint curve and global quintic time map',
            joint_margin_policy=policy,offline_passed=passed,native_geometry_passed=True,
            geometry_manifest_sha256=hashlib.sha256(REGISTERED_MANIFEST.read_bytes()).hexdigest(),
            robot_scene_sha256=hashlib.sha256(ROBOT.read_bytes()).hexdigest(),
            executed=False,continuous_collision_certified=False)
        write(trajectory,plan)
        result=dict(source=str(path.relative_to(ROOT)),trial=trial,number=row['number'],offline_passed=passed,
            sample_count=len(q),exported_sample_count=len(profile['q']),central_placement_passed=central,
            foot_x_interval_m=plan['foot_x_interval_m'],joint_margin_policy=policy,
            minimum_slack_m=minimum,minimum_joint_margin_deg=min(margin(k,v) for v in q),
            endpoint_position_error_m=error,endpoint_orientation_error_deg=angle,
            maximum_velocity_fraction=vf,maximum_acceleration_fraction=af,
            failures=failures,opening_checks=opening,virtual_closure_checks=closure,
            geometry_manifest_sha256=plan['geometry_manifest_sha256'],trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            frozen_cloth_geometry_only=True,executed=False,continuous_collision_certified=False)
        write(prefix.with_name(prefix.name+'-validation.json'),result)
        with prefix.with_name(prefix.name+'-trajectory.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['time_s']+[n+'_deg' for n in ARM])
            writer.writerows(np.c_[profile['t'],np.rad2deg(profile['q'])])
        candidates.append(dict(trajectory=str(trajectory.relative_to(ROOT)),passed=passed,number=row['number'],duration_s=profile['duration'],samples=len(q)))
        print('VALIDATED',path.stem,trial,passed,'duration',profile['duration'],'error',error,'angle',angle,flush=True)
    return candidates


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/centered-reaching')
    ap.add_argument('--targets',type=int,nargs='+')
    args=ap.parse_args();args.output=args.output.resolve();results=[]
    for path in sorted((args.output/'search').glob('target-*.json')):
        if args.targets is None or int(path.stem.split('-')[1]) in args.targets:
            results+=validate(path,args.output)
    write(args.output/'trajectory-summary.json',dict(routes=results,passing_routes=sum(r['passed'] for r in results),
        registered_geometry=str(REGISTERED_MANIFEST.relative_to(ROOT)),executed=False))
    write(args.output/'trajectory-status.json',dict(generated=any(r['passed'] for r in results),
        reason='Validated centered route exported' if any(r['passed'] for r in results) else 'No centered endpoint and connected approach passed; no accepted trajectory exported',
        candidate_routes=len(results),executed=False,continuous_collision_certified=False))


if __name__=='__main__':
    main()
