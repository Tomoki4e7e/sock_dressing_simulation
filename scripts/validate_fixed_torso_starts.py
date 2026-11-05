#!/usr/bin/env python3
"""Validate reselected starts with registered shapes and export smooth trajectories."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_fixed_torso_starts import context
from search_centered_reaching import state_for,write,REGISTERED_MANIFEST,ROBOT
from validate_compound_trajectory import smooth_profile
from sock_dressing_simulation.arm_reaching import ARM,HANDEDNESS
from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM


def validate(path,out,time_scale=1.):
    row=json.loads(path.read_text());attempts=[(i,a) for i,a in enumerate(row['attempts']) if a.get('trajectory',{}).get('passed')]
    if not attempts:return []
    state=state_for(row['geometry_record']);k,checker,dl,du,contract,scene=context(row['number'],state,REGISTERED_MANIFEST)
    registration=REGISTERED_MANIFEST.parent;native=json.loads((registration/'native-validation/summary.json').read_text())
    if not native['passed'] or native['payload_sha256']!=hashlib.sha256((registration/'payload.json').read_bytes()).hexdigest():raise ValueError('Native registration validation is missing or stale')
    fixed_names=['base_link','torso/link_1','torso/link_2','torso/link_3','head/link_1','head/link_2','head/link_3']
    fixed_names=[n for n in fixed_names if k.tree.find(f"link[@name='{n}']") is not None]
    reference=k.fk(k.start,only=fixed_names);results=[]
    for trial,a in attempts:
        checker.park_left(a['left_q_rad'],row['left_gap_m']);raw=np.array(a['trajectory']['raw_q']);k.start=raw[0].copy();d=np.array(a['translation_m']);gap=a['gap_m']
        profile=smooth_profile(raw,k.velocity,k.acceleration)
        profile['t']*=time_scale;profile['duration']*=time_scale;profile['velocity']/=time_scale;profile['acceleration']/=time_scale**2;profile['velocity_bound']/=time_scale;profile['acceleration_bound']/=time_scale**2
        u=np.unique(np.r_[profile['u'],(profile['u'][:-1]+profile['u'][1:])/2]);samples=profile['curve'](u);failures=[];minslack=float('inf');frame_error=0.;frame_angle=0.
        for i,q in enumerate(samples):
            check=checker.slacks(q,gap,d,details=True);minslack=min(minslack,check['minimum_slack_m'])
            if not check['passed']:failures.append(dict(index=i,check=check))
            frames=k.fk(q,only=fixed_names)
            for name in fixed_names:
                frame_error=max(frame_error,float(np.linalg.norm(frames[name][:3,3]-reference[name][:3,3])))
                frame_angle=max(frame_angle,float(Rotation.from_matrix(frames[name][:3,:3]@reference[name][:3,:3].T).magnitude()))
            if i%100==0:print('VALIDATE',path.stem,trial,i,len(samples),minslack,flush=True)
        qmargin=float(np.rad2deg(np.minimum(samples-k.lower,k.upper-samples)).min());left=np.array(a['left_q_rad']);leftmargin=float(np.rad2deg(np.minimum(left-checker.left_lower,checker.left_upper-left)).min())
        opening=[checker.slacks(k.start,float(g),d,details=True) for g in np.linspace(.001,.05,31)]
        closure=[checker.slacks(samples[-1],float(g),d,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,61)]
        c=row['geometry_record']['candidate'];p=np.array(c['grasp_center_unity_world_m'])+d
        tcp=[]
        for q in profile['q']:
            T=k.fk(q,only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];tcp.append((k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b).tolist())
        T=k.fk(samples[-1],only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];R=k.A@T[:3,:3]@HANDEDNESS
        pe=float(np.linalg.norm(np.array(tcp[-1])-p));ae=float(np.rad2deg(Rotation.from_matrix(R@np.array(c['rotation_matrix']).T).magnitude()));central=bool(np.all(d>=dl-1e-9) and np.all(d<=du+1e-9))
        clearance=checker.gripper_clearances(k.start,gap,d);separated=all(min(clearance[s]['skin_m'],clearance[s]['cloth_m'])>=.02-1e-7 and clearance[s]['robot_body_m']>=.03-1e-7 for s in ('left','right'))
        vf=float(np.max(profile['velocity_bound']/k.velocity));af=float(np.max(profile['acceleration_bound']/k.acceleration));distance=float(np.linalg.norm(np.array(tcp[0])-p))
        passed=not failures and central and separated and qmargin>=5-1e-6 and leftmargin>=5-1e-6 and all(v['passed'] for v in opening+closure) and pe<=1e-4 and ae<=.5 and frame_error<1e-12 and frame_angle<1e-12 and vf<=.1+1e-9 and af<=.1+1e-9 and distance>=.15-1e-4
        name=path.parent.name+'-'+path.stem+f'-trial-{trial}';prefix=out/'routes'/name
        plan=dict(number=row['number'],source_record=str(path.relative_to(ROOT)),trial=trial,geometry_record=row['geometry_record'],
            joint_names=ARM,left_joint_names=LEFT_ARM,start_q_rad=k.start.tolist(),left_q_rad=left.tolist(),
            torso_joints_rad=checker.reference_torso,fixed_joint_values_rad=checker.reference_fixed_joints,
            translation_m=d.tolist(),central_contract=contract,opening_gap_m=gap,left_gap_m=row['left_gap_m'],
            times_s=profile['t'].tolist(),q_rad=profile['q'].tolist(),velocity_rad_s=profile['velocity'].tolist(),acceleration_rad_s2=profile['acceleration'].tolist(),
            tcp_world_m=tcp,duration_s=profile['duration'],time_scale=time_scale,raw_q_rad=raw.tolist(),offline_passed=bool(passed),executed=False,continuous_collision_certified=False,
            geometry_manifest_sha256=hashlib.sha256(REGISTERED_MANIFEST.read_bytes()).hexdigest(),reference_robot_scene_sha256=hashlib.sha256(ROBOT.read_bytes()).hexdigest())
        route_path=prefix.with_name(prefix.name+'-trajectory.json');write(route_path,plan)
        result=dict(number=row['number'],offline_passed=bool(passed),source_record=str(path.relative_to(ROOT)),trial=trial,sample_count=len(samples),exported_sample_count=len(profile['q']),minimum_slack_m=minslack,
            minimum_joint_margin_deg=qmargin,left_joint_margin_deg=leftmargin,central_placement_passed=central,start_separated_passed=separated,start_tcp_distance_m=distance,start_gripper_clearances=clearance,
            endpoint_position_error_m=pe,endpoint_orientation_error_deg=ae,maximum_fixed_frame_position_error_m=frame_error,maximum_fixed_frame_orientation_error_rad=frame_angle,
            maximum_velocity_fraction=vf,maximum_acceleration_fraction=af,failures=failures,start_opening_checks=opening,virtual_closure_checks=closure,
            trajectory_sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),geometry_manifest_sha256=plan['geometry_manifest_sha256'],executed=False,continuous_collision_certified=False)
        write(prefix.with_name(prefix.name+'-validation.json'),result)
        with prefix.with_name(prefix.name+'-trajectory.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['time_s']+[n+'_deg' for n in ARM]);w.writerows(np.c_[profile['t'],np.rad2deg(profile['q'])])
        results.append(dict(trajectory=str(route_path.relative_to(ROOT)),passed=bool(passed),number=row['number'],duration_s=profile['duration'],samples=len(samples)));print('VALIDATED',name,passed,flush=True)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');ap.add_argument('--source',type=Path);ap.add_argument('--time-scale',type=float,default=1.);a=ap.parse_args();a.output=a.output.resolve();results=[]
    if a.time_scale<1:ap.error('--time-scale must be at least 1')
    paths=[a.source.resolve()] if a.source else [p for folder in ('search','refined','planned') for p in sorted((a.output/folder).glob('target-*.json'))]
    for path in paths:results+=validate(path,a.output,a.time_scale)
    write(a.output/'trajectory-status.json',dict(generated=any(v['passed'] for v in results),routes=results,torso_fixed=True,executed=False,
        reason='Registered-geometry trajectory validation completed' if results else 'No full-condition endpoint and connected route passed; no accepted trajectory exported'))


if __name__=='__main__':main()
