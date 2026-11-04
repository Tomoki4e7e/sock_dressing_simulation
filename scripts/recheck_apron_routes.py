#!/usr/bin/env python3
"""Recheck saved apron routes and PCHIP interval midpoints without overwriting them."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load
from search_compound_reaching import target_data
from validate_compound_trajectory import smooth_profile
from apron_reaching_common import overlay_robot,AuditedScene,start_margin_transition
from sock_dressing_simulation.arm_reaching import Kinematics,HANDEDNESS


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--numbers',type=int,nargs='+',default=[10,38,50]);ap.add_argument('--subdivisions',type=int,default=2);args=ap.parse_args()
    if args.subdivisions<1:ap.error('Subdivisions must be positive')
    out=ROOT/'artifacts/compound-reaching/followup/route-recheck';out.mkdir(parents=True,exist_ok=True)
    robot_path=ROOT/'artifacts/apron-reaching/robot-scene.json';manifest=ROOT/'artifacts/compound-reaching/decomposition.json';reports=[]
    for path in sorted((ROOT/'artifacts/apron-reaching/routes').glob('*-trajectory.json')):
        saved=json.loads(path.read_text())
        if not saved['offline_passed'] or saved['number'] not in args.numbers:continue
        validation_path=path.with_name(path.name.replace('-trajectory.json','-validation.json'));old=json.loads(validation_path.read_text());source=ROOT/old['source'];row=json.loads(source.read_text());attempt=row['attempts'][old['trial']];c,state,_=target_data(row)
        scene=overlay_robot(load(row['number']),json.loads(robot_path.read_text()));scene['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',scene);checker=AuditedScene(k,scene,state['triangles'],manifest,ROOT)
        raw=np.array(attempt['trajectory']['start_route_q']+attempt['trajectory']['approach_q'][1:]);smoothed=smooth_profile(raw,k.velocity,k.acceleration)
        assert digest(robot_path)==saved['robot_scene_sha256'];np.testing.assert_allclose(smoothed['q'],saved['q_rad'],rtol=0,atol=1e-12)
        u=np.concatenate([np.linspace(a,b,args.subdivisions+1)[:-1] for a,b in zip(smoothed['u'][:-1],smoothed['u'][1:])]+[np.array([1.])]);angles=smoothed['curve'](u);gap=saved['opening_gap_m'];delta=np.array(saved['translation_m']);minimum=np.inf;failures=[];began=time.monotonic()
        for i,q in enumerate(angles):
            check=checker.slacks(q,gap,delta,details=True);minimum=min(minimum,check['minimum_slack_m'])
            if not check['passed']:failures.append(dict(index=i,u=float(u[i]),check=check))
            if i%200==0:print('recheck',row['number'],i,len(angles),'slack',minimum,flush=True)
        opening=[checker.slacks(k.start,float(g),delta,details=True) for g in np.linspace(saved['initial_gap_m'],gap,41)]
        closure=[checker.slacks(angles[-1],float(g),delta,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,61)]
        T=k.fk(angles[-1],only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];end=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b;rot=k.A@T[:3,:3]@HANDEDNESS;position_error=float(np.linalg.norm(end-(np.array(c['grasp_center_unity_world_m'])+delta)));angle_error=float(np.rad2deg(Rotation.from_matrix(rot@np.array(c['rotation_matrix']).T).magnitude()));policy=start_margin_transition(angles,k.lower,k.upper,k.start)
        velocity=float(np.max(smoothed['velocity_bound']/k.velocity));acceleration=float(np.max(smoothed['acceleration_bound']/k.acceleration));passed=bool(not failures and policy['passed'] and all(r['passed'] for r in opening+closure) and position_error<=1e-4 and angle_error<=.5 and velocity<=.1 and acceleration<=.1)
        result=dict(number=row['number'],offline_passed=passed,source_trajectory=str(path.relative_to(ROOT)),source_sha256=digest(path),robot_scene_sha256=digest(robot_path),geometry_manifest_sha256=digest(manifest),sample_count=len(angles),original_sample_count=len(saved['q_rad']),minimum_slack_m=float(minimum),endpoint_position_error_m=position_error,endpoint_orientation_error_deg=angle_error,maximum_velocity_fraction=velocity,maximum_acceleration_fraction=acceleration,joint_margin_policy=policy,failures=failures,opening_checks=opening,virtual_closure_checks=closure,elapsed_s=time.monotonic()-began,executed=False,continuous_collision_certified=False)
        (out/f'pose-{row["number"]:02d}.json').write_text(json.dumps(result,indent=2)+'\n');reports.append(result);print('RECHECKED',row['number'],passed,len(angles),flush=True)
    summary=dict(requested_numbers=args.numbers,validated_numbers=[r['number'] for r in reports],offline_passed=len(reports)==len(args.numbers) and all(r['offline_passed'] for r in reports),routes=reports,executed=False,continuous_collision_certified=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
