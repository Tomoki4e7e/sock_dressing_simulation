#!/usr/bin/env python3
"""Independently validate and time-parameterize apron-start route candidates."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib,copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load,margin
from search_compound_reaching import target_data
from validate_compound_trajectory import smooth_profile
from apron_reaching_common import overlay_robot,AuditedScene,start_margin_transition
from sock_dressing_simulation.arm_reaching import Kinematics,HANDEDNESS,ARM
OUT=ROOT/'artifacts/apron-reaching'

def task(arg):
 source,trial=arg;r=json.loads(Path(source).read_text());a=r['attempts'][trial];c,state,_=target_data(r);scene=overlay_robot(load(r['number']),json.loads((OUT/'robot-scene.json').read_text()));scene['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',scene);checker=AuditedScene(k,scene,state['triangles'],ROOT/'artifacts/compound-reaching/decomposition.json',ROOT);d=np.array(a['translation_m']);gap=a['gap_m'];raw=np.array(a['trajectory']['start_route_q']+a['trajectory']['approach_q'][1:]);smoothed=smooth_profile(raw,k.velocity,k.acceleration);failures=[];minimum=float('inf');joint_min=180.;tcp=[]
 for i,q in enumerate(smoothed['q']):
  check=checker.slacks(q,gap,d,details=True);minimum=min(minimum,check['minimum_slack_m']);joint_min=min(joint_min,margin(k,q));T=k.fk(q,only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];tcp.append((k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b).tolist())
  if not check['passed']:failures.append(dict(index=i,check=check))
  if i%100==0:print('route',r['index'],i,len(smoothed['q']),minimum,flush=True)
 initial_gap=k.q0['right_gripper/finger_joint']+k.hand.gap_offset_m;opening=[checker.slacks(k.start,float(g),d,details=True) for g in np.linspace(initial_gap,gap,21)];closure=[checker.slacks(smoothed['q'][-1],float(g),d,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,31)]
 target=np.array(c['grasp_center_unity_world_m'])+d;end=np.array(tcp[-1]);T=k.fk(smoothed['q'][-1],only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];rot=k.A@T[:3,:3]@HANDEDNESS;angle=float(np.rad2deg(Rotation.from_matrix(rot@np.array(c['rotation_matrix']).T).magnitude()));joint_policy=start_margin_transition(smoothed['q'],k.lower,k.upper,k.start);passed=not failures and joint_policy['passed'] and all(x['passed'] for x in opening+closure) and np.linalg.norm(end-target)<=1e-4 and angle<=.5
 destination=OUT/'routes';destination.mkdir(exist_ok=True);prefix=Path(source).stem+f'-trial-{trial}';native=json.loads((ROOT/'artifacts/compound-reaching/native-validation.json').read_text());result=dict(joint_margin_policy=joint_policy,number=r['number'],source=str(Path(source).relative_to(ROOT)),trial=trial,offline_passed=bool(passed),native_geometry_passed=native['passed'],sample_count=len(smoothed['q']),minimum_margin_slack_m=minimum,minimum_joint_margin_deg=joint_min,endpoint_position_error_m=float(np.linalg.norm(end-target)),endpoint_orientation_error_deg=angle,maximum_velocity_fraction=float(np.max(smoothed['velocity_bound']/k.velocity)),maximum_acceleration_fraction=float(np.max(smoothed['acceleration_bound']/k.acceleration)),failures=failures,opening_checks=opening,virtual_closure_checks=closure,executed=False,continuous_collision_certified=False)
 (destination/(prefix+'-validation.json')).write_text(json.dumps(result,indent=2));plan=dict(joint_margin_policy=joint_policy,source=r['source'],number=r['number'],joint_names=ARM,translation_m=d.tolist(),opening_gap_m=gap,virtual_closing_gap_m=.001,initial_gap_m=initial_gap,times_s=smoothed['t'].tolist(),q_rad=smoothed['q'].tolist(),velocity_rad_s=smoothed['velocity'].tolist(),acceleration_rad_s2=smoothed['acceleration'].tolist(),tcp_world_m=tcp,duration_s=smoothed['duration'],offline_passed=bool(passed),native_geometry_passed=native['passed'],executed=False,continuous_collision_certified=False,robot_scene_sha256=hashlib.sha256((OUT/'robot-scene.json').read_bytes()).hexdigest())
 (destination/(prefix+'-trajectory.json')).write_text(json.dumps(plan,indent=2));print('VALIDATED',r['number'],passed,'samples',len(smoothed['q']),'duration',smoothed['duration'],flush=True);return result

def main():
 existing=OUT/'routes';tasks=[];seen=set()
 for folder in [OUT/'connected/search',OUT/'search']:
  for path in sorted(folder.glob('*.json')):
   row=json.loads(path.read_text())
   for i,a in enumerate(row['attempts']):
    if not a.get('trajectory',{}).get('passed'):continue
    key=(row['source'],round(a['gap_m'],6))
    if key in seen:continue
    seen.add(key)
    if (existing/(path.stem+f'-trial-{i}-validation.json')).exists():continue
    tasks.append((str(path),i))
 with ProcessPoolExecutor(2) as pool:list(pool.map(task,tasks))
if __name__=='__main__':main()
