#!/usr/bin/env python3
"""Independently recompute FK, floor constraints and retained feasible states."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data
from search_translated_reaching import load,margin
from sock_dressing_simulation.arm_reaching import Kinematics,HANDEDNESS
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def main():
 k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',load(39));counts=Counter();groups={};floor_rejections=[];accepted=[]
 for folder in ['search','expanded/search','floor-aware/search']:
  group=Counter()
  for path in sorted((OUT/folder).glob('target-*.json')):
   row=json.loads(path.read_text());c,state,gaps=target_data(row);p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);native=load(row['number']);minimum_human_y=float(np.array(native['human_skin']['vertices'])[:,1].min());group['targets']+=1
   for i,a in enumerate(row['attempts']):
    q=np.array(a['q']);d=np.array(a['translation_m']);T=k.fk(q,only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];actual=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b;rot=k.A@T[:3,:3]@HANDEDNESS;error=np.linalg.norm(actual-p-d);angle=np.rad2deg(Rotation.from_matrix(rot@R.T).magnitude());assert abs(error-a['position_error_m'])<1e-7 and abs(angle-a['orientation_error_deg'])<1e-5;assert margin(k,q)>=5-1e-6;group['trials']+=1
    floor_clear=minimum_human_y+d[1]>=-1e-9
    if row.get('floor_constraint_included'):assert floor_clear
    if a['endpoint_and_start_passed']:
     group['reported_endpoint_passes']+=1
     if not floor_clear:floor_rejections.append(dict(file=str(path.relative_to(ROOT)),trial=i,minimum_human_y_m=minimum_human_y+d[1]));continue
     group['floor_clear_endpoint_passes']+=1
     if folder=='floor-aware/search':accepted.append((path,row,i,a,c,state))
  groups[folder]=dict(group);counts.update(group)
 endpoint_checks=[]
 for path,row,i,a,c,state in accepted:
  native=load(row['number']);native['particles_world']=state['particles_world'];checker=CompoundScene(k,native,state['triangles'],OUT/'decomposition.json',ROOT);endpoint=checker.slacks(a['q'],a['gap_m'],a['translation_m'],details=True);start=checker.slacks(k.start,a['gap_m'],a['translation_m'],details=True);assert endpoint['passed'] and start['passed'];endpoint_checks.append(dict(file=str(path.relative_to(ROOT)),trial=i,endpoint=endpoint,start=start))
 coupled=[]
 for path in sorted((OUT/'coupled-approach').glob('*.json')):
  row=json.loads(path.read_text());c,state,_=target_data(row);native=load(row['number']);native['particles_world']=state['particles_world'];checker=CompoundScene(k,native,state['triangles'],OUT/'decomposition.json',ROOT);checks=[]
  for q in row.get('trajectory',{}).get('approach_q',[]):
   check=checker.slacks(q,row['gap_m'],row['translation_m'],details=True);assert check['passed'] and margin(k,np.array(q))>=5-1e-6;checks.append(check['minimum_slack_m'])
  coupled.append(dict(file=str(path.relative_to(ROOT)),approach_samples=len(checks),minimum_approach_slack_m=min(checks) if checks else None))
 native=json.loads((OUT/'native-validation.json').read_text());geometry=json.loads((OUT/'decomposition.json').read_text());hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'decomposition.json']+[ROOT/r['parts_file'] for r in geometry['parts']]};rrt=[json.loads(p.read_text()) for p in (OUT/'rrt').glob('*.json')];opt=[json.loads(p.read_text()) for p in (OUT/'optimized-route').glob('*.json')]
 result=dict(passed=True,native_geometry_passed=native['passed'],groups=groups,fk_trials_recomputed=counts['trials'],invalidated_floor_unchecked_candidates=floor_rejections,endpoint_rechecks=endpoint_checks,coupled_approach_rechecks=coupled,rrt_runs=len(rrt),rrt_iterations=sum(r['iterations'] for r in rrt),rrt_successes=sum(r['found'] for r in rrt),optimized_route_successes=sum(r['found'] for r in opt),geometry_hashes=hashes,robot_motion_executed=False,continuous_collision_certified=False)
 (OUT/'validation.json').write_text(json.dumps(result,indent=2));print('FK trials',counts['trials'],'floor-aware endpoints',len(endpoint_checks),'native',native['passed'],'rrt successes',result['rrt_successes'])
if __name__=='__main__':main()
