#!/usr/bin/env python3
"""Recompute all rerun FK results and accepted endpoint collisions from fresh na data."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data
from search_translated_reaching import load,margin
from apron_reaching_common import overlay_robot,AuditedScene
from sock_dressing_simulation.arm_reaching import Kinematics,HANDEDNESS
OUT=ROOT/'artifacts/apron-reaching'

def task(path):
 row=json.loads(path.read_text());robot=json.loads((OUT/'robot-scene.json').read_text());c,state,gaps=target_data(row);s=overlay_robot(load(row['number']),robot);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);np.testing.assert_allclose(row['start_q_rad'],k.start,atol=1e-10);assert row['audited_shoulder'];p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);minimum_y=np.array(s['human_skin']['vertices'])[:,1].min();valid=[];checker=None
 for i,a in enumerate(row['attempts']):
  q=np.array(a['q']);d=np.array(a['translation_m']);T=k.fk(q,only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];actual=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b;rot=k.A@T[:3,:3]@HANDEDNESS;error=float(np.linalg.norm(actual-p-d));angle=float(np.rad2deg(Rotation.from_matrix(rot@R.T).magnitude()));assert abs(error-a['position_error_m'])<1e-7 and abs(angle-a['orientation_error_deg'])<1e-5;assert margin(k,q)>=5-1e-6;assert minimum_y+d[1]>=-1e-9
  if a['endpoint_and_start_passed']:
   assert error<=1e-4 and angle<=.5
   if checker is None:checker=AuditedScene(k,s,state['triangles'],ROOT/row['geometry_manifest'],ROOT)
   end=checker.slacks(q,a['gap_m'],d,details=True);start=checker.slacks(k.start,a['gap_m'],d,details=True);assert end['passed'] and start['passed'];valid.append(dict(trial=i,endpoint=end,start=start))
 return dict(file=str(path.relative_to(ROOT)),trials=len(row['attempts']),passed=True,endpoint_rechecks=valid)

def main():
 baseline=list((OUT/'search').glob('*.json'));connected=list((OUT/'connected/search').glob('*.json'));assert len(baseline)==92 and len(connected)==12
 with ProcessPoolExecutor(3) as pool:rows=list(pool.map(task,sorted(baseline+connected)))
 routes=[json.loads(path.read_text()) for path in (OUT/'routes').glob('*-validation.json')];robot=json.loads((OUT/'start-validation.json').read_text());assert robot['passed'];result=dict(passed=all(r['passed'] for r in rows),baseline_targets=len(baseline),connected_targets=len(connected),trials=sum(r['trials'] for r in rows),endpoint_checks=sum(len(r['endpoint_rechecks']) for r in rows),start_validation=robot,results=rows,route_count=len(routes),offline_route_passes=sum(r['offline_passed'] for r in routes),offline_route_numbers=sorted(set(r['number'] for r in routes if r['offline_passed'])),native_geometry_passed=False,robot_motion_executed=False,robot_scene_sha256=hashlib.sha256((OUT/'robot-scene.json').read_bytes()).hexdigest(),note='Explicit 131-degree start is allowed 4-degree joint-4 margin only during inward transition; goal and approach retain 5 degrees. Native PhysX geometry gate remains unresolved.')
 (OUT/'validation.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ['results','start_validation']})
if __name__=='__main__':main()
