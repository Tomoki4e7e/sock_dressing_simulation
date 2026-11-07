#!/usr/bin/env python3
"""Reevaluate previous coupled targets and refine new endpoint survivors with na arms."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,copy
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load
from search_compound_reaching import target_data
from apron_reaching_common import overlay_robot,AuditedScene
from sock_dressing_simulation.arm_reaching import Kinematics
OUT=ROOT/'artifacts/apron-reaching'

def old_targets():
 robot=json.loads((OUT/'robot-scene.json').read_text());rows=[]
 for path in sorted((ROOT/'artifacts/compound-reaching/coupled-approach').glob('*.json')):
  r=json.loads(path.read_text());c,state,_=target_data(r);s=overlay_robot(load(r['number']),robot);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);checker=AuditedScene(k,s,state['triangles'],OUT/'decomposition.json',ROOT);d=r['translation_m'];gap=r['gap_m'];start=checker.slacks(k.start,gap,d,details=True);end=checker.slacks(r['q'],gap,d,details=True);pre=checker.slacks(r['trajectory']['approach_q'][0],gap,d,details=True);rows.append(dict(source=str(path.relative_to(ROOT)),start=start,endpoint=end,preapproach=pre,new_start_q=k.start.tolist()))
 (OUT/'old-target-recheck.json').write_text(json.dumps(rows,indent=2));print('old targets',len(rows),flush=True)

def task(arg):
 if '/connected/' in str(arg[0]):
  clone=OUT/'connected-inputs'/('connected-'+Path(arg[0]).name);clone.parent.mkdir(exist_ok=True);clone.write_text(Path(arg[0]).read_text());arg=(str(clone),arg[1])
 destination=OUT/'coupled-approach'/(Path(arg[0]).stem+f'-trial-{arg[1]}.json')
 if destination.exists():return destination.name
 import refine_compound_approach as module
 class ConnectedKinematics(Kinematics):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.upper[0]=min(self.upper[0],np.deg2rad(114.5))
 module.Kinematics=ConnectedKinematics
 robot=json.loads((OUT/'robot-scene.json').read_text());module.OUT=OUT;module.load=lambda n:overlay_robot(load(n),robot);module.CompoundScene=AuditedScene
 return module.task(arg)

def main():
 old_targets();tasks=[]
 for path in sorted(list((OUT/'search').glob('*.json'))+list((OUT/'connected/search').glob('*.json'))):
  row=json.loads(path.read_text())
  for i,a in enumerate(row['attempts']):
   if a['endpoint_and_start_passed'] and not a.get('trajectory',{}).get('passed'):
    if np.rad2deg(a['q'][0])>110:
     continue  # A validated necessary shoulder slice blocks this joint solution.
    tasks.append((str(path),i))
 gates=[]
 for path in sorted((OUT/'search').glob('*.json')):
  row=json.loads(path.read_text())
  for i,a in enumerate(row['attempts']):
   if a['endpoint_and_start_passed']:gates.append(dict(file=str(path.relative_to(ROOT)),trial=i,goal_joint1_deg=float(np.rad2deg(a['q'][0])),pruned_by_shoulder_slice=bool(np.rad2deg(a['q'][0])>110),certificate='artifacts/apron-reaching/shoulder-barrier.json'))
 (OUT/'route-gates.json').write_text(json.dumps(gates,indent=2))
 with ProcessPoolExecutor(3) as pool:list(pool.map(task,tasks))
if __name__=='__main__':main()
