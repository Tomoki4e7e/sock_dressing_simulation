#!/usr/bin/env python3
"""Finite rigid human/chair placement screening, not continuous certification."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,sys,time,copy,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_contact_pregrasp import contact_scene
from search_fixed_torso_starts import context
from search_centered_reaching import REGISTERED_MANIFEST,write
from sock_dressing_simulation.common_opening_approach import inventory,load_state,own_arm_issues
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene
from sock_dressing_simulation.contact_pregrasp import ContactGoal

def job(payload):
 index,delta,ids,out=payload;delta=np.array(delta);start=time.monotonic()
 ref=yaml.safe_load((ROOT/'artifacts/contact-pregrasp-start-search/candidate-initial.yaml').read_text());results=[]
 for row in [r for r in inventory(ROOT) if r['id'] in ids]:
  state=load_state(ROOT,row);k,base,lo,hi,contract,_=context(row['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=ref['left_foot_floor_allowance_m']);base.use_spatial_aabb=True
  if delta[1]<lo[1]-1e-9:results.append(dict(id=row['id'],passed=False,reason='floor'));continue
  goal=ContactGoal(state,k.hand);original=list(base.pairs);found={};source=json.loads((ROOT/'artifacts/contact-pregrasp-start-search/endpoints'/f'{row["id"]}.json').read_text())
  for side in ('left','right'):
   model=ControlledArmKinematics(k,side,ref['right_q_rad'],ref['right_gap_m']);comfortable=copy.copy(model);comfortable.lower=model.lower+np.deg2rad(.5);comfortable.upper=model.upper-np.deg2rad(.5)
   base.park_left(ref['left_q_rad'],ref['left_gap_m']);checker=ControlledCollisionScene(base,side,ref['right_q_rad'],ref['right_gap_m']);accepted=[]
   for old in source['sides'][side]['accepted'][:6]:
    c=old['contact_check'];sol=comfortable.solve(np.array(c['position_m'])+delta,np.array(c['rotation']),np.clip(old['q_rad'],comfortable.lower+1e-8,comfortable.upper-1e-8))
    if sol['position_error_m']>1e-5 or sol['orientation_error_deg']>.05:continue
    q=sol['q'];actual,rot=model.pose(q);check=goal.inspect(actual-delta,rot,old['opening_gap_m'],side,c['cloth_finger'])
    if not check['passed']:continue
    base.pairs=list(original);contact_scene(base,{side:check['foot_finger']});whole=checker.slacks(q,old['opening_gap_m'],delta,details=True)
    if own_arm_issues(whole,side):continue
    accepted.append(dict(q_rad=np.asarray(q).tolist(),opening_gap_m=old['opening_gap_m'],contact_check=check,tcp_world_m=actual.tolist(),rotation=rot.tolist(),whole_scene_check=whole))
    if len(accepted)>=2:break
   found[side]=accepted
  paired=[]
  for left,right in itertools.product(found['left'],found['right']):
   base.pairs=list(original);contact_scene(base,{s:e['contact_check']['foot_finger'] for s,e in [('left',left),('right',right)]});base.park_left(left['q_rad'],left['opening_gap_m']);check=base.slacks(right['q_rad'],right['opening_gap_m'],delta,details=True)
   if check['passed']:paired.append(dict(left=left,right=right,check=check))
  results.append(dict(id=row['id'],passed=bool(paired),sides=found,pairs=paired))
 result=dict(index=index,translation_m=delta.tolist(),passed_states=sum(r['passed'] for r in results),evaluated_states=len(results),rows=results,elapsed_s=time.monotonic()-start,scope='Static IK/contact screening; old collision model; no continuous or drive certificate.')
 write(out/'placements'/f'{index:03d}.json',result);print('PLACEMENT',index,delta.tolist(),result['passed_states'],len(results),round(result['elapsed_s'],1),flush=True);return {k:v for k,v in result.items() if k!='rows'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-placement-search');p.add_argument('--workers',type=int,default=6);p.add_argument('--ids',nargs='+',default=['A-10','B-19','C-47','D-34','E-14','E-50']);p.add_argument('--all-for',type=int);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
 if a.all_for is None:
  deltas=list(itertools.product([-.0842748674,-.0342748674,.0157251326],[-.12,-.06,0.],[-.4,-.3,-.2,-.1,0.]));jobs=[(i,d,a.ids,out) for i,d in enumerate(deltas)]
 else:
  old=json.loads((out/'placements'/f'{a.all_for:03d}.json').read_text());jobs=[(a.all_for,old['translation_m'],[r['id'] for r in inventory(ROOT)],out)]
 results=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(job,j) for j in jobs]):
   results.append(f.result());write(out/'placement-ranking.json',sorted(results,key=lambda r:(-r['passed_states'],r['elapsed_s'])))
if __name__=='__main__':main()
