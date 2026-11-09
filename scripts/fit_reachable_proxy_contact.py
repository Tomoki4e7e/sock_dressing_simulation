#!/usr/bin/env python3
"""Jointly fit arm joints, open gap and human translation to contact surfaces."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,time,yaml
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_proxy_contact_geometry import ProxyGoal
from search_fixed_torso_starts import context
from search_contact_pregrasp import contact_scene,retain_blocking_pairs
from search_centered_reaching import write,REGISTERED_MANIFEST
from sock_dressing_simulation.common_opening_approach import inventory,load_state,own_arm_issues
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene
from sock_dressing_simulation.planning_human_colliders import augmented_scene
OUT=ROOT/'artifacts/contact-placement-search'
row=next(r for r in inventory(ROOT) if r['id']=='D-01');initial=yaml.safe_load((OUT/'lower-candidates/019/candidate-initial.yaml').read_text());state=load_state(ROOT,row);manifest=initial['human_collider_manifest'];k,base,*_=context(row['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=.15,human_collider_manifest=manifest);base.use_spatial_aabb=True;base.park_left(initial['left_q_rad'],.04);model=ControlledArmKinematics(k,'right',initial['right_q_rad'],.04);checker=ControlledCollisionScene(base,'right',initial['right_q_rad'],.04);goal=ProxyGoal(state,k.hand,augmented_scene(dict(obstacles=[]),ROOT,manifest,[row['scene_number']])['obstacles']);pairs=list(base.pairs);shoulders=dict(base.shoulder_pairs)
old=json.loads((ROOT/'artifacts/contact-pregrasp-start-search/endpoints/D-01.json').read_text())['sides']['right']['accepted'];lower=np.r_[model.lower+np.deg2rad(1),.0001,[-.4,-.15,-.8]];upper=np.r_[model.upper-np.deg2rad(1),.05,[.4,.3,.4]];accepted=[];attempts=[];began=time.monotonic();rng=np.random.default_rng(773)
class Deadline(Exception):pass
for index in range(100):
 if time.monotonic()-began>240:break
 seed=old[index%len(old)];finger=seed['contact_check']['cloth_finger'];q=np.array(seed['q_rad']);q+=rng.normal(0,0 if index<len(old) else .5,7);x=np.clip(np.r_[q,seed['opening_gap_m'],[-.03427486741542815,-.12,-.3]],lower+1e-8,upper-1e-8);deadline=min(began+240,time.monotonic()+20)
 def residual(x):
  if time.monotonic()>deadline:raise Deadline
  p,R=model.pose(x[:7]);return goal.residual(p-x[8:],R,x[7],'right',finger)
 base.pairs=list(pairs);base.shoulder_pairs=dict(shoulders);base.fixed_pair_cache={};base.fixed_pair_valid_cache={}
 try:
  fit=least_squares(residual,x,bounds=(lower,upper),max_nfev=250,diff_step=2e-5,x_scale=np.r_[np.ones(7),.02,[.05,.05,.05]],ftol=1e-8,xtol=1e-8,gtol=1e-6);x=fit.x
  p,R=model.pose(x[:7]);check=goal.inspect(p-x[8:],R,x[7],'right',finger)
  if not check['passed']:attempts.append(dict(index=index,stage='contact',residual=float(np.linalg.norm(fit.fun))));continue
  contact_scene(base,{'right':check['foot_finger']});full_pairs=list(base.pairs);full_shoulders=dict(base.shoulder_pairs);whole=checker.slacks(x[:7],x[7],x[8:],details=True);issues=own_arm_issues(whole,'right')
  if issues:
   retain_blocking_pairs(base,full_pairs,full_shoulders,{tuple(v['pair']) for v in issues});deadline=min(began+240,time.monotonic()+20)
   def repair(x):return np.r_[residual(x),10000*np.minimum(checker.slacks(x[:7],x[7],x[8:])-.00005,0)]
   try:x=least_squares(repair,x,bounds=(lower,upper),max_nfev=100,diff_step=2e-5).x
   finally:base.pairs=full_pairs;base.shoulder_pairs=full_shoulders;base.fixed_pair_cache={};base.fixed_pair_valid_cache={}
   p,R=model.pose(x[:7]);check=goal.inspect(p-x[8:],R,x[7],'right',finger);whole=checker.slacks(x[:7],x[7],x[8:],details=True);issues=own_arm_issues(whole,'right')
  if check['passed'] and not issues:
   accepted.append(dict(q_rad=x[:7].tolist(),opening_gap_m=float(x[7]),translation_m=x[8:].tolist(),contact_check=check,whole_scene_check=whole,tcp_world_m=p.tolist(),rotation=R.tolist()));print('REACHABLE CONTACT',x[8:].tolist(),flush=True)
  else:attempts.append(dict(index=index,stage='collision',issues=issues,contact_passed=check['passed']))
 except Deadline:attempts.append(dict(index=index,stage='deadline'))
 finally:write(OUT/'joint-placement-D-01.json',dict(accepted=accepted,attempts=attempts,elapsed_s=time.monotonic()-began,all_70_verified=False))
 if len(accepted)>=4:break
