#!/usr/bin/env python3
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,yaml,itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_fixed_torso_starts import context
from search_contact_pregrasp import contact_scene
from search_centered_reaching import write,REGISTERED_MANIFEST
from sock_dressing_simulation.common_opening_approach import inventory,load_state,own_arm_issues
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene
OUT=ROOT/'artifacts/contact-placement-search'
def job(row):
 initial=yaml.safe_load((OUT/'lower-candidates/019/candidate-initial.yaml').read_text());state=load_state(ROOT,row);k,base,*_=context(row['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=.15,human_collider_manifest=initial['human_collider_manifest']);base.use_spatial_aabb=True;base.park_left(initial['left_q_rad'],.04);model=ControlledArmKinematics(k,'right',initial['right_q_rad'],.04);checker=ControlledCollisionScene(base,'right',initial['right_q_rad'],.04);original=list(base.pairs)
 lower=np.r_[model.lower+np.deg2rad(1),[-.4,-.15,-.8]];upper=np.r_[model.upper-np.deg2rad(1),[.4,.3,.4]];rng=np.random.default_rng(row['number']+742);old=json.loads((ROOT/'artifacts/contact-pregrasp-start-search/endpoints'/f'{row["id"]}.json').read_text());seeds=[e['q_rad'] for e in old['sides']['right']['accepted']];seeds.extend(rng.uniform(lower[:7],upper[:7],(40,7)));goals=json.loads((OUT/'proxy-goals/geometry'/f'{row["id"]}.json').read_text())['sides']['right']['accepted'];accepted=[];tested=0;counts=Counter();examples=[]
 for c,seed in itertools.product(goals,seeds):
  p=np.array(c['position_m']);R=np.array(c['rotation']);gap=c['opening_gap_m'];base.pairs=list(original);contact_scene(base,{'right':c['foot_finger']})
  def residual(x):
   actual,rot=model.pose(x[:7]);return np.r_[1000*(actual-p-x[7:]),100*Rotation.from_matrix(rot@R.T).as_rotvec()]
  fit=least_squares(residual,np.clip(np.r_[seed,initial['translation_m']],lower+1e-8,upper-1e-8),bounds=(lower,upper),max_nfev=140,ftol=1e-9,xtol=1e-9,gtol=1e-7);tested+=1
  if np.linalg.norm(fit.fun)>1e-3:counts['pose_fit_failed']+=1;continue
  q,delta=fit.x[:7],fit.x[7:];check=checker.slacks(q,gap,delta,details=True)
  issues=own_arm_issues(check,'right')
  if issues:
   counts.update(str(v['pair']) for v in issues)
   if len(examples)<5:examples.append(dict(translation_m=delta.tolist(),q_rad=q.tolist(),issues=issues))
   continue
  accepted.append(dict(q_rad=q.tolist(),translation_m=delta.tolist(),contact_check=c,whole_scene_check=check,residual=float(np.linalg.norm(fit.fun))))
 write(OUT/'fitted-placements'/f'{row["id"]}.json',dict(id=row['id'],tested=tested,accepted=accepted,counts=dict(counts),examples=examples));print('FIT',row['id'],len(accepted),tested,flush=True)
if __name__=='__main__':
 rows=[r for r in inventory(ROOT) if r['id'] in ['D-01','E-14','E-50']]
 with ProcessPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(job,r) for r in rows]):f.result()
