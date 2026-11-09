#!/usr/bin/env python3
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,yaml,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.common_opening_approach import union_scene,inventory,load_state
from search_centered_reaching import write
out=ROOT/'artifacts/contact-placement-search';initial=yaml.safe_load((out/'reference-initial.yaml').read_text());pools=json.loads((out/'proxy-goals/additional-start-pairs.json').read_text())['pools'];rows=inventory(ROOT);k,s,state=union_scene(ROOT,rows,initial);s.use_spatial_aabb=True
foot=np.concatenate([load_state(ROOT,r)['source']['surfaces'][0]['vertices'] for r in rows]);pairs=sorted(itertools.product(pools['left'],pools['right']),key=lambda p:-min(e['minimum_joint_margin_deg'] for e in p));accepted=[];rejected=[]
for delta in itertools.product([-.0342748674,.0157251326,.0657251326],[-.12,-.09,-.06],[-.5,-.45,-.4,-.35,-.3]):
 bounds=[foot.min(0)+delta,foot.max(0)+delta];last=None
 for left,right in pairs[:6]:
  if not(right['position_m'][0]<bounds[0][0] and left['position_m'][0]>bounds[1][0]):continue
  s.park_left(left['q_rad'],.04);check=s.slacks(right['q_rad'],.04,delta,details=True);last=check
  if check['passed']:
   lower_distance=lambda p:float(np.linalg.norm(np.maximum(np.maximum(bounds[0]-p,p-bounds[1]),0)))
   accepted.append(dict(index=len(accepted),translation_m=list(delta),left=left,right=right,foot_bounds_world_m=[b.tolist() for b in bounds],tcp_to_foot_aabb_lower_bound_m={side:lower_distance(np.array(e['position_m'])) for side,e in [('left',left),('right',right)]},all_70_initial_union_check=check));print('PLACED INITIAL',len(accepted),delta,flush=True);break
 if last and not last['passed']:rejected.append(dict(translation_m=list(delta),check=last))
 write(out/'lower-placed-initials.json',dict(accepted=accepted,rejected=rejected,all_70_verified=False,scope='Initial geometry only, including the union of 70 cloth/foot states and active human colliders.'))
print('ACCEPTED',len(accepted),flush=True)
