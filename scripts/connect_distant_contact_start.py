#!/usr/bin/env python3
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,yaml,copy,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.common_opening_approach import union_scene,inventory,load_state,linear_quintic_profile,edge_samples
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene
from plan_fixed_torso_route import rrt
from search_contact_pregrasp import export_route_csv
from search_centered_reaching import write
out=ROOT/'artifacts/contact-placement-search';source=out/'skin-mesh';dest=out/'skin-mesh-far';dest.mkdir(exist_ok=True);initial=yaml.safe_load((source/'candidate-initial.yaml').read_text());rows=inventory(ROOT);k,scene,state=union_scene(ROOT,rows,initial);scene.use_spatial_aabb=True;delta=np.array(initial['translation_m']);pools=json.loads((out/'proxy-goals/additional-start-pairs.json').read_text())['pools'];foot=np.concatenate([load_state(ROOT,r)['source']['surfaces'][0]['vertices'] for r in rows])+delta;lo,hi=foot.min(0),foot.max(0)
def distance(e):
 p=np.array(e['position_m']);return float(np.linalg.norm(np.maximum(np.maximum(lo-p,p-hi),0)))
candidates=[]
for left,right in itertools.product(pools['left'],pools['right']):
 if not(right['position_m'][0]<lo[0] and left['position_m'][0]>hi[0]):continue
 if min(distance(left),distance(right))<.15:continue
 if (right['q_rad'][0]-np.deg2rad(110))*(initial['right_q_rad'][0]-np.deg2rad(110))<0:continue
 score=sum(np.linalg.norm(np.array(e['q_rad'])-initial[s+'_q_rad']) for s,e in [('left',left),('right',right)]);candidates.append((score,left,right))
candidates.sort(key=lambda t:t[0]);found=None
for score,left,right in candidates[:6]:
 scene.park_left(left['q_rad'],.04);check=scene.slacks(right['q_rad'],.04,delta,details=True)
 if not check['passed']:continue
 for order in [('left','right'),('right','left')]:
  current={s:np.array(e['q_rad']) for s,e in [('left',left),('right',right)]};segments=[]
  for side in order:
   scene.park_left(current['left'],.04);model=ControlledArmKinematics(k,side,current['right'],.04);checker=ControlledCollisionScene(scene,side,current['right'],.04);target=np.array(initial[side+'_q_rad'])
   result=rrt(model,checker,current[side],target,.04,delta,timeout_s=30,joint_margin_deg=.5,shortcut_iterations=10)
   if not result['passed']:break
   # RRT dense output follows exact straight edges; retain only corner knots.
   from sock_dressing_simulation.common_opening_approach import piecewise_knots
   q=np.array(result['raw_q']);q=q[piecewise_knots(q)];profile=linear_quintic_profile(q,model.velocity,model.acceleration)
   segments.append(dict(side=side,q_rad=profile['q'].tolist(),times_s=profile['t'].tolist(),duration_s=profile['duration'],knots_q_rad=profile['knots_q_rad'],segment_durations_s=profile['segment_durations_s']));current[side]=target
  if len(segments)==2:found=(left,right,segments,check);break
 if found:break
if not found:write(dest/'bridge-search.json',dict(passed=False,considered=min(len(candidates),6)));print('NO BRIDGE',flush=True);sys.exit(0)
left,right,segments,check=found;config=copy.deepcopy(initial)
for s,e in [('left',left),('right',right)]:
 config[s+'_q_rad']=e['q_rad'];config[s+'_q_deg']=np.rad2deg(e['q_rad']).tolist();config[s+'_tcp_world_m']=e['position_m'];config[s+'_rotation']=e['rotation']
config.update(source_candidate_pair_index=9001,reachability_verified=False,verified_paired_state_count=0,native_dynamics_verified=False,continuous_collision_certified=False,minimum_joint_margin_deg=min(left['minimum_joint_margin_deg'],right['minimum_joint_margin_deg']),initial_foot_bounds_world_m=[lo.tolist(),hi.tolist()],tcp_to_foot_aabb_lower_bound_m=dict(left=distance(left),right=distance(right)),collision_model='Actual saved human skin meshes; auxiliary Unity capsules excluded; joint-drive validation not claimed.')
(dest/'candidate-initial.yaml').write_text(yaml.safe_dump(config,allow_unicode=True));write(dest/'bridge-search.json',dict(passed=True,segments=segments,all_70_initial_union_check=check,continuous_collision_certified=False))
for row in rows:
 route=json.loads((source/'routes'/f'{row["id"]}.json').read_text());route.update(pair_index=9001,initial_left_q_rad=left['q_rad'],initial_right_q_rad=right['q_rad'],initial_transition_segments=segments+route.get('initial_transition_segments',[]),initial_transition_all_70_union_checked=True)
 write(dest/'routes'/f'{row["id"]}.json',route);export_route_csv(dest/'routes'/f'{row["id"]}.csv',config,route)
print('BRIDGE FOUND',config['left_tcp_world_m'],config['right_tcp_world_m'],config['tcp_to_foot_aabb_lower_bound_m'],flush=True)
