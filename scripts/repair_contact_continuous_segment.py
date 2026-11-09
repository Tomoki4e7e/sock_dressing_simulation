#!/usr/bin/env python3
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,time,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from verify_contact_pregrasp_continuous import make_scene
from sock_dressing_simulation.common_opening_approach import inventory,linear_quintic_profile
from sock_dressing_simulation.continuous_pregrasp import SweptPolicy,trajectory_intervals,certify_interval
from search_contact_pregrasp import export_route_csv
from search_centered_reaching import write
source=ROOT/'artifacts/contact-placement-search/skin-mesh';row=next(r for r in inventory(ROOT) if r['id']=='A-37');policy=SweptPolicy(maximum_depth=28,condition_timeout_s=600);scene,route,initial=make_scene(row,source,policy);intervals=trajectory_intervals(route,initial);bad=json.loads((ROOT/'artifacts/contact-pregrasp-verification/continuous/A-37.json').read_text())['intervals'][-1]['index'];interval=intervals[bad];a,b=interval['start'],interval['end'];side=interval['side'];sl=slice(0,7) if side=='left' else slice(7,14);model=scene.models[side];rng=np.random.default_rng(20261109);trials=[];began=time.monotonic();success=None
for scale in [.1,.25,.5,1.,2.,4.]:
 for n in range(35):
  mid=(a+b)/2;offset=rng.normal(0,1,7);offset=offset/np.linalg.norm(offset)*np.deg2rad(scale);mid[sl]+=offset
  if np.any(mid[sl]<model.lower) or np.any(mid[sl]>model.upper):continue
  first=certify_interval(scene,a,mid,deadline=time.monotonic()+3)
  if first['status']!='certified':continue
  second=certify_interval(scene,mid,b,deadline=time.monotonic()+3)
  if second['status']!='certified':continue
  success=(mid,first,second);break
 if success:break
if success:
 mid,first,second=success;found=False
 for segment in route['segments']:
  if segment['side']!=side:continue
  knots=np.array(segment['knots_q_rad'])
  for i in range(len(knots)-1):
   if np.max(abs(knots[i]-a[sl]))<1e-9 and np.max(abs(knots[i+1]-b[sl]))<1e-9:
    knots=np.insert(knots,i+1,mid[sl],axis=0);profile=linear_quintic_profile(knots,model.velocity,model.acceleration);segment.update(raw_q_rad=knots.tolist(),knots_q_rad=profile['knots_q_rad'],segment_durations_s=profile['segment_durations_s'],q_rad=profile['q'].tolist(),times_s=profile['t'].tolist(),velocity_rad_s=profile['velocity'].tolist(),acceleration_rad_s2=profile['acceleration'].tolist(),duration_s=profile['duration']);found=True;break
  if found:break
 if not found:raise ValueError('No exact route interval matched the certificate')
 route['continuous_repair']=dict(original_interval_index=bad,detour_joint_norm_deg=scale,first=first,second=second,elapsed_s=time.monotonic()-began);write(source/'routes/A-37.json',route);export_route_csv(source/'routes/A-37.csv',initial,route);print('REPAIRED',bad,scale,flush=True)
else:print('NO LOCAL DETOUR FOUND',flush=True)
write(source/'repair-A-37.json',dict(repaired=bool(success),elapsed_s=time.monotonic()-began,all_70_verified=False))
