#!/usr/bin/env python3
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from verify_contact_pregrasp_continuous import make_scene
from sock_dressing_simulation.continuous_pregrasp import SweptPolicy,trajectory_intervals
from sock_dressing_simulation.common_opening_approach import inventory,sha256
from search_centered_reaching import write
SOURCE=ROOT/'artifacts/contact-placement-search/skin-mesh-far';OUT=ROOT/'artifacts/contact-placement-search/skin-far-verification'
def run(row):
 scene,route,initial=make_scene(row,SOURCE,SweptPolicy());intervals=trajectory_intervals(route,initial);final=intervals[-1]['end'];scene.base.park_left(final[:7],final[14]);checks={}
 for side,sl,i in [('left',slice(0,7),14),('right',slice(7,14),15)]:
  p,R=scene.models[side].pose(final[sl]);c=scene.goal.inspect(p-scene.delta,R,final[i],side,route['final_contact_checks'][side]['cloth_finger']);checks[side]=c
 whole=scene.base.slacks(final[7:14],final[15],scene.delta,details=True)
 result=dict(id=row['id'],passed=whole['passed'] and all(c['passed'] for c in checks.values()),checks=checks,whole_scene_check=whole,route_sha256=sha256(SOURCE/'routes'/f'{row["id"]}.json'),initial_sha256=sha256(SOURCE/'candidate-initial.yaml'),state_sha256=row['state_sha256'],closure_performed=False,scope='Independent final FK and finite-jaw contact/rim/side inspection, including strict whole-scene endpoint margins; commanded geometry, not native drive feedback.')
 write(OUT/'endpoint-audit'/f'{row["id"]}.json',result);print('ENDPOINT AUDIT',row['id'],result['passed'],flush=True);return {k:v for k,v in result.items() if k not in ('checks','whole_scene_check')}
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,r) for r in inventory(ROOT)]):rows.append(f.result());write(OUT/'endpoint-audit-summary.json',dict(expected=70,completed=len(rows),passed=sum(r['passed'] for r in rows),rows=rows))
