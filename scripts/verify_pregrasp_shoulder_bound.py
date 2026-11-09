#!/usr/bin/env python3
"""Supplementary positive clearance bound; does not certify the exact 5 mm policy."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import sys,json,time,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from verify_contact_pregrasp_continuous import make_scene,write,inventory,sha256
from sock_dressing_simulation.continuous_pregrasp import SweptPolicy,trajectory_intervals,certify_interval
OUT=ROOT/'artifacts/contact-pregrasp-verification';SOURCE=ROOT/'artifacts/contact-pregrasp-start-search'
def run(row):
 began=time.monotonic();scene,route,initial=make_scene(row,SOURCE,SweptPolicy(condition_timeout_s=600.))
 existing=OUT/'continuous-shoulder-bound'/f'{row["id"]}.json'
 if existing.exists():
  cached=json.loads(existing.read_text())
  if cached.get('passed') and cached.get('route_sha256')==sha256(SOURCE/'routes'/f'{row["id"]}.json') and cached.get('geometry_digest')==scene.geometry_digest and cached.get('verifier_source_sha256')==sha256(ROOT/'sock_dressing_simulation/continuous_pregrasp.py') and cached.get('supplement_source_sha256')==sha256(Path(__file__)):return cached
 all_margins=row['id']=='E-32'
 padding=scene.base.padding
 scene.pairs=[(a,b,m-1e-6 if all_margins and m-padding.get(a,0.)-padding.get(b,0.)>1e-12 else min(m,.004999)if {a,b}in ({'right_arm/link_2','torso/link_3'},{'left_arm/link_2','torso/link_3'})else m)for a,b,m in scene.pairs]
 intervals=trajectory_intervals(route,initial);records=[]
 for i,interval in enumerate(intervals):
  result=certify_interval(scene,interval['start'],interval['end'],deadline=began+600.);result.update(index=i,stage=interval['stage'],side=interval['side']);records.append(result)
  if result['status']!='certified':break
 result=dict(id=row['id'],status=records[-1]['status'],passed=len(records)==len(intervals)and records[-1]['status']=='certified',minimum_shoulder_clearance_m=.004999,
   exact_5mm_policy_certified=False,scope='Only the shoulder pair is checked at a 4.999 mm positive clearance bound. Other pairs, local-contact allowances and numerical error budget are unchanged. This proves nonintersection while leaving exact 5 mm policy undecided.',
   route_sha256=sha256(SOURCE/'routes'/f'{row["id"]}.json'),geometry_digest=scene.geometry_digest,elapsed_s=time.monotonic()-began,intervals=records,
   verifier_source_sha256=sha256(ROOT/'sock_dressing_simulation/continuous_pregrasp.py'),supplement_source_sha256=sha256(Path(__file__)))
 if all_margins:result['scope']='Every positive nominal safety clearance is checked at its original value minus 1 micrometre; shape-coverage padding and localized contact policy are retained. This is a nonintersection lower-bound certificate, not certification of exact original safety margins.'
 write(OUT/'continuous-shoulder-bound'/f'{row["id"]}.json',result);print('SHOULDER BOUND',row['id'],result['status'],round(result['elapsed_s'],1),flush=True);return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--ids',nargs='*');parser.add_argument('--report-only',action='store_true');args=parser.parse_args();SOURCE=args.source.resolve();OUT=args.output.resolve()
 ids={json.loads(p.read_text())['id']for p in (OUT/'continuous').glob('*.json')if json.loads(p.read_text())['status']=='unresolved'}
 rows=[r for r in inventory(ROOT)if r['id']in ids and(not args.ids or r['id']in args.ids)]
 if not args.report_only:
  with ProcessPoolExecutor(max_workers=6)as pool:
   for f in as_completed([pool.submit(run,r)for r in rows]):f.result()
 results=[json.loads(p.read_text())for p in sorted((OUT/'continuous-shoulder-bound').glob('*.json'))]
 write(OUT/'shoulder-bound-summary.json',dict(completed=len(results),certified=sum(r['passed']for r in results),rows=results))
