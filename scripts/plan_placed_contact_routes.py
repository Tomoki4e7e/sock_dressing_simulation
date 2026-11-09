#!/usr/bin/env python3
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,sys,yaml,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_contact_pregrasp import route_one
from sock_dressing_simulation.common_opening_approach import inventory
from search_centered_reaching import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--ids',nargs='+');p.add_argument('--seconds',type=float,default=60);p.add_argument('--workers',type=int,default=2);a=p.parse_args();source=a.source.resolve();path=source/'candidate-initial.yaml';initial=yaml.safe_load(path.read_text());pair=dict(index=initial['source_candidate_pair_index'],**{s:dict(q_rad=initial[s+'_q_rad']) for s in ('left','right')});jobs=[]
 for r in inventory(ROOT):
  if a.ids and r['id'] not in a.ids:continue
  ep=source/'endpoints'/f'{r["id"]}.json'
  if ep.exists() and all(json.loads(ep.read_text())['sides'].get(s,{}).get('accepted') for s in ('left','right')):jobs.append((r,source,path,pair,a.seconds,1.))
 results=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(route_one,j) for j in jobs]):
   result=f.result();results.append(result);(source/'routes').mkdir(exist_ok=True);src=source/'pair-routes'/f'pair-{pair["index"]:02d}'/f'{result["id"]}.json';shutil.copy2(src,source/'routes'/src.name)
   if src.with_suffix('.csv').exists():shutil.copy2(src.with_suffix('.csv'),source/'routes'/src.with_suffix('.csv').name)
   write(source/'route-summary.json',dict(rows=results,all_70_verified=False));print('PLACED ROUTE',result,flush=True)
if __name__=='__main__':main()
