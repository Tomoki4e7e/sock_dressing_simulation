#!/usr/bin/env python3
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,sys,yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_contact_pregrasp import endpoint_one,protocol
from search_centered_reaching import write
from sock_dressing_simulation.common_opening_approach import inventory

def main():
 p=argparse.ArgumentParser();p.add_argument('--placement-file',default='placed-initials.json');p.add_argument('--candidate-root',default='candidates');p.add_argument('--candidates',nargs='+',type=int,default=[4,16,28]);p.add_argument('--ids',nargs='+',default=['D-01','E-14','E-50']);p.add_argument('--all',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--seconds',type=float,default=45);p.add_argument('--workers',type=int,default=6);a=p.parse_args();out=ROOT/'artifacts/contact-placement-search';accepted=json.loads((out/a.placement_file).read_text())['accepted'];reference=yaml.safe_load((out/'reference-initial.yaml').read_text());jobs=[]
 for index in a.candidates:
  pair=accepted[index];dest=out/a.candidate_root/f'{index:03d}';dest.mkdir(parents=True,exist_ok=True);initial=dict(reference,translation_m=pair['translation_m'],source_candidate_pair_index=index,source_reference='artifacts/contact-placement-search/'+a.placement_file,verified_paired_state_count=0,reachability_verified=False,all_70_initial_union_check=pair['all_70_initial_union_check'],initial_foot_bounds_world_m=pair['foot_bounds_world_m'])
  initial['minimum_joint_margin_deg']=min(pair[s]['minimum_joint_margin_deg'] for s in ('left','right'))
  initial['grounded_left_foot_penetration_m']=max(0.,-(.02339816093444824+pair['translation_m'][1]))
  initial['human_proxy_floor_policy']='Existing 150 mm standing left-leg allowance; raised right leg and other obstacles retain floor constraints.'
  for side in ('left','right'):
   initial[side+'_q_rad']=pair[side]['q_rad'];initial[side+'_q_deg']=np.rad2deg(pair[side]['q_rad']).tolist();initial[side+'_tcp_world_m']=pair[side]['position_m'];initial[side+'_rotation']=pair[side]['rotation']
  path=dest/'candidate-initial.yaml';path.write_text(yaml.safe_dump(initial));write(dest/'protocol.json',protocol())
  if not(dest/'geometry').exists():(dest/'geometry').symlink_to(out/'proxy-goals/geometry',target_is_directory=True)
  for row in inventory(ROOT):
   if a.all or row['id'] in a.ids:
    old=dest/'endpoints'/f'{row["id"]}.json'
    if a.resume and old.exists():
     d=json.loads(old.read_text())
     from sock_dressing_simulation.common_opening_approach import sha256
     if d['initial_sha256']==sha256(path) and all(d['sides'].get(s,{}).get('accepted') for s in ('left','right')):continue
    jobs.append((row,dest,path,a.seconds,2,1.))
 results=[]
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  fs={pool.submit(endpoint_one,j):j[1].name for j in jobs}
  for f in as_completed(fs):
   result=f.result();result['candidate']=fs[f];results.append(result);print('ENDPOINT SCREEN',result,flush=True);write(out/'endpoint-screen-summary.json',dict(rows=results,all_70_verified=False))
if __name__=='__main__':main()
