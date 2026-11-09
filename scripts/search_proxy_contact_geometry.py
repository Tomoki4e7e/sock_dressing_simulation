#!/usr/bin/env python3
"""Search actual-skin/cloth contact while avoiding active human proxies."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcl,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import search_contact_pregrasp as search
from sock_dressing_simulation.contact_pregrasp import ContactGoal
from sock_dressing_simulation.planning_human_colliders import augmented_scene
from sock_dressing_simulation.arm_reaching import shape_vertices,triangle_object
from sock_dressing_simulation.grasp_geometry import convex_geometry
from sock_dressing_simulation.common_opening_approach import inventory,sha256
from search_centered_reaching import write
BASE_SEED_POSES=search.seed_poses
class ProxyGoal(ContactGoal):
 def __init__(self,state,hand,obstacles):
  super().__init__(state,hand);self.obstacles=[]
  for row in obstacles:
   points=shape_vertices(row)
   obj=triangle_object(points,row['triangles']) if row['kind']=='MeshCollider' and not row['convex'] else fcl.CollisionObject(convex_geometry(points)[0])
   self.obstacles.append((row['name'],obj,points.min(0),points.max(0)))
 def proxy_slacks(self,p,R,gap):
  self.place(p,R,gap);values=[]
  for part in self.parts:
   points=(self.vertices[part.label]+[0,self.coefficient(part.label)*(gap-self.reference_gap),0])@R.T+p;lo,hi=points.min(0),points.max(0)
   nearest=float('inf')
   for name,obj,blo,bhi in self.obstacles:
    bound=np.linalg.norm(np.maximum(np.maximum(lo-bhi,blo-hi),0))
    d=bound if bound>.0011 else self.distance(part,obj)
    nearest=min(nearest,d-.001)
   values.append(nearest)
  return np.array(values)
 def residual(self,p,R,gap,side,finger):
  return np.r_[super().residual(p,R,gap,side,finger),4000*np.minimum(self.proxy_slacks(p,R,gap),0)]
 def inspect(self,p,R,gap,side,finger,check_rim=True):
  result=super().inspect(p,R,gap,side,finger,check_rim);slacks=self.proxy_slacks(p,R,gap);result['human_proxy_minimum_slack_m']=float(slacks.min());result['human_proxies_passed']=bool(slacks.min()>=-1e-9);result['passed']=result['passed'] and result['human_proxies_passed'];return result

def job(payload):
 row,out,seconds,maximum,manifest=payload;obstacles=augmented_scene(dict(obstacles=[]),ROOT,manifest,[row['scene_number']])['obstacles'];search.ContactGoal=lambda state,hand:ProxyGoal(state,hand,obstacles)
 original=BASE_SEED_POSES
 def seeds(goal,row,side,rng):
  old=json.loads((ROOT/'artifacts/contact-pregrasp-start-search/geometry'/f'{row["id"]}.json').read_text());prior=old['sides'][side]['accepted'];return [(np.array(c['position_m']),np.array(c['rotation']),c['opening_gap_m'],c['cloth_finger']) for c in prior]+original(goal,row,side,rng)
 search.seed_poses=seeds
 result=search.geometry_one((row,out,seconds,maximum));path=out/'geometry'/f'{row["id"]}.json';d=json.loads(path.read_text());d['human_collider_manifest_sha256']=sha256(manifest);write(path,d);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-placement-search/proxy-goals');p.add_argument('--manifest',type=Path,default=ROOT/'artifacts/contact-placement-search/human-colliders/manifest.json');p.add_argument('--ids',nargs='+',default=['D-01','E-14','E-50']);p.add_argument('--all',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--seconds',type=float,default=60);p.add_argument('--workers',type=int,default=3);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);rows=[r for r in inventory(ROOT) if a.all or r['id'] in a.ids];results=[]
 if a.resume:
  pending=[]
  for r in rows:
   path=a.output/'geometry'/f'{r["id"]}.json'
   old=json.loads(path.read_text()) if path.exists() else {}
   if old.get('human_collider_manifest_sha256')==sha256(a.manifest) and all(old.get('sides',{}).get(s,{}).get('accepted') for s in ('left','right')):
    results.append(dict(id=r['id'],left=len(old['sides']['left']['accepted']),right=len(old['sides']['right']['accepted'])))
   else:pending.append(r)
  rows=pending
 with ProcessPoolExecutor(max_workers=a.workers) as pool:
  for f in as_completed([pool.submit(job,(r,a.output.resolve(),a.seconds,2,a.manifest.resolve())) for r in rows]):results.append(f.result());write(a.output/'summary.json',dict(rows=results,all_70_verified=False))
if __name__=='__main__':main()
