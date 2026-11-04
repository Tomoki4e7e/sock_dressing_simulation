#!/usr/bin/env python3
"""Reverse escape, IK goal families and waypoint-seeded bidirectional planning."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,time,copy,hashlib
from pathlib import Path
import numpy as np,fcl
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data,pose_constraint
from search_translated_reaching import load,good,margin
from sock_dressing_simulation.arm_reaching import Kinematics,triangle_object
from sock_dressing_simulation.grasp_geometry import HANDEDNESS
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/reverse-reaching'

def edge_samples(a,b,step):
 n=max(1,int(np.ceil(np.max(abs(b-a))/step)))
 return np.linspace(a,b,n+1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='target-013-trial-1.json');ap.add_argument('--iterations',type=int,default=4500);ap.add_argument('--escape-steps',type=int,default=35);ap.add_argument('--seed',type=int,default=20261107);args=ap.parse_args();out=OUT/Path(args.source).stem;out.mkdir(parents=True,exist_ok=True)
 row=json.loads((ROOT/'artifacts/compound-reaching/coupled-approach'/args.source).read_text());c,state,_=target_data(row);s=load(row['number']);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);manifest=ROOT/'artifacts/compound-reaching/decomposition-original-parts.json';checker=CompoundScene(k,s,state['triangles'],manifest,ROOT);delta=np.array(row['translation_m']);gap=row['gap_m'];pre=np.array(row['trajectory']['approach_q'][0]);lower=k.lower+np.deg2rad(5);upper=k.upper-np.deg2rad(5);rng=np.random.default_rng(args.seed);start=time.monotonic();cache={};checks=0
 # Cross-check the shoulder pair using triangle surfaces: the convex GJK
 # distance can overestimate clearance near the narrow shoulder passage.
 shoulder=next(x for x in checker.groups['right_arm/link_2']);torso=next(x for x in checker.groups['torso/link_3']);sa=triangle_object(shoulder['vertices'],ConvexHull(shoulder['vertices']).simplices);sb=triangle_object(torso['vertices'],ConvexHull(torso['vertices']).simplices)
 Tf=k.fk(k.start,only=['torso/link_3'])['torso/link_3'];sb.setTransform(fcl.Transform(k.A@Tf[:3,:3]@HANDEDNESS,k.A@Tf[:3,3]+k.b))
 def shoulder_clear(q):
  T=k.fk(q,only=['right_arm/link_2'])['right_arm/link_2'];sa.setTransform(fcl.Transform(k.A@T[:3,:3]@HANDEDNESS,k.A@T[:3,3]+k.b))
  return float(fcl.distance(sa,sb,fcl.DistanceRequest(),fcl.DistanceResult()))>=.005-1e-9
 def valid(q):
  nonlocal checks
  key=tuple(np.round(q,10))
  if key not in cache:checks+=1;cache[key]=bool(np.all(q>=lower-1e-9) and np.all(q<=upper+1e-9) and shoulder_clear(q) and checker.is_clear(q,gap,delta))
  return cache[key]
 def edge(a,b,step=np.deg2rad(1)):
  return all(valid(q) for q in edge_samples(a,b,step))
 assert valid(k.start) and valid(pre)
 trees=[([k.start.copy()],[-1]),([pre.copy()],[-1])];records=[]
 def add_chain(side,chain):
  parent=0
  for q in chain[1:]:trees[side][0].append(q.copy());trees[side][1].append(parent);parent=len(trees[side][0])-1
  return parent
 # Move the wrist while it is still far below the foot.
 for fraction in [.25,.5,1.]:
  q=k.start.copy();q[4:]+=fraction*(pre[4:]-k.start[4:]);passed=edge(k.start,q)
  records.append(dict(kind='wrist_preparation',fraction=fraction,passed=passed,q=q.tolist()))
  if passed:add_chain(0,[k.start,q])
 # Follow the redundant IK manifold, retaining checked links to the original approach.
 comfortable=copy.copy(k);comfortable.lower=lower;comfortable.upper=upper;p,R=k.pose(pre)
 for sign in [-1,1]:
  chain=[pre];direction=None
  for i in range(16):
   q=chain[-1];_,_,jp,ja=k.fast_pose(q);_,_,V=np.linalg.svd(np.vstack([jp,ja]));null=V[-1]
   if direction is None:direction=null*sign
   elif null@direction<0:direction=-null
   else:direction=null
   seed=np.clip(q+.035*direction,lower+1e-8,upper-1e-8);sol=comfortable.solve(p,R,seed);nextq=np.array(sol['q'])
   if not good(sol) or np.linalg.norm(nextq-q)<.003 or not edge(q,nextq,np.deg2rad(.5)):break
   chain.append(nextq)
  add_chain(1,chain);records.append(dict(kind='alternative_preapproach_ik',sign=sign,states=len(chain),q=[q.tolist() for q in chain]));print('IK family',sign,len(chain),flush=True)
 # Escape from the constrained goal region with free tool orientation.
 hand_ids=np.array([i for i,(a,b,_) in enumerate(checker.pairs) if a.startswith('right_gripper/') and b in ['human_full_skin','right_leg_actual_skin_rigid','frozen_cloth']])
 for mode,bias in [('outside',np.array([-.12,.03,0.])),('above',np.array([0.,.12,0.])),('toe_side',np.array([0.,.03,.12]))]:
  chain=[pre];last_cache={};mode_start=time.monotonic()
  for i in range(args.escape_steps):
   q0=chain[-1];tcp,_,_,_=k.fast_pose(q0);target=tcp+bias;preferred=k.start.copy();preferred[4:]=.5*(pre[4:]+k.start[4:]);box=.09;lo=np.maximum(lower,q0-box);hi=np.minimum(upper,q0+box)
   def constraints(q):
    if 'q' not in last_cache or not np.array_equal(q,last_cache['q']):
     g,J=checker.slacks(q,gap,delta,with_jacobian=True,cap=.06);last_cache.update(q=q.copy(),g=g,J=J[:,:7])
    return last_cache['g'],last_cache['J']
   def cost(q):
    g,J=constraints(q);pos,_,jp,_=k.fast_pose(q);deficit=np.minimum(g[hand_ids]-.025,0)
    value=.04*np.sum((q-preferred)**2)+3*np.sum((pos-target)**2)+30*np.sum(deficit**2)
    grad=.08*(q-preferred)+6*jp.T@(pos-target)+60*J[hand_ids].T@deficit
    return value,grad
   fit=minimize(cost,np.clip(q0,lo,hi),jac=True,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=lambda q:constraints(q)[0],jac=lambda q:constraints(q)[1])],options=dict(maxiter=35,ftol=1e-10))
   q1=fit.x;accepted=False
   for scale in [1.,.5,.25,.125]:
    q=q0+scale*(q1-q0)
    if np.linalg.norm(q-q0)>.002 and edge(q0,q,np.deg2rad(.5)):chain.append(q);accepted=True;break
   if not accepted:break
   if edge(k.start,chain[-1]):break
  add_chain(1,chain);records.append(dict(kind='reverse_escape',mode=mode,states=len(chain),q=[q.tolist() for q in chain],elapsed_s=time.monotonic()-mode_start));print('escape',mode,len(chain),'seconds',round(time.monotonic()-mode_start,1),flush=True)
 (out/'waypoints.json').write_text(json.dumps(records,indent=2))
 def ancestry(tree,index):
  nodes,parents=tree;chain=[]
  while index>=0:chain.append(nodes[index]);index=parents[index]
  return chain[::-1]
 found=None
 # Try deterministic connections between prepared and escaped configurations.
 for a in range(len(trees[0][0])):
  for b in reversed(range(len(trees[1][0]))):
   if edge(trees[0][0][a],trees[1][0][b]):found=ancestry(trees[0],a)+ancestry(trees[1],b)[::-1];break
  if found is not None:break
 def extend(tree,target):
  nodes,parents=tree;nearest=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)));q=nodes[nearest];dist=np.linalg.norm(target-q);end=target if dist<=.22 else q+(target-q)*.22/dist
  if not edge(q,end):return None,False
  nodes.append(end.copy());parents.append(nearest);return len(nodes)-1,dist<=.22
 iterations=0
 for i in range(args.iterations):
  if found is not None:break
  iterations=i+1;side=i%2;other=1-side;mode=rng.random()
  if mode<.2:sample=trees[other][0][int(rng.integers(len(trees[other][0])))]
  elif mode<.75:
   anchor=trees[side][0][int(rng.integers(len(trees[side][0])))];sample=np.clip(anchor+rng.normal(0,.35,7),lower,upper)
  else:sample=rng.uniform(lower,upper)
  a,_=extend(trees[side],sample)
  if a is not None:
   for _ in range(80):
    b,reached=extend(trees[other],trees[side][0][a])
    if b is None:break
    if reached:
     paths=[None,None];paths[side]=ancestry(trees[side],a);paths[other]=ancestry(trees[other],b);found=paths[0]+paths[1][-2::-1];break
  if i%200==0:print('rrt',i,'nodes',[len(t[0]) for t in trees],'checks',checks,'seconds',round(time.monotonic()-start,1),flush=True)
 result=dict(source=row['source'],number=row['number'],coupled_source=args.source,translation_m=delta.tolist(),gap_m=gap,found=found is not None,iterations=iterations,checks=checks,elapsed_s=time.monotonic()-start,floor_constraint_included=True,orientation_free_before_last_50mm=True,planning_geometry=str(manifest.relative_to(ROOT)),planning_geometry_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),executed=False,continuous_collision_certified=False,waypoint_records=records)
 if found is not None:
  for _ in range(60):
   a,b=sorted(rng.choice(len(found),2,replace=False))
   if b>a+1 and edge(found[a],found[b],np.deg2rad(.5)):found=found[:a+1]+found[b:]
  dense=[]
  for a,b in zip(found[:-1],found[1:]):dense.extend(edge_samples(a,b,np.deg2rad(.25))[:-1])
  dense.append(found[-1]);passed=all(valid(q) for q in dense);result.update(start_route_q=[q.tolist() for q in dense],approach_q=row['trajectory']['approach_q'],dense_recheck_passed=passed,minimum_joint_margin_deg=min(margin(k,q) for q in dense))
 (out/'result.json').write_text(json.dumps(result,indent=2));print('DONE',result['found'],result.get('dense_recheck_passed'),round(result['elapsed_s'],1),flush=True)
if __name__=='__main__':main()
