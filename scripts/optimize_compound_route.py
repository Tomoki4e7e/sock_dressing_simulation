#!/usr/bin/env python3
"""Optimize a joint-space route with clearance constraints at all intermediate knots."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,sys,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data
from search_translated_reaching import load,margin
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='target-013-trial-1.json');ap.add_argument('--knots',type=int,default=15);args=ap.parse_args();row=json.loads((OUT/'coupled-approach'/args.source).read_text());c,state,_=target_data(row);s=load(row['number']);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);checker=CompoundScene(k,s,state['triangles'],OUT/'decomposition-original-parts.json',ROOT);gap=row['gap_m'];delta=np.array(row['translation_m']);start=k.start;goal=np.array(row['trajectory']['approach_q'][0]);N=args.knots;pad=np.deg2rad(5);lower=k.lower+pad;upper=k.upper-pad;initial=np.linspace(start,goal,N);cache={};calls=0;began=time.monotonic()
 assert checker.is_clear(start,gap,delta) and checker.is_clear(goal,gap,delta)
 keep=np.array([checker.groups[a][0]['moving'] or b in checker.groups and checker.groups[b][0]['moving'] for a,b,_ in checker.pairs]);keep=np.r_[keep,np.zeros(len(checker.floor_minima),dtype=bool)];m=int(keep.sum())
 def route(x):return np.vstack([start,x.reshape(-1,7),goal])
 def evaluate(x):
  nonlocal calls
  if 'x' not in cache or not np.array_equal(cache['x'],x):
   calls+=1;path=route(x);v=[];J=np.zeros((m*(N-2),7*(N-2)))
   for i,q in enumerate(path[1:-1]):
    g,G=checker.slacks(q,gap,delta,with_jacobian=True);v.extend(g[keep]);J[i*m:(i+1)*m,i*7:(i+1)*7]=G[keep,:7]
   cache.update(x=x.copy(),v=np.array(v),J=J)
  return cache['v'],cache['J']
 def energy(x,penalty=0):
  path=route(x);diff=np.diff(path,axis=0);gradient=2*(2*path[1:-1]-path[:-2]-path[2:]);cost=np.sum(diff**2)
  if penalty:
   g,J=evaluate(x);negative=np.minimum(g,0);cost+=penalty*np.sum(negative**2);gradient=gradient.ravel()+2*penalty*J.T@negative
  return float(cost),gradient.ravel()
 bounds=list(zip(np.tile(lower,N-2),np.tile(upper,N-2)));x0=initial[1:-1].ravel();rest=minimize(lambda x:energy(x,20000),x0,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=120,ftol=1e-11,gtol=1e-8));print('restored',float(evaluate(rest.x)[0].min()),'calls',calls,flush=True)
 fit=minimize(lambda x:energy(x)[0],rest.x,jac=lambda x:energy(x)[1],method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=lambda x:evaluate(x)[0],jac=lambda x:evaluate(x)[1])],options=dict(maxiter=160,ftol=1e-9));path=route(fit.x);g=evaluate(fit.x)[0];dense=[]
 for a,b in zip(path[:-1],path[1:]):
  n=max(1,int(np.ceil(np.max(abs(b-a))/np.deg2rad(.5))));dense.extend(a+(b-a)*t for t in np.linspace(0,1,n+1)[:-1])
 dense.append(goal);valid=[checker.is_clear(q,gap,delta) for q in dense];result=dict(source=row['source'],number=row['number'],coupled_source=args.source,translation_m=delta.tolist(),gap_m=gap,floor_constraint_included=True,knots=N,minimum_knot_slack_m=float(g.min()),solver_message=str(fit.message),constraint_evaluations=calls,elapsed_s=time.monotonic()-began,found=bool(g.min()>=-1e-9 and all(valid)),dense_recheck_passed=bool(all(valid)),failed_dense_samples=[i for i,v in enumerate(valid) if not v],start_route_q=[q.tolist() for q in dense],approach_q=row['trajectory']['approach_q'],executed=False,continuous_collision_certified=False)
 destination=OUT/'optimized-route';destination.mkdir(exist_ok=True);(destination/args.source).write_text(json.dumps(result,indent=2));print('route',result['found'],'slack',result['minimum_knot_slack_m'],'failed',len(result['failed_dense_samples']),'seconds',round(result['elapsed_s'],1),flush=True)
if __name__=='__main__':main()
