#!/usr/bin/env python3
"""Joint right-arm, left-arm and human-translation refinement with fixed torso."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares,minimize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import state_for,write
from search_compound_reaching import pose_constraint
from search_fixed_torso_starts import context,retract_route


def task(ref,out,iterations):
    row=json.loads((ROOT/ref['file']).read_text());a=row['attempts'][ref['trial']];r=row['geometry_record'];state=state_for(r)
    k,checker,dl,du,contract,scene=context(r['number'],state);parks=json.loads((out/'left-parks.json').read_text())['candidates'];left=np.array(parks[ref.get('left_park_index',0)]['q_rad'])
    p=np.array(r['candidate']['grasp_center_unity_world_m']);R=np.array(r['candidate']['rotation_matrix']);gap=ref['gap_m'];pad=np.deg2rad(5)
    lo=np.r_[k.lower+pad,checker.left_lower+pad,dl];hi=np.r_[k.upper-pad,checker.left_upper-pad,du];x0=np.clip(np.r_[a['q'],left,a['translation_m']],lo+1e-10,hi-1e-10)
    cache={};calls=0
    def poses(x,jac=False):
        e,J=pose_constraint(k,p,R,np.r_[x[:7],x[14:]],True);P=np.zeros((6,17));P[:,:7]=J[:,:7];P[:,14:]=J[:,7:]
        return (e,P) if jac else e
    def constraints(x):
        nonlocal calls
        if 'x' not in cache or not np.array_equal(cache['x'],x):
            calls+=1;checker.park_left(x[7:14]);g,J=checker.slacks(x[:7],gap,x[14:],with_jacobian=True);G=np.zeros((len(g),17));G[:,:7]=J[:,:7];G[:,14:]=J[:,7:]
            h=1e-5
            for j in range(7):
                plus=x[7:14].copy();minus=plus.copy();plus[j]+=h;minus[j]-=h
                checker.park_left(plus);gp=checker.slacks(x[:7],gap,x[14:]);checker.park_left(minus);gm=checker.slacks(x[:7],gap,x[14:]);G[:,7+j]=(gp-gm)/(2*h)
            checker.park_left(x[7:14]);cache.update(x=x.copy(),g=g,G=G)
            if calls%25==0:print('REFINE',ref['rank'],'calls',calls,'pose',np.linalg.norm(poses(x)[:3]),'slack',g.min(),flush=True)
        return cache['g'],cache['G']
    weights=np.r_[np.full(3,5.),np.full(3,.75)]
    def restore(x):return np.r_[poses(x)*weights,5*np.minimum(constraints(x)[0],0)]
    def jac(x):
        e,P=poses(x,True);g,G=constraints(x);return np.vstack([P*weights[:,None],5*G*(g<0)[:,None]])
    began=time.monotonic();rest=least_squares(restore,x0,jac=jac,bounds=(lo,hi),max_nfev=iterations,ftol=1e-9,xtol=1e-9,gtol=1e-9)
    preferred=np.r_[np.array(a['q']),left,np.array([(dl[0]+du[0])/2,0.,-.4])]
    def objective(x):return .001*np.sum((x[:14]-preferred[:14])**2)+.02*np.sum((x[14:]-preferred[14:])**2)
    def gradient(x):return np.r_[.002*(x[:14]-preferred[:14]),.04*(x[14:]-preferred[14:])]
    fit=minimize(objective,rest.x,jac=gradient,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=poses,jac=lambda x:poses(x,True)[1]),dict(type='ineq',fun=lambda x:constraints(x)[0],jac=lambda x:constraints(x)[1])],options=dict(maxiter=iterations,ftol=1e-10))
    x=fit.x;e=poses(x);checker.park_left(x[7:14]);check=checker.slacks(x[:7],gap,x[14:],details=True)
    passed=np.linalg.norm(e[:3])<=1e-4 and np.linalg.norm(e[3:])<=np.deg2rad(.5) and check['passed']
    attempt=dict(q=x[:7].tolist(),left_q_rad=x[7:14].tolist(),translation_m=x[14:].tolist(),gap_m=gap,position_error_m=float(np.linalg.norm(e[:3])),orientation_error_deg=float(np.rad2deg(np.linalg.norm(e[3:]))),stage='endpoint_passed' if passed else 'endpoint_failed',endpoint=check,solver_message=str(fit.message),solver_success=bool(fit.success),solver_iterations=fit.nit,restoration_cost=float(rest.cost),constraint_evaluations=calls)
    if passed:attempt['trajectory']=retract_route(k,checker,r['candidate'],x[:7],x[14:],gap)
    result=dict(index=ref['rank'],number=r['number'],geometry_record=r,geometry_reference=row['geometry_reference'],input=ref,central_contract=contract,attempts=[attempt],torso_joints_rad=checker.reference_torso,base_and_head_joints_rad=checker.reference_fixed_joints,joint_limit_margin_deg=5,left_gap_m=.04,all_joint1_branches_searched=True,elapsed_s=time.monotonic()-began,executed=False)
    path=out/'refined'/f"target-{ref['rank']:04d}.json";write(path,result);print('REFINED',ref['rank'],passed,attempt.get('trajectory',{}).get('stage'),attempt['position_error_m'],attempt['orientation_error_deg'],check['minimum_slack_m'],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');ap.add_argument('--inputs',type=Path);ap.add_argument('--rank',type=int);ap.add_argument('--count',type=int,default=12);ap.add_argument('--iterations',type=int,default=80);a=ap.parse_args();a.output=a.output.resolve();inputs=a.inputs or a.output/'refinement-inputs.json'
    if not a.inputs:
        ranked=[]
        for path in (a.output/'search').glob('target-*.json'):
            row=json.loads(path.read_text())
            for i,v in enumerate(row['attempts']):
                if v['stage']=='ik_failed':continue
                for check in v.get('park_checks',[])[:1]:ranked.append((max(0,-check['check']['minimum_slack_m']),path,i,check))
        refs=[];seen=set()
        for score,path,i,c in sorted(ranked,key=lambda v:v[0]):
            row=json.loads(path.read_text());candidate=row['geometry_record']['candidate'];key=(row['number'],candidate['sample_index'],tuple(np.round(np.array(candidate['rotation_matrix']).ravel(),7)))
            if key in seen:continue
            seen.add(key);refs.append(dict(rank=len(refs),file=str(path.relative_to(ROOT)),trial=i,gap_m=c['gap_m'],left_park_index=c['park_index'],initial_maximum_violation_m=score))
            if len(refs)>=a.count:break
        write(inputs,dict(inputs=refs))
    refs=json.loads(inputs.read_text())['inputs']
    for ref in refs:
        if a.rank is None or a.rank==ref['rank']:
            if not (a.output/'refined'/f"target-{ref['rank']:04d}.json").exists():task(ref,a.output,a.iterations)


if __name__=='__main__':main()
