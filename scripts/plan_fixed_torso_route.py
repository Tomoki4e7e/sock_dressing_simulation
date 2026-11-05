#!/usr/bin/env python3
"""Trace a changing-orientation retreat, then plan from a separated arm start."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.optimize import least_squares,minimize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_fixed_torso_starts import context
from search_centered_reaching import state_for,write


def changing_orientation_retreat(k,checker,qgoal,gap,delta,p,R,preferred,direction,max_distance=.35,extra_clearance=.0001,body_clearance=None):
    lower=k.lower+np.deg2rad(5);upper=k.upper-np.deg2rad(5);rows=[np.array(qgoal)];checks=[];failure=None
    clearance=np.full(len(checker.pairs)+len(checker.floor_minima),extra_clearance)
    if body_clearance is not None:
        for i,(aa,bb,mm) in enumerate(checker.pairs):
            if not (aa.startswith('right_gripper/') or bb.startswith('right_gripper/')):continue
            other=bb if aa.startswith('right_gripper/') else aa
            if other in checker.groups and not other.startswith('right_gripper/') and other not in {'right_arm/link_5','right_arm/link_6','right_arm/link_7'}:clearance[i]=body_clearance
    for distance in np.arange(.005,max_distance+.0001,.005):
        target=p+distance*direction;prev=rows[-1];cache={}
        def equality(q):return k.fast_pose(q)[0]-target
        def eqjac(q):return k.fast_pose(q)[2]
        def constraints(q):
            if 'q' not in cache or not np.array_equal(cache['q'],q):
                g,J=checker.slacks(q,gap,delta,with_jacobian=True);cache.update(q=q.copy(),g=g,J=J[:,:7])
            return cache['g'],cache['J']
        fit=least_squares(equality,prev,jac=eqjac,bounds=(lower,upper),max_nfev=80,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        seed=fit.x
        opt=minimize(lambda q:np.sum((q-prev)**2)+.0001*np.sum((q-preferred)**2),seed,
            jac=lambda q:2*(q-prev)+.0002*(q-preferred),method='SLSQP',bounds=list(zip(lower,upper)),
            constraints=[dict(type='eq',fun=equality,jac=eqjac),dict(type='ineq',fun=lambda q:constraints(q)[0]-clearance,jac=lambda q:constraints(q)[1])],options=dict(maxiter=100,ftol=1e-12))
        q=opt.x;error=float(np.linalg.norm(equality(q)));check=checker.slacks(q,gap,delta,details=True)
        if error>1e-4 or not check['passed']:
            failure=dict(stage='retreat_position_or_collision',distance_m=float(distance),position_error_m=error,check=check,solver_message=str(opt.message));break
        # Joint edges are checked, not just the Cartesian target points.
        steps=max(2,int(np.ceil(np.max(abs(q-prev))/np.deg2rad(.25))))
        segment=np.linspace(prev,q,steps+1)
        if not all(checker.is_clear(v,gap,delta) for v in segment):failure=dict(stage='retreat_edge_collision',distance_m=float(distance));break
        rows.append(q);checks.append(dict(distance_m=float(distance),position_error_m=error))
        if distance>=.15-1e-9:
            clearance=checker.gripper_clearances(q,gap,delta)
            if all(min(clearance[s]['skin_m'],clearance[s]['cloth_m'])>=.02 and clearance[s]['robot_body_m']>=.03 for s in ('left','right')):
                return dict(passed=True,stage='changing_orientation_retreat',raw_q=[v.tolist() for v in rows[::-1]],start_q=q.tolist(),distance_m=float(distance),start_clearances=clearance,position_checks=checks)
    return dict(passed=False,stage='changing_orientation_retreat_failed',samples=len(rows),failure=failure,retreat_q=[v.tolist() for v in rows])


def rrt(k,checker,start,goal,gap,delta,iterations=3000):
    lower=k.lower+np.deg2rad(5);upper=k.upper-np.deg2rad(5);rng=np.random.default_rng(20261111);cache={};calls=0
    def valid(q):
        nonlocal calls
        key=tuple(np.round(q,9))
        if key not in cache:calls+=1;cache[key]=bool(np.all(q>=lower-1e-9) and np.all(q<=upper+1e-9) and checker.is_clear(q,gap,delta))
        return cache[key]
    def edge(a,b,step=np.deg2rad(1)):
        n=max(1,int(np.ceil(np.max(abs(b-a))/step)))
        return all(valid(q) for q in np.linspace(a,b,n+1))
    if not valid(start) or not valid(goal):return dict(passed=False,stage='rrt_invalid_endpoint')
    if edge(start,goal):return dict(passed=True,stage='straight_joint_route',raw_q=np.linspace(start,goal,max(2,int(np.ceil(np.max(abs(start-goal))/np.deg2rad(.25))))+1).tolist(),start_q=start.tolist())
    trees=[([start.copy()],[-1]),([goal.copy()],[-1])];began=time.monotonic();found=None
    def chain(t,i):
        nodes,parents=t;path=[]
        while i>=0:path.append(nodes[i]);i=parents[i]
        return path[::-1]
    def extend(tree,target):
        nodes,parents=tree;i=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)));q=nodes[i];dist=np.linalg.norm(target-q);nxt=target if dist<=.22 else q+(target-q)*(.22/dist)
        if not edge(q,nxt):return None,False
        nodes.append(nxt.copy());parents.append(i);return len(nodes)-1,dist<=.22
    for i in range(iterations):
        side=i%2;other=1-side;mode=rng.random()
        if mode<.2:sample=trees[other][0][int(rng.integers(len(trees[other][0])))]
        elif mode<.7:
            t=rng.random();sample=np.clip(start*(1-t)+goal*t+rng.normal(0,.55,7),lower,upper)
        else:sample=rng.uniform(lower,upper)
        j,_=extend(trees[side],sample)
        if j is not None:
            for _ in range(80):
                h,reached=extend(trees[other],trees[side][0][j])
                if h is None:break
                if reached:
                    paths=[None,None];paths[side]=chain(trees[side],j);paths[other]=chain(trees[other],h);found=paths[0]+paths[1][-2::-1];break
        if found is not None:break
        if i%100==0:print('RRT',i,[len(t[0]) for t in trees],calls,round(time.monotonic()-began,1),flush=True)
    if found is None:return dict(passed=False,stage='rrt_exhausted',iterations=iterations,checks=calls)
    for _ in range(80):
        if len(found)<3:break
        a,b=sorted(rng.choice(len(found),2,replace=False))
        if b>a+1 and edge(found[a],found[b]):found=found[:a+1]+found[b:]
    dense=[]
    for a,b in zip(found[:-1],found[1:]):dense.extend(np.linspace(a,b,max(2,int(np.ceil(np.max(abs(b-a))/np.deg2rad(.25))))+1)[:-1])
    dense.append(found[-1]);passed=all(valid(q) for q in dense)
    return dict(passed=passed,stage='rrt_route',raw_q=[q.tolist() for q in dense],start_q=start.tolist(),iterations=i+1,checks=calls,elapsed_s=time.monotonic()-began)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');ap.add_argument('--source',type=Path,default=ROOT/'artifacts/fixed-torso-start-search/continuous-robust/target-0000.json');ap.add_argument('--mode',choices=['retreat','rrt','both'],default='both');ap.add_argument('--iterations',type=int,default=3000);ap.add_argument('--prefix',default='route');a=ap.parse_args();a.output=a.output.resolve()
    a.source=a.source.resolve()
    row=json.loads(a.source.read_text());gr=row['geometry_record'];state=state_for(gr);k,checker,dl,du,contract,scene=context(row['number'],state);attempt=next(v for v in row['attempts'] if v['stage']=='endpoint_passed');q=np.array(attempt['q']);gap=attempt['gap_m'];d=np.array(attempt['translation_m']);checker.park_left(attempt['left_q_rad']);p=np.array(gr['candidate']['grasp_center_unity_world_m'])+d;R=np.array(gr['candidate']['rotation_matrix']);preferred=np.deg2rad([125,-80,60,65,0,-60,55]);routes=[]
    if a.mode in ('retreat','both'):
        directions=[-R[:,0],np.array([0,.4,-1.])/np.linalg.norm([0,.4,-1.]),np.array([-1.,.2,-.3])/np.linalg.norm([-1.,.2,-.3])]
        for i,direction in enumerate(directions):
            route=changing_orientation_retreat(k,checker,q,gap,d,p,R,preferred,direction,extra_clearance=max(.0001,row.get('extra_clearance_m',0)),body_clearance=row.get('body_clearance_m'));routes.append(route);write(a.output/'planning'/f'{a.prefix}-retreat-{i}.json',route);print('RETREAT',i,route['stage'],route.get('distance_m'),flush=True)
            if route['passed']:break
    if not any(v['passed'] for v in routes) and a.mode in ('rrt','both'):
        parks=json.loads((a.output/'left-parks.json').read_text())['candidates']
        for i,park in enumerate(parks):
            start=np.array(park['q_rad'])
            if (start[0]-np.deg2rad(110))*(q[0]-np.deg2rad(110))<0:continue
            if not checker.is_clear(start,gap,d):continue
            clearance=checker.gripper_clearances(start,gap,d)
            if not all(min(clearance[s]['skin_m'],clearance[s]['cloth_m'])>=.02 and clearance[s]['robot_body_m']>=.03 for s in ('left','right')):continue
            route=rrt(k,checker,start,q,gap,d,a.iterations);routes.append(route);write(a.output/'planning'/f'rrt-{i}.json',route)
            if route['passed']:break
    for i,route in enumerate(routes):
        if not route['passed']:continue
        closure=[checker.slacks(q,float(g),d,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,31)];route['virtual_closure_checks']=closure;route['passed']=all(v['passed'] for v in closure)
        new=copy.deepcopy(row);new['attempts']=[dict(attempt,trajectory=route)];new['source_endpoint_record']=str(a.source.relative_to(ROOT));write(a.output/'planned'/f'target-{a.prefix}-{i:04d}.json',new)
        print('PLANNED',route['passed'],len(route['raw_q']),flush=True)
        if route['passed']:break


if __name__=='__main__':main()
