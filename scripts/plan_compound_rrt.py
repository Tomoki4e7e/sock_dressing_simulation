#!/usr/bin/env python3
"""Bidirectional RRT-Connect to the already validated Cartesian approach."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,argparse,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data
from search_translated_reaching import load,margin
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',default='target-013-trial-1.json');ap.add_argument('--iterations',type=int,default=1800);args=ap.parse_args();row=json.loads((OUT/'coupled-approach'/args.source).read_text());trajectory=row['trajectory'];assert trajectory['stage']=='start_route_collision';c,state,_=target_data(row);s=load(row['number']);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);checker=CompoundScene(k,s,state['triangles'],OUT/'decomposition.json',ROOT);gap=row['gap_m'];delta=np.array(row['translation_m']);start=k.start;goal=np.array(trajectory['approach_q'][0]);lower=k.lower+np.deg2rad(5);upper=k.upper-np.deg2rad(5);rng=np.random.default_rng(20261106);cache={};checks=0
    def valid(q):
        nonlocal checks
        key=tuple(np.round(q,8))
        if key not in cache:checks+=1;cache[key]=bool(np.all(q>=lower-1e-8) and np.all(q<=upper+1e-8) and checker.is_clear(q,gap,delta))
        return cache[key]
    def edge(a,b,step=np.deg2rad(2)):
        n=max(1,int(np.ceil(np.max(abs(b-a))/step)))
        return all(valid(a+(b-a)*t) for t in np.linspace(0,1,n+1))
    assert valid(start) and valid(goal)
    trees=[([start.copy()],[-1]),([goal.copy()],[-1])];began=time.monotonic();found=None
    def extend(tree,target):
        nodes,parents=tree;nearest=int(np.argmin(np.linalg.norm(np.asarray(nodes)-target,axis=1)));q=nodes[nearest];distance=np.linalg.norm(target-q);v=target if distance<=.35 else q+(target-q)*(.35/distance)
        if not edge(q,v):return None,False
        nodes.append(v);parents.append(nearest);return len(nodes)-1,distance<=.35
    def chain(tree,index):
        nodes,parents=tree;path=[]
        while index>=0:path.append(nodes[index]);index=parents[index]
        return path[::-1]
    for i in range(args.iterations):
        active=i%2;other=1-active;mode=rng.random()
        if mode<.15:sample=trees[other][0][0]
        elif mode<.65:
            t=rng.random();sample=np.clip(start*(1-t)+goal*t+rng.normal(0,.65,7),lower,upper)
        else:sample=rng.uniform(lower,upper)
        index,_=extend(trees[active],sample)
        if index is not None:
            target=trees[active][0][index]
            for _ in range(60):
                second,reached=extend(trees[other],target)
                if second is None:break
                if reached:
                    paths=[None,None];paths[active]=chain(trees[active],index);paths[other]=chain(trees[other],second);found=paths[0]+paths[1][-2::-1];break
        if found is not None:break
        if i%100==0:print('rrt',i,'nodes',[len(t[0]) for t in trees],'checks',checks,'seconds',round(time.monotonic()-began,1),flush=True)
    result=dict(source=row['source'],number=row['number'],coupled_source=args.source,translation_m=delta.tolist(),gap_m=gap,iterations=i+1,collision_evaluations=checks,elapsed_s=time.monotonic()-began,found=found is not None,floor_constraint_included=True,executed=False)
    if found is not None:
        for _ in range(80):
            if len(found)<3:break
            a,b=sorted(rng.choice(len(found),2,replace=False))
            if b>a+1 and edge(found[a],found[b]):found=found[:a+1]+found[b:]
        dense=[]
        for a,b in zip(found[:-1],found[1:]):
            n=max(1,int(np.ceil(np.max(abs(b-a))/np.deg2rad(.5))));dense.extend(a+(b-a)*t for t in np.linspace(0,1,n+1)[:-1])
        dense.append(found[-1]);reports=[checker.slacks(q,gap,delta,details=True) for q in dense];result.update(start_route_q=[q.tolist() for q in dense],approach_q=trajectory['approach_q'],minimum_joint_margin_deg=min(margin(k,q) for q in dense),minimum_slack_m=min(r['minimum_slack_m'] for r in reports),dense_recheck_passed=all(r['passed'] for r in reports),start_route_samples=len(dense),continuous_collision_certified=False)
    destination=OUT/'rrt';destination.mkdir(exist_ok=True);(destination/args.source).write_text(json.dumps(result,indent=2));print('found',result['found'],'rechecked',result.get('dense_recheck_passed'),'seconds',round(time.monotonic()-began,1),flush=True)
if __name__=='__main__':main()
