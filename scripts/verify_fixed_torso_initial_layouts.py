#!/usr/bin/env python3
"""Solve grasp IK at explicitly collision-free initial human/arm layouts."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_fixed_torso_starts import context,retract_route
from search_centered_reaching import state_for,write
from search_translated_reaching import good
from sock_dressing_simulation.grasp_geometry import HANDEDNESS


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');ap.add_argument('--seeds',type=int,default=8);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--shard-index',type=int,default=0);a=ap.parse_args();a.output=a.output.resolve()
    layouts=json.loads((a.output/'initial-configurations.json').read_text())['candidates'];refs=json.loads((ROOT/'artifacts/centered-reaching/full-grid/geometry-summary.json').read_text())['accepted'];contexts={};total=Counter()
    for index,ref in enumerate(refs):
        if index%a.shards!=a.shard_index:continue
        path=a.output/'placed-ik'/f'target-{index:04d}.json'
        if path.exists():continue
        r=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']];state=state_for(r);n=r['number']
        if n not in contexts:contexts[n]=context(n,state)
        k,checker,dl,du,contract,scene=contexts[n];comfortable=copy.copy(k);comfortable.lower=k.lower+np.deg2rad(5);comfortable.upper=k.upper-np.deg2rad(5)
        shoulder=k.A@(k.arm_parent@k.arm_origins[0])[:3,3]+k.b;radius=sum(np.linalg.norm(v[:3,3]) for v in k.arm_origins[1:])+np.linalg.norm(k.arm_tail[:3,3]);c=r['candidate'];R=np.array(c['rotation_matrix']);p=np.array(c['grasp_center_unity_world_m']);rows=[];counts=Counter();found=False
        for li,layout in enumerate(layouts):
            d=np.clip(np.array(layout['translation_m']),dl,du);checker.park_left(layout['left_q_rad']);start=np.array(layout['right_q_rad']);gap=max(v['gap_m'] for v in r['valid_openings'])
            initial=checker.slacks(start,.04,d,details=True);entry=dict(layout_index=li,translation_m=d.tolist(),initial=initial,attempts=[]);rows.append(entry);counts['layouts']+=1
            if not initial['passed']:entry['stage']='initial_collision';counts['initial_collision']+=1;continue
            base=p+d-R@HANDEDNESS@k.hand.grasp_center_urdf;distance=float(np.linalg.norm(base-shoulder));entry.update(required_gripper_base_distance_m=distance,arm_length_upper_bound_m=float(radius))
            if distance>radius+1e-9:entry['stage']='arm_length_bound';counts['arm_length_bound']+=1;continue
            rng=np.random.default_rng(20261110+index*100+li);entry['stage']='ik_failed'
            seeds=[start,np.deg2rad([160,-60,0,90,0,0,45]),np.deg2rad([200,-50,60,70,0,-30,45]),k.start]
            for si in range(a.seeds):
                seed=seeds[si] if si<len(seeds) else rng.uniform(comfortable.lower,comfortable.upper);sol=comfortable.solve(p+d,R,seed);counts['ik_trials']+=1;entry['attempts'].append(sol)
                if not good(sol):continue
                counts['ik_passes']+=1;entry['stage']='endpoint_collision';checks=[];sol['checks']=checks
                for opening in r['valid_openings']:
                    gap=opening['gap_m'];check=checker.slacks(np.array(sol['q']),gap,d,details=True);checks.append(dict(gap_m=gap,check=check))
                    if not check['passed']:continue
                    counts['endpoint_passes']+=1;entry['stage']='endpoint_passed';route=retract_route(k,checker,c,np.array(sol['q']),d,gap)
                    record=dict(index=index,number=n,geometry_reference=ref,geometry_record=r,central_contract=contract,
                        torso_joints_rad=checker.reference_torso,base_and_head_joints_rad=checker.reference_fixed_joints,left_gap_m=.04,
                        attempts=[dict(sol,left_q_rad=layout['left_q_rad'],translation_m=d.tolist(),gap_m=gap,stage='endpoint_passed',endpoint=check,trajectory=route)],
                        initial_layout_index=li,executed=False)
                    write(a.output/'planned'/f'target-{index:04d}.json',record)
                    print('PLACED ENDPOINT',index,n,li,route['stage'],flush=True)
                    if route['passed']:counts['routes']+=1;found=True;break
                if found:break
            if found:break
        total.update(counts);write(path,dict(index=index,number=n,geometry_reference=ref,counts=dict(counts),layouts=rows,torso_fixed=True));print('PLACED',index,len(refs),n,dict(counts),flush=True)
        if found:break
    write(a.output/f'placed-summary-shard-{a.shard_index}.json',dict(counts=dict(total),executed=False))


if __name__=='__main__':main()
