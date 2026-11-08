#!/usr/bin/env python3
"""Evaluate every unchanged planned endpoint with the previous A grasp rule."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from common_opening_replay_support import evaluate_grasp,verify_source
from search_centered_reaching import REGISTERED_MANIFEST,write
from search_fixed_torso_starts import context
from sock_dressing_simulation.common_opening_approach import load_state,sha256


def one(args):
    paths,out=args;rows=[]
    first=json.loads(paths[0].read_text());state=load_state(ROOT,first['condition']);k,checker,*_=context(first['condition']['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=.15)
    for path in paths:
        record=verify_source(path,path.parents[2]/'initial.yaml',ROOT/first['condition']['state']);side=record['side'];goal=record['route']['q_rad'][-1];right=goal if side=='right' else record['other_arm_q_rad'];left=goal if side=='left' else record['other_arm_q_rad']
        result=evaluate_grasp(record,state,k,checker,right,left,.001,np.array(record['translation_m']));result.update(source=str(path.relative_to(ROOT)),source_sha256=sha256(path),condition=record['condition'],evaluation='Unchanged planned endpoint with virtual 1 mm jaw closure, previous A geometric rule; separate from measured motion')
        destination=out/'replays'/side/record['condition']['id'];destination.mkdir(parents=True,exist_ok=True);write(destination/'planned-goal-grasp.json',result)
        rows.append(dict(side=side,condition=record['condition'],passed=result['passed'],reason=result['geometric_check_reason'],capture_area_mm2=result['actual_projection']['area_m2']*1e6,source=result['source'],source_sha256=result['source_sha256'],result=str((destination/'planned-goal-grasp.json').relative_to(ROOT))))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/'artifacts/common-opening-approach/negative-wrist-start');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/common-opening-dynamics');ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();source=a.source.resolve();out=a.output.resolve();groups={}
    for p in sorted(source.glob('routes/*/*.json')):groups.setdefault(p.stem,[]).append(p)
    if len(groups)!=70 or any(len(v)!=2 for v in groups.values()):raise ValueError('Expected 70 states with both arm routes')
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for batch in pool.map(one,[(paths,out) for paths in groups.values()]):
            rows.extend(batch);print('GOALS',len(rows),'passed',sum(r['passed'] for r in rows),flush=True)
    write(out/'planned-grasp-summary.json',dict(complete=True,total=len(rows),reason_counts=dict(Counter(r['reason'] for r in rows)),maximum_capture_area_mm2=max(r['capture_area_mm2'] for r in rows),passed=sum(r['passed'] for r in rows),criteria_sha256=sha256(ROOT/'config/geometric_closure.yaml'),rows=rows,scope='Planned endpoint joints, frozen saved cloth, virtual 1 mm jaw gap; previous A geometric capture rule'))

if __name__=='__main__':main()
