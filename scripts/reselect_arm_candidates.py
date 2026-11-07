#!/usr/bin/env python3
"""Add bounded right-arm IK to every one of the 60 geometric configurations."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics
K=None

def initialize():
    global K
    m=json.loads((ROOT/'artifacts/arm-reaching/pose-03/manifest.json').read_text())
    K=Kinematics(ROOT/m['urdf'],json.loads((ROOT/'artifacts/arm-reaching/pose-03/initial-scene.json').read_text()))
    K.preferred_seed=np.deg2rad([0,-60,0,90,0,-60,55])

def task(group):
    key,rows=group;out=ROOT/'artifacts/arm-reaching-reselection/existing';out.mkdir(parents=True,exist_ok=True);path=out/(key+'.json')
    if path.exists():
        cached=json.loads(path.read_text())
        if 'attempts' in cached:return cached
    c=rows[0]['candidate'];p=np.asarray(c['grasp_center_unity_world_m']);R=np.asarray(c['rotation_matrix'])
    solutions,attempts=K.multistart(p,R,count=48);pairs=[]
    for solution in solutions:
        pre=K.solve(p-R[:,0]*.05,R,np.asarray(solution['q']))
        if pre['position_error_m']>.0001 or pre['orientation_error_deg']>.5:continue
        margin=float(np.rad2deg(np.minimum(np.asarray(solution['q'])-K.lower,K.upper-np.asarray(solution['q']))).min())
        pairs.append(dict(endpoint=solution,preapproach=pre,endpoint_minimum_limit_margin_deg=margin))
    result=dict(configurations=[dict(number=r['number'],candidate_index=r['candidate_index']) for r in rows],
        ik_seeds=48,ik_solution_count=len(solutions),endpoint_solutions=solutions,preapproach_pairs=pairs,
        comfortable_pairs=[r for r in pairs if r['endpoint_minimum_limit_margin_deg']>=5],
        best_attempt=min(attempts,key=lambda r:r['position_error_m']**2+(.15*np.deg2rad(r['orientation_error_deg']))**2),
        attempts=attempts,preferred_seed=K.preferred_seed.tolist(),collision_checked=False)
    path.write_text(json.dumps(result,indent=2)+'\n');print(result['configurations'],'IK',len(solutions),'pairs',len(pairs),flush=True);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);a=parser.parse_args();groups={}
    for path in (ROOT/'artifacts/geometric-closure').glob('pose-*/report.json'):
        for r in json.loads(path.read_text())['candidates']:
            if not r['accepted']:continue
            c=r['candidate'];key=hashlib.sha256(json.dumps([r['number'],c['grasp_center_unity_world_m'],c['rotation_matrix']]).encode()).hexdigest()[:16]
            groups.setdefault(key,[]).append(r)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize) as pool:results=list(pool.map(task,groups.items()))
    summary=dict(input_configurations=60,unique_target_poses=len(results),
        configurations_with_endpoint_ik=sum(len(r['configurations']) for r in results if r['ik_solution_count']),
        configurations_with_preapproach_ik=sum(len(r['configurations']) for r in results if r['preapproach_pairs']),
        configurations_with_comfortable_ik=sum(len(r['configurations']) for r in results if r['comfortable_pairs']),
        comfortable_joint_limit_margin_deg=5,results=results)
    (ROOT/'artifacts/arm-reaching-reselection/existing-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:v for k,v in summary.items() if k!='results'})
if __name__=='__main__':main()
