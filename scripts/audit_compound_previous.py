#!/usr/bin/env python3
"""Compare the previous successful IK trials under the compound collision model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data
from search_translated_reaching import load,good
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def task(arg):
    i,row=arg;old=json.loads((ROOT/f'artifacts/translated-reaching/target-{i:03d}.json').read_text());valid=[(n,a) for n,a in enumerate(old['attempts']) if good(a)]
    if not valid:return dict(index=i,source=row['source'],checks=[])
    c,state,gaps=target_data(row);s=load(row['number']);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);scene=CompoundScene(k,s,state['triangles'],OUT/'decomposition.json',ROOT);checks=[]
    for n,a in valid:
        for gap in gaps:
            report=scene.slacks(a['q'],gap,a['translation_m'],details=True);checks.append(dict(attempt=n,gap_m=gap,**report))
    print('audit',i,len(checks),flush=True)
    return dict(index=i,source=row['source'],checks=checks)

def main():
    rows=json.loads((ROOT/'artifacts/translated-reaching/summary.json').read_text())['results']
    with ProcessPoolExecutor(3) as pool:results=list(pool.map(task,enumerate(rows)))
    pairs=Counter();total=0;passed=0
    for row in results:
        for c in row['checks']:
            total+=1;passed+=c['passed']
            for issue in c['issues']:pairs[tuple(issue['pair'])]+=1
    report=dict(checks=total,endpoint_passes=passed,pairs=[dict(pair=list(p),count=n) for p,n in pairs.most_common()],results=results,scope='Old successful IK trials rechecked with compound geometry; includes fixed robot versus translated human and support')
    (OUT/'previous-ik-compound-audit.json').write_text(json.dumps(report,indent=2));print('total',total,'passed',passed)
if __name__=='__main__':main()
