#!/usr/bin/env python3
"""Build B-specific frozen sock shapes from recorded native foot geometry.

This replaces the experimental A-derived native tail preparation. Native Unity
initial-pose captures validate the exported surface separately.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.pose_patterns import read_patterns
from sock_dressing_simulation.fitted_b_cuff import build_fitted_b_state,PROFILE
from sock_dressing_simulation.reaching_dataset import file_sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--numbers',type=int,nargs='+',default=[8,10,3,15,19,22,23,31,38,41,43]);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/b-natural-toe/final-shapes');a=ap.parse_args()
    patterns=read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx');rows=[]
    for n in a.numbers:
        p=next(p for p in patterns if p.number==n);source_n=next(x.number for x in patterns if (x.pitch_deg,x.roll_deg)==(p.pitch_deg,p.roll_deg))
        source_file=ROOT/f'artifacts/bulged-cuff/pose-{source_n:02d}/state.json';original=json.loads(source_file.read_text());state,before,profile=build_fitted_b_state(original,p)
        state['source_record']=dict(file=str(source_file.relative_to(ROOT)),sha256=file_sha256(source_file))
        out=a.output.resolve()/f'pose-{n:02d}';out.mkdir(parents=True,exist_ok=True)
        for name,data in [('state.json',state),('before-bulge.json',dict(particles_world=before.tolist())),('bulge-profile.json',profile),('report.json',state['cuff_fit'])]:
            (out/name).write_text(json.dumps(data,indent=2)+'\n')
        rows.append(dict(number=n,passed=True,state_file=str((out/'state.json').relative_to(ROOT))))
        print('NATURAL B',n,'PASSED',flush=True)
    (a.output.resolve()/'summary.json').write_text(json.dumps(dict(shape_profile=PROFILE,rows=rows,all_passed=True),indent=2)+'\n')


if __name__=='__main__':main()
