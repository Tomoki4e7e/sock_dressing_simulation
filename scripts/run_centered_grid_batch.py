#!/usr/bin/env python3
"""Run independent foot-state grids in processes and assemble their saved inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from search_centered_reaching import write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/centered-reaching/coarse-grid')
    ap.add_argument('--numbers',type=int,nargs='+')
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--theta-step',type=int,default=2)
    ap.add_argument('--roll-step',type=int,default=30)
    ap.add_argument('--depths',type=float,nargs='+',default=[.01,.015,.02])
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    numbers=args.numbers or [r['number'] for r in json.loads((ROOT/'artifacts/bulged-cuff/summary.json').read_text())['poses']]
    def run(number):
        folder=args.output/f'pose-{number:02d}'
        command=[sys.executable,str(ROOT/'scripts/search_centered_reaching.py'),'grid','--output',str(folder),'--numbers',str(number),
            '--theta-step',str(args.theta_step),'--roll-step',str(args.roll_step),'--depths']+list(map(str,args.depths))
        if args.resume:command.append('--resume')
        with (args.output/f'pose-{number:02d}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        record=json.loads((folder/'geometry-summary.json').read_text())
        print('COMPLETED',number,record['counts'],flush=True)
        return record
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records=list(pool.map(run,numbers))
    counts=Counter()
    for record in records:counts.update(record['counts'])
    write(args.output/'geometry-summary.json',dict(counts=dict(counts),records=len(records),
        settings=dict(numbers=numbers,theta_step=args.theta_step,roll_step_deg=args.roll_step,depths_m=args.depths),
        accepted=[ref for record in records for ref in record['accepted']]))


if __name__=='__main__':
    main()
