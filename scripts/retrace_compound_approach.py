#!/usr/bin/env python3
"""Trace each coupled survivor using collision-constrained IK at every waypoint."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data,trace
from search_translated_reaching import load
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def task(path):
 r=json.loads(path.read_text())
 if not r['endpoint_and_preapproach_passed']:return
 c,state,_=target_data(r);s=load(r['number']);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);checker=CompoundScene(k,s,state['triangles'],OUT/'decomposition.json',ROOT)
 result=trace(k,checker,c,r['gap_m'],np.r_[r['q'],r['translation_m']],pre_seed=r['preapproach_q']);r['trajectory_before_constrained_ik']=r.get('trajectory');r['trajectory']=result;path.write_text(json.dumps(r,indent=2));print(path.name,result['stage'],flush=True)

def main():
 with ProcessPoolExecutor(3) as pool:list(pool.map(task,sorted((OUT/'coupled-approach').glob('*.json'))))
if __name__=='__main__':main()
