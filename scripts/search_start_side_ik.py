#!/usr/bin/env python3
"""Search alternative endpoint IK on the start side of the shoulder barrier."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,copy
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load
from search_compound_reaching import target_data
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',default='target-013-trial-1.json');ap.add_argument('--seeds',type=int,default=48);args=ap.parse_args();r=json.loads((ROOT/'artifacts/compound-reaching/coupled-approach'/args.source).read_text());c,state,_=target_data(r);s=load(r['number']);s['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);bounded=copy.copy(k);bounded.lower=k.lower+np.deg2rad(5);bounded.upper=k.upper-np.deg2rad(5);bounded.upper[0]=np.deg2rad(105);bounded.start=np.clip(k.start,bounded.lower,bounded.upper);checker=CompoundScene(k,s,state['triangles'],ROOT/'artifacts/compound-reaching/decomposition-original-parts.json',ROOT);p=np.array(c['grasp_center_unity_world_m'])+r['translation_m'];R=np.array(c['rotation_matrix']);out={}
 for name,target in [('grasp',p),('preapproach',p-R[:,0]*.05)]:
  solutions,attempts=bounded.multistart(target,R,count=args.seeds);checks=[dict(ik=sol,collision=checker.slacks(sol['q'],r['gap_m'],r['translation_m'],details=True)) for sol in solutions];out[name]=dict(solutions=solutions,attempts=attempts,checks=checks);print(r['number'],name,'IK',len(solutions),'collision free',sum(c['collision']['passed'] for c in checks),flush=True)
 destination=ROOT/'artifacts/reverse-reaching';(destination/(Path(args.source).stem+'-start-side-ik.json')).write_text(json.dumps(dict(source=r['source'],number=r['number'],joint1_upper_deg=105,joint_limit_margin_deg=5,seeds=args.seeds,results=out),indent=2))
if __name__=='__main__':main()
