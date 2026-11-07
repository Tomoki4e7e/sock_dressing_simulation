#!/usr/bin/env python3
"""Finite-difference checks on full native-scene clearance and IK derivatives."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load
from search_compound_reaching import pose_constraint
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'
s=load(39);state=json.loads((ROOT/'artifacts/bulged-cuff/pose-39/state.json').read_text());k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);checker=CompoundScene(k,s,state['triangles'],OUT/'decomposition.json',ROOT);q=k.start.copy();delta=np.array([-.12,.05,.15]);x=np.r_[q,delta];v,J=checker.slacks(q,.03,delta,with_jacobian=True);fd=[]
for i in range(10):
    probe=x.copy();probe[i]+=1e-6;fd.append((checker.slacks(probe[:7],.03,probe[7:])-v)/1e-6)
error=abs(J-np.array(fd).T).max(1);mask=abs(v-.02)>1e-7
p,R=k.pose(q);pose,P=pose_constraint(k,p,R,x,True);fd=[]
for i in range(10):
    probe=x.copy();probe[i]+=1e-6;fd.append((pose_constraint(k,p,R,probe)-pose)/1e-6)
pose_error=float(abs(P-np.array(fd).T).max());distance_error=float(error[mask].max());result=dict(passed=distance_error<.002 and pose_error<1e-5,active_pairs=int(mask.sum()),maximum_clearance_gradient_error=distance_error,maximum_pose_gradient_error=pose_error,finite_difference_step=1e-6,scope='Native scene safe-start q with translated human; active positive and penetrating contacts, explicit translation columns and reflected world coordinates')
(OUT/'gradient-validation.json').write_text(json.dumps(result,indent=2));print(result)
assert result['passed']
