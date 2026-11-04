#!/usr/bin/env python3
"""Necessary shoulder-space connectivity condition, independent of wrist routing."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
import numpy as np,fcl
from scipy.ndimage import label
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene
from sock_dressing_simulation.grasp_geometry import HANDEDNESS
out=ROOT/'artifacts/reverse-reaching';s=load(29);k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',s);state=json.loads((ROOT/'artifacts/bulged-cuff/pose-29/state.json').read_text());ck=CollisionScene(k,s,state['triangles']);a=next(r for r in ck.shapes if r['link']=='right_arm/link_2');b=next(r for r in ck.shapes if r['link']=='torso/link_3');T=k.fk(k.start,only=[b['link']])[b['link']];b['obj'].setTransform(fcl.Transform(k.A@T[:3,:3]@HANDEDNESS,k.A@T[:3,3]+k.b));pad=np.deg2rad(5);axis1=np.linspace(k.lower[0]+pad,k.upper[0]-pad,151);axis2=np.linspace(k.lower[1]+pad,k.upper[1]-pad,141);dist=np.zeros((len(axis1),len(axis2)))
for i,x in enumerate(axis1):
 for j,y in enumerate(axis2):
  q=k.start.copy();q[:2]=[x,y];T=k.fk(q,only=[a['link']])[a['link']];a['obj'].setTransform(fcl.Transform(k.A@T[:3,:3]@HANDEDNESS,k.A@T[:3,3]+k.b));dist[i,j]=float(fcl.distance(a['obj'],b['obj'],fcl.DistanceRequest(),fcl.DistanceResult()))
 if i%30==0:print('grid',i,len(axis1),flush=True)
free=dist>=.005;components,n=label(free);row=json.loads((ROOT/'artifacts/compound-reaching/coupled-approach/target-013-trial-1.json').read_text());goal=np.array(row['trajectory']['approach_q'][0]);si=(int(abs(axis1-k.start[0]).argmin()),int(abs(axis2-k.start[1]).argmin()));gi=(int(abs(axis1-goal[0]).argmin()),int(abs(axis2-goal[1]).argmin()));result=dict(grid_shape=list(dist.shape),joint1_degrees=np.rad2deg(axis1).tolist(),joint2_degrees=np.rad2deg(axis2).tolist(),start_component=int(components[si]),goal_component=int(components[gi]),components=int(n),same_component=bool(components[si] and components[si]==components[gi]),blocked_joint2_rows_degrees=np.rad2deg(axis2[np.all(~free,axis=0)]).tolist(),minimum_distance_by_joint2_m=dist.min(0).tolist(),maximum_distance_by_joint2_m=dist.max(0).tolist(),scope='Finite grid necessary-condition test for one collision pair; not a continuous-space impossibility proof')
np.savez_compressed(out/'shoulder-grid.npz',joint1=axis1,joint2=axis2,distance=dist,components=components);(out/'shoulder-connectivity.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ['joint1_degrees','joint2_degrees','minimum_distance_by_joint2_m','maximum_distance_by_joint2_m']})
