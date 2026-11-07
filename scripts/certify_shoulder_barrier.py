#!/usr/bin/env python3
"""Conservative necessary-condition barrier test for the fixed planning model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,hashlib,argparse
from pathlib import Path
import numpy as np,fcl
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene,ARM,triangle_object
from sock_dressing_simulation.grasp_geometry import HANDEDNESS
parser=argparse.ArgumentParser();parser.add_argument('--robot-scene');parser.add_argument('--output',default='artifacts/reverse-reaching');args=parser.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True);s=load(29)
if args.robot_scene:
 from apron_reaching_common import overlay_robot
 s=overlay_robot(s,json.loads((ROOT/args.robot_scene).read_text()))
urdf=ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf';k=Kinematics(urdf,s);state=json.loads((ROOT/'artifacts/bulged-cuff/pose-29/state.json').read_text());ck=CollisionScene(k,s,state['triangles']);moving=[r for r in ck.shapes if r['link']=='right_arm/link_2'];fixed=[r for r in ck.shapes if r['link']=='torso/link_3'];assert len(moving)==len(fixed)==1;a=moving[0];b=fixed[0]
a=dict(a,obj=triangle_object(a['vertices'],ConvexHull(a['vertices']).simplices));b=dict(b,obj=triangle_object(b['vertices'],ConvexHull(b['vertices']).simplices))
# Verify that no other moving arm joint can affect either shape.
def dependencies(link):
 result=[]
 while link in k.parents:
  joint=k.parents[link]
  if joint.get('name') in ARM:result.append(joint.get('name'))
  link=joint.find('parent').get('link')
 return result
assert set(dependencies(a['link']))==set(ARM[:2]) and dependencies(b['link'])==[]
q=k.start.copy();q[0]=np.deg2rad(110);q[1]=0;frames=k.fk(q);T=frames[b['link']];b['obj'].setTransform(fcl.Transform(k.A@T[:3,:3]@HANDEDNESS,k.A@T[:3,3]+k.b));joint=k.joints[ARM[1]];parent=joint.find('parent').get('link');Tj=frames[parent]@k.origins[ARM[1]];origin=k.A@Tj[:3,3]+k.b;axis=k.A@Tj[:3,:3]@k.axes[ARM[1]];Ta=frames[a['link']];points=a['vertices']@(k.A@Ta[:3,:3]@HANDEDNESS).T+k.A@Ta[:3,3]+k.b;v=points-origin;radius=float(np.linalg.norm(v-np.outer(v@axis,axis),axis=1).max());lo=k.lower[1]+np.deg2rad(5);hi=k.upper[1]-np.deg2rad(5);values=np.linspace(lo,hi,int(np.ceil((hi-lo)/np.deg2rad(.005)))+1);distances=[]
for angle in values:
 q[1]=angle;T=k.fk(q,only=[a['link']])[a['link']];a['obj'].setTransform(fcl.Transform(k.A@T[:3,:3]@HANDEDNESS,k.A@T[:3,3]+k.b));raw=float(fcl.distance(a['obj'],b['obj'],fcl.DistanceRequest(),fcl.DistanceResult()));assert np.isfinite(raw);distances.append(max(0,raw))
np.savez_compressed(out/'shoulder-slice.npz',joint2=values,distance=distances)
maximum=max(distances);half_step=float(np.diff(values).max()/2);numeric_allowance=1e-5;upper_bound=maximum+radius*half_step+numeric_allowance;goals=[]
for path in sorted((ROOT/'artifacts/compound-reaching/coupled-approach').glob('*.json')):
 row=json.loads(path.read_text());goals.append(dict(source=path.name,joint1_deg=float(np.rad2deg(row['trajectory']['approach_q'][0][0]))))
result=dict(robot_scene=args.robot_scene,pair=[a['link'],b['link']],joint1_slice_deg=110.,joint2_range_deg=np.rad2deg([lo,hi]).tolist(),joint_margin_deg=5,samples=len(values),maximum_sampled_distance_m=maximum,rotation_radius_bound_m=radius,maximum_half_step_rad=half_step,numerical_allowance_m=numeric_allowance,conservative_distance_upper_bound_m=upper_bound,distance_method='BVH triangle-to-triangle distance on the exact same convex-hull surfaces; convex GJK overestimation rejected',required_clearance_m=.005,barrier_confirmed=bool(upper_bound<.005),start_joint1_deg=float(np.rad2deg(k.start[0])),goals=goals,dependencies={a['link']:dependencies(a['link']),b['link']:dependencies(b['link'])},urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),scope='Fixed offline convex collision model and 5-degree joint margin. Distance Lipschitz bound uses maximum vertex radius around joint 2. This is not a statement about the physical robot or an unverified PhysX collider.')
(out/'shoulder-barrier.json').write_text(json.dumps(result,indent=2));print(result)
