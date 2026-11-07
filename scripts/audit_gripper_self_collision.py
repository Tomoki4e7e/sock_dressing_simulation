#!/usr/bin/env python3
"""Diagnose convex wrist contacts against original URDF triangle surfaces."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
import numpy as np,trimesh,fcl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load,good,OUT
from sock_dressing_simulation.arm_reaching import Kinematics,triangle_object
from sock_dressing_simulation.grasp_geometry import origin,HANDEDNESS

def main():
 k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',load(39));objects={};meta={}
 for name in ['right_gripper/gripper_base','right_arm/link_5']:
  link=k.tree.find(f"link[@name='{name}']");objects[name]=[];meta[name]=[]
  for col in link.findall('collision'):
   mesh=col.find('geometry/mesh');assert mesh is not None
   path=k.path.parent/mesh.get('filename');m=trimesh.load_mesh(str(path),process=False);T=origin(col.find('origin'));p=np.array(m.vertices)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ');p=(p@T[:3,:3].T+T[:3,3])@HANDEDNESS
   objects[name].append(triangle_object(p,np.array(m.faces)[:,::-1]));meta[name].append(dict(mesh=str(path.relative_to(ROOT)),watertight_after_vertex_merge=bool(trimesh.Trimesh(m.vertices,m.faces,process=True).is_watertight)))
 rows=[]
 for path in sorted(OUT.glob('target-*.json')):
  for i,a in enumerate(json.loads(path.read_text())['attempts']):
   if not good(a):continue
   issues=a.get('endpoint_collision',{}).get('issues',[])
   if not any(set(v['pair'])==set(objects) for v in issues):continue
   frames=k.fk(a['q']);
   for name,objs in objects.items():
    T=frames[name]
    for obj in objs:obj.setTransform(fcl.Transform(k.A@T[:3,:3]@HANDEDNESS,k.A@T[:3,3]+k.b))
   ds=[float(fcl.distance(x,y,fcl.DistanceRequest(),fcl.DistanceResult())) for x in objects['right_gripper/gripper_base'] for y in objects['right_arm/link_5']]
   rows.append(dict(target=path.name,attempt=i,original_triangle_surface_distance_m=max(0,min(ds)),original_surfaces_intersect=min(ds)<=0))
   if len(rows)>=24:break
  if len(rows)>=24:break
 out=dict(scope='First 24 convex wrist failures, original URDF triangle surface check; separation does not exclude full containment',meshes=meta,checks=rows,original_surface_intersections=sum(r['original_surfaces_intersect'] for r in rows))
 (OUT/'wrist-original-mesh-audit.json').write_text(json.dumps(out,indent=2));print('checks',len(rows),'original intersections',out['original_surface_intersections'])
if __name__=='__main__':main()
