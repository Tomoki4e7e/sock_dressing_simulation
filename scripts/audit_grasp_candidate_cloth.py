#!/usr/bin/env python3
"""Check photographed hand convex pieces against the actual cloth triangles."""
import json,sys
from pathlib import Path
import numpy as np
import fcl
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody

def main():
 directory=ROOT/'artifacts/grasp-candidate-photos';photos=json.loads((directory/'report.json').read_text());state=json.loads((directory/'state.json').read_text())
 v=np.asarray(state['particles_world']);faces=np.asarray(state['triangles'],dtype=np.int32).reshape(-1,3)
 mesh=fcl.BVHModel();mesh.beginModel(len(faces),len(v));mesh.addSubModel(v,faces);mesh.endModel();cloth=fcl.CollisionObject(mesh)
 prior=json.loads((ROOT/'artifacts/right-grasp-geometry/pose-01/report.json').read_text());hand=RightHand(ROOT/prior['settings']['urdf']);skin=SkinBody(state['source']['skin_surface']);rows=[]
 for photo in photos['candidates']:
  c=photo['candidate'];R=np.asarray(c['rotation_matrix']);p=np.asarray(c['grasp_center_unity_world_m']);normal=np.asarray(photo['sample']['normal']);parts=[];pads=[]
  for part in hand.bank(photo['gap_m']):
   part.obj.setTransform(fcl.Transform(R,p));result=fcl.CollisionResult();count=fcl.collide(part.obj,cloth,fcl.CollisionRequest(num_max_contacts=1000,enable_contact=True),result)
   distance=0. if count else float(fcl.distance(part.obj,cloth,fcl.DistanceRequest(),fcl.DistanceResult()))
   if distance<0:raise RuntimeError('Invalid unsigned cloth distance')
   parts.append(dict(link=part.label,cloth_triangle_contact_count=count,cloth_surface_distance_m=distance))
   if part.label.endswith('/left_finger_link') or part.label.endswith('/right_finger_link'):
    planes=ConvexHull(part.vertices).equations;a=planes[:,1];lo=float(np.max(-planes[a< -1e-8,3]/a[a< -1e-8]));hi=float(np.min(-planes[a>1e-8,3]/a[a>1e-8]));value=lo if lo>0 else hi;point=p+R[:,1]*value;offset=float(np.dot(point-p,normal))
    pads.append(dict(link=part.label,pad_axis_coordinate_m=value,pad_point_world=point.tolist(),signed_offset_from_local_cloth_plane_m=offset,side='cloth_inside_toward_foot' if offset<0 else 'cloth_outside',inside_skin=bool(skin.winding(point[None])[0]>=.5)))
  sides=sorted(x['side'] for x in pads)
  row=dict(name=photo['name'],gap_m=photo['gap_m'],skin_clear=photo['skin_clear'],skin_clearance_m=photo['skin_clearance_m'],parts=parts,pad_points=pads,
   cloth_between_pad_points_local=sides==['cloth_inside_toward_foot','cloth_outside'] and not any(x['inside_skin'] for x in pads),
   hand_cloth_no_intersection=not any(x['cloth_triangle_contact_count'] for x in parts),
   limitation='Local tangent-plane test at the target only; no closure, physical capture or retention test.')
  rows.append(row);print(json.dumps(row),flush=True)
 (directory/'cloth-audit.json').write_text(json.dumps(dict(candidates=rows,contact_count_is_geometric_not_physical=True),indent=2)+'\n')
if __name__=='__main__':main()
