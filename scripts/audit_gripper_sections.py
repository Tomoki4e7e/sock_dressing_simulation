#!/usr/bin/env python3
"""Compare original URDF triangle cross sections with the planning convex hull."""
import sys,json
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import RightHand,origin
h=RightHand(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf');rows=[]
for name in ['right_gripper/left_finger_link','right_gripper/right_finger_link']:
    link=h.tree.find(f"link[@name='{name}']");col=link.find('collision');el=col.find('geometry/mesh');mesh=trimesh.load_mesh(str(h.urdf.parent/el.get('filename')),process=False);O=origin(col.find('origin'));T=h.link_transform(name,h.joint_for_gap(.05))@O
    points=(np.array(mesh.vertices)*np.fromstring(el.get('scale','1 1 1'),sep=' '))@T[:3,:3].T+T[:3,3];points=points[:,[0,2,1]];tri=points[mesh.faces];planes=ConvexHull(points).equations
    for dx in [-.004,0,.004]:
      for dy in [-.004,0,.004]:
        x,y=h.grasp_center_urdf[[0,2]]+[dx,dy];a=tri[:,0,:2];u=tri[:,1,:2]-a;v=tri[:,2,:2]-a;det=u[:,0]*v[:,1]-u[:,1]*v[:,0];ok=abs(det)>1e-14
        s=np.zeros(len(tri));t=s.copy();w=np.array([x,y])-a;s[ok]=(w[ok,0]*v[ok,1]-w[ok,1]*v[ok,0])/det[ok];t[ok]=(u[ok,0]*w[ok,1]-u[ok,1]*w[ok,0])/det[ok];hit=ok&(s>=-1e-8)&(t>=-1e-8)&(s+t<=1+1e-8)
        z=tri[:,0,2]+s*(tri[:,1,2]-tri[:,0,2])+t*(tri[:,2,2]-tri[:,0,2]);z=z[hit];b=planes[:,0]*x+planes[:,1]*y+planes[:,3];lo=planes[:,2]<-1e-10;hi=planes[:,2]>1e-10;l=max(-b[lo]/planes[lo,2]);r=min(-b[hi]/planes[hi,2]);rows.append(dict(link=name,dx_m=dx,width_offset_m=dy,triangle_intersections_closing_axis_m=sorted(set(np.round(z,10))),hull_interval_closing_axis_m=[l,r],hull_thickness_m=r-l,outer_boundary_difference_m=None if not len(z) else max(abs(l-z.min()),abs(r-z.max()))))
(ROOT/'artifacts/translated-reaching/gripper-sections.json').write_text(json.dumps(rows,indent=2));assert all(r['triangle_intersections_closing_axis_m'] and r['hull_thickness_m']>0 for r in rows);print('maximum outer boundary discrepancy (mm)',max(r['outer_boundary_difference_m'] for r in rows)*1000)
