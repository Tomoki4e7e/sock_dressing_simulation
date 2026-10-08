#!/usr/bin/env python3
"""Build a fixed-pose convex foot solid covering the audited human surface.

The display mesh, pose and cloth limits are preserved. Original box bounds
are retained in diagnostics; the calf retains its existing collider.
This produces a candidate asset; it does not certify a dressing result.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def build(skin: dict, frame: dict) -> dict:
    if not skin.get('recorded_bone_pose_validated') or skin.get('maximum_baked_skin_error_m', 1) > 1e-5:
        raise ValueError('Human skin reconstruction must match recorded bones and Unity BakeMesh')
    if skin['selected_vertices'] != skin['expected_selected_vertices'] or skin['complete_vertices'] != skin['selected_vertices']:
        raise ValueError('Incomplete human skin reconstruction')
    vertices=np.array([[v['world'][axis] for axis in ('x','y','z')] for v in skin['vertices']])
    support=np.array([[v[axis] for axis in ('x','y','z')] for v in skin.get('support_vertices_world',[])])
    if not len(support):support=vertices
    hull=ConvexHull(support)
    # Ten micrometres outward protects skin coverage against float roundoff.
    centroid=support[hull.vertices].mean(axis=0)
    interior_radius=float(np.min(-(hull.equations[:,:3]@centroid+hull.equations[:,3])))
    if interior_radius<=1e-6:raise ValueError('Degenerate foot skin envelope')
    enlargement=1+1e-5/interior_radius
    # A uniform outward expansion keeps the original hull's topology.
    # Intersecting shifted planes instead can introduce edges that collapse
    # when converted to Unity float coordinates.
    padded_corners=centroid+(support[hull.vertices]-centroid)*enlargement
    regions=[]; world_solids=[];original_world_boxes=[]
    for box in frame['dressing_qa']['foot_collision_safety']['foot_collider_geometry']:
        rotation=Rotation.from_quat(box['rotation']);matrix=rotation.as_matrix()
        scale=np.array(box['scale']);position=np.array(box['position']);center=np.array(box['center']);half=np.array(box['size'])/2
        world_center=position+matrix@(center*scale)
        box_planes=[]
        for axis in range(3):
            for sign in (-1,1):
                normal=matrix[:,axis]*sign
                box_planes.append(np.r_[normal,-normal@world_center-half[axis]*abs(scale[axis])])
        original_world_boxes.append(np.array(box_planes))
        if box['region']!='forefoot':continue
        # One continuous hull avoids both duplicate overlapping native
        # constraints and seams that vertex-only box coverage can miss.
        corners=padded_corners
        solid=ConvexHull(corners);triangles=[]
        for face,equation in zip(solid.simplices,solid.equations):
            a,b,c=map(int,face)
            if np.dot(np.cross(corners[b]-corners[a],corners[c]-corners[a]),equation[:3])<0:b,c=c,b
            triangles.extend([a,b,c])
        local=(rotation.inv().apply(corners-position)/scale).astype(np.float32)
        faces=np.array(triangles).reshape(-1,3)
        areas=np.linalg.norm(np.cross(local[faces[:,1]]-local[faces[:,0]],local[faces[:,2]]-local[faces[:,0]]),axis=1)
        if np.any(areas<1e-12):raise ValueError('Foot mesh has a face that degenerates at Unity float precision')
        world_solids.append(solid.equations)
        regions.append(dict(region=box['region'],vertices=[dict(zip(('x','y','z'),map(float,p))) for p in local],triangles=triangles,
                            reference_position=box['position'],reference_rotation=box['rotation'],reference_scale=box['scale']))
    def covered(point):
        return any(np.max(s[:,:3]@point+s[:,3])<=2e-6 for s in world_solids)
    uncovered=sum(not covered(v) for v in vertices)
    # Include triangle interiors, so covering vertices cannot conceal gaps
    # between adjacent region solids. Audit each selected skin triangle on
    # an eleven-sample barycentric lattice, independently of cloth contact.
    samples=[]
    for a,b,c in np.array(skin['triangles']).reshape(-1,3):
        for i in range(11):
            for j in range(11-i):
                samples.append((vertices[a]*i+vertices[b]*j+vertices[c]*(10-i-j))/10)
    uncovered_faces=sum(not covered(p) for p in samples)
    original_uncovered_faces=sum(not any(np.max(s[:,:3]@p+s[:,3])<=2e-6 for s in original_world_boxes) for p in samples)
    support_samples=[]
    for a,b,c in np.array(skin.get('support_triangles',[])).reshape(-1,3):
        for i in range(11):
            for j in range(11-i):support_samples.append((support[a]*i+support[b]*j+support[c]*(10-i-j))/10)
    uncovered_support=sum(not covered(p) for p in support_samples)
    if uncovered or uncovered_faces or uncovered_support:
        raise ValueError(f'Candidate leaves human skin uncovered: vertices={uncovered}, face_samples={uncovered_faces}')
    return dict(schema='surface-covering-foot-solids-v1',regions=regions,
                audit=dict(skin_vertex_count=len(vertices),uncovered_skin_vertices=uncovered,
                           skin_triangle_sample_count=len(samples),uncovered_skin_triangle_samples=uncovered_faces,
                           support_skin_vertex_count=len(support),support_skin_triangle_sample_count=len(support_samples),
                           uncovered_support_skin_triangle_samples=uncovered_support,
                           outward_skin_padding_m=1e-5,original_box_bounds_retained_as_diagnostics=True,
                           maximum_hull_vertex_outward_displacement_m=float(np.linalg.norm(padded_corners-support[hull.vertices],axis=1).max()),
                           original_box_union_uncovered_skin_triangle_samples=original_uncovered_faces,
                           status='candidate_geometry_only'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skin',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();raw=args.skin.read_bytes();skin=json.loads(raw)
    frame=json.loads(Path(skin['source_frame']).read_text().splitlines()[0])
    result=build(skin,frame);result['source_skin_sha256']=hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['audit']))


if __name__=='__main__':main()
