#!/usr/bin/env python3
"""Prepare conditioned colliders, using tetrahedra only where cooking failed."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import ROS_UNITY
from diagnose_compound_geometry import conditioned_part,vector


def condition(points,index,link,minimum_width):
    shape=conditioned_part(dict(link=link,vertices=[vector(v) for v in points]),index,'pca1')
    scale=np.array([shape['scale'][a] for a in 'xyz']);new_scale=np.maximum(scale,minimum_width)
    shape['scale']=vector(new_scale);shape['conditioned']=True
    R=Rotation.from_quat([shape['rotation'][a] for a in 'xyzw']).as_matrix();center=np.array([shape['position'][a] for a in 'xyz']);v=np.array([[v[a] for a in 'xyz'] for v in shape['vertices']]);physical=(v*new_scale)@R.T+center
    return shape,physical,float(np.linalg.norm(physical-points,axis=1).max())


def tetrahedra(points):
    # A convex solid is exactly the union of pyramids from a strict interior
    # point to each outward boundary triangle (four vertices per pyramid).
    center=points.mean(0);return [np.r_[points[face],center[None,:]] for face in ConvexHull(points).simplices]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'artifacts/compound-reaching/registration/candidate');ap.add_argument('--minimum-width',type=float,default=2e-5);ap.add_argument('--previous-comparison',type=Path);ap.add_argument('--parent-manifest',type=Path,help='Reuse recorded tetrahedral parent choices for exact reproduction');ap.add_argument('--threshold',type=float,default=9e-5);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    base=ROOT/'artifacts/compound-reaching';manifest=json.loads((base/'decomposition.json').read_text());originals=[]
    for row in manifest['parts']:
        for part in json.loads((ROOT/row['parts_file']).read_text())['parts']:originals.append((row['link'],np.array(part['vertices'])@ROS_UNITY.T))
    baseline=json.loads((base/'followup/cooked-mesh-all/comparison.json').read_text())['parts'];fallback={p['index'] for p in baseline if p['variant']=='pca1' and (not p['serialized_plane_hull_closed'] or max(p['cooked_vertex_hull_hausdorff_m'],p['serialized_plane_hull_hausdorff_m'])>args.threshold)}
    if args.parent_manifest:fallback=set(json.loads(args.parent_manifest.read_text())['tetrahedral_parent_indices'])
    if args.previous_comparison:
        previous=json.loads(args.previous_comparison.read_text());old_payload=json.loads((args.previous_comparison.parent/'payload.json').read_text())['parts']
        for p in previous['parts']:
            if not p['serialized_plane_hull_closed'] or max(p['cooked_vertex_hull_hausdorff_m'],p['serialized_plane_hull_hausdorff_m'])>args.threshold:fallback.add(old_payload[p['index']]['parent_index'])
    parts=[];by_link={row['link']:[] for row in manifest['parts']};maximum_change=0
    for parent,(link,points) in enumerate(originals):
        pieces=tetrahedra(points) if parent in fallback else [points]
        for piece in pieces:
            shape,physical,change=condition(piece,len(parts),link,args.minimum_width);shape['parent_index']=parent;shape['tetrahedron']=parent in fallback;parts.append(shape);maximum_change=max(maximum_change,change)
            hull=ConvexHull(physical);by_link[link].append(dict(vertices=(physical@ROS_UNITY).tolist(),faces=hull.simplices.tolist(),parent_part=parent))
    output_manifest=copy.deepcopy(manifest)
    for row in output_manifest['parts']:
        path=args.output/(row['link'].replace('/','-')+'-parts.json');path.write_text(json.dumps(dict(link=row['link'],parts=by_link[row['link']])));row.update(parts=len(by_link[row['link']]),parts_file=str(path.relative_to(ROOT)),maximum_part_vertices=max(len(p['vertices']) for p in by_link[row['link']]),maximum_part_triangles=max(len(p['faces']) for p in by_link[row['link']]),maximum_original_surface_outside_union_m=row['maximum_original_surface_outside_union_m']+maximum_change)
    output_manifest.update(registration='PCA with selective tetrahedral fallback and minimum principal width',maximum_conditioning_vertex_displacement_m=maximum_change,minimum_principal_width_m=args.minimum_width,tetrahedral_parent_indices=sorted(fallback))
    (args.output/'manifest.json').write_text(json.dumps(output_manifest,indent=2)+'\n');(args.output/'payload.json').write_text(json.dumps(dict(parts=parts,registrationVersion=2,cookUnitScale=1)))
    print('prepared',len(parts),'parts;',len(fallback),'tetrahedral parents; maximum stretch displacement',maximum_change,flush=True)


if __name__=='__main__':main()
