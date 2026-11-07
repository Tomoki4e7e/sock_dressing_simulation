#!/usr/bin/env python3
"""Decompose only palm and forearm, recording original-surface coverage errors."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib,inspect
from pathlib import Path
import numpy as np,trimesh,coacd
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import origin
import xml.etree.ElementTree as ET
OUT=ROOT/'artifacts/compound-reaching';URDF=ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf'

def local_mesh(name):
    tree=ET.parse(URDF).getroot();meshes=[]
    for col in tree.find(f"link[@name='{name}']").findall('collision'):
        e=col.find('geometry/mesh');m=trimesh.load_mesh(str(URDF.parent/e.get('filename')),process=True);T=origin(col.find('origin'));m.vertices=np.array(m.vertices)*np.fromstring(e.get('scale','1 1 1'),sep=' ');m.apply_transform(T);meshes.append(m)
    return trimesh.util.concatenate(meshes)

def outside_distance(points,hulls):
    # Exact distance to a union of closed convex solids: zero for inside points.
    best=np.full(len(points),np.inf)
    for h in hulls:
        eq=ConvexHull(h.vertices).equations;inside=(points@eq[:,:3].T+eq[:,3]).max(1)<=1e-9
        best[inside]=0;mask=(~inside)&(best>0)
        if mask.any():best[mask]=np.minimum(best[mask],trimesh.proximity.closest_point(h,points[mask])[1])
    return best

def main():
    OUT.mkdir(exist_ok=True);rows=[];params=dict(threshold=.01,preprocess_mode='auto',preprocess_resolution=100,resolution=5000,mcts_nodes=20,mcts_iterations=100,mcts_max_depth=3,merge=True,decimate=True,max_ch_vertex=64,seed=20261105)
    coacd.set_log_level('warn')
    for name in ['right_gripper/gripper_base','right_arm/link_5']:
        dest=OUT/'geometry'/name.replace('/','-');dest.mkdir(parents=True,exist_ok=True);m=local_mesh(name)
        selected=dict(params)
        if name=='right_arm/link_5':selected.update(threshold=.015,preprocess_resolution=50,resolution=2000,mcts_nodes=10,mcts_iterations=40,mcts_max_depth=2)
        cache=dest/('fast-parts.json' if name=='right_arm/link_5' else 'parts.json')
        if cache.exists():data=json.loads(cache.read_text());parts=[trimesh.Trimesh(p['vertices'],p['faces'],process=True).convex_hull for p in data['parts']]
        else:
            parts=[trimesh.Trimesh(v,f,process=True).convex_hull for v,f in coacd.run_coacd(coacd.Mesh(m.vertices,m.faces),**selected)]
            cache.write_text(json.dumps(dict(link=name,parameters=selected,parts=[dict(vertices=h.vertices.tolist(),faces=h.faces.tolist()) for h in parts]),indent=2))
        np.random.seed(20261105);sample=np.r_[m.vertices,trimesh.sample.sample_surface(m,12000)[0]];coverage=outside_distance(sample,parts)
        # Distance from decomposed surfaces to the original is diagnostic only:
        # internal partition faces legitimately have a large surface distance.
        sample2=np.concatenate([trimesh.sample.sample_surface(h,300)[0] for h in parts]);original_dist=trimesh.proximity.closest_point(m,sample2)[1]
        original_hull=m.convex_hull;volume_sum=sum(abs(h.volume) for h in parts)
        for i,h in enumerate(parts):h.export(dest/f'part-{i:03d}.stl')
        row=dict(parameters=selected,link=name,parts=len(parts),original_watertight=m.is_watertight,original_hull_volume_m3=abs(original_hull.volume),sum_part_volumes_m3=volume_sum,volume_fraction_of_single_hull=volume_sum/abs(original_hull.volume),coverage_samples=len(sample),maximum_original_surface_outside_union_m=float(coverage.max()),p99_original_surface_outside_union_m=float(np.percentile(coverage,99)),sampled_partition_surface_to_original_distance_max_m=float(original_dist.max()),maximum_part_vertices=max(len(h.vertices) for h in parts),maximum_part_triangles=max(len(h.faces) for h in parts),all_parts_convex=all(h.is_convex for h in parts),parts_file=str(cache.relative_to(ROOT)),source_sha256=hashlib.sha256(np.array(m.vertices).tobytes()+np.array(m.faces).tobytes()).hexdigest())
        row['geometry_gate_passed']=row['all_parts_convex'] and row['maximum_part_triangles']<=255 and row['maximum_original_surface_outside_union_m']<=.002
        rows.append(row);print(row,flush=True)
    result=dict(parts=rows,parameters=params,coverage_tolerance_m=.002,passed=all(r['geometry_gate_passed'] for r in rows),note='Finite surface sampling, not a continuous enclosure proof. Open forearm is preprocessed by CoACD; repair and inner solid are approximations. Original URDF unchanged.')
    (OUT/'decomposition.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
