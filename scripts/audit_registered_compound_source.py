#!/usr/bin/env python3
"""Bound the source-union change caused by conditioned registration."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from prepare_registered_compound import tetrahedra,ROOT
from sock_dressing_simulation.arm_reaching import ROS_UNITY


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args();manifest=json.loads((ROOT/'artifacts/compound-reaching/decomposition.json').read_text());payload=json.loads((args.folder/'payload.json').read_text())['parts'];originals=[];checks=[]
    for row in manifest['parts']:
        for part in json.loads((ROOT/row['parts_file']).read_text())['parts']:originals.append(np.array(part['vertices'])@ROS_UNITY.T)
    for parent,points in enumerate(originals):
        shapes=[p for p in payload if p['parent_index']==parent];pieces=tetrahedra(points) if shapes[0]['tetrahedron'] else [points]
        if len(shapes)!=len(pieces):raise ValueError('Missing parent pieces')
        error=0.
        for source,shape in zip(pieces,shapes):
            vertices=np.array([[p[a] for a in 'xyz'] for p in shape['vertices']]);scale=np.array([shape['scale'][a] for a in 'xyz']);position=np.array([shape['position'][a] for a in 'xyz']);R=Rotation.from_quat([shape['rotation'][a] for a in 'xyzw']).as_matrix();physical=(vertices*scale)@R.T+position;error=max(error,float(np.linalg.norm(physical-source,axis=1).max()))
        volume_error=abs(sum(ConvexHull(p).volume for p in pieces)-ConvexHull(points).volume)
        checks.append(dict(parent=parent,pieces=len(pieces),maximum_corresponding_vertex_change_m=error,preconditioning_volume_sum_error_m3=float(volume_error)))
    result=dict(passed=len(checks)==len(originals) and max(p['maximum_corresponding_vertex_change_m'] for p in checks)<2e-5 and max(p['preconditioning_volume_sum_error_m3'] for p in checks)<1e-12,parents=len(checks),registered_parts=len(payload),maximum_union_hausdorff_bound_m=max(p['maximum_corresponding_vertex_change_m'] for p in checks),maximum_preconditioning_volume_error_m3=max(p['preconditioning_volume_sum_error_m3'] for p in checks),proof='Each convex source is the union of its center-to-facet tetrahedra. Corresponding-vertex displacement bounds each convex piece in both directions, hence also their unions.',parts=checks)
    result['payload_sha256']=hashlib.sha256((args.folder/'payload.json').read_bytes()).hexdigest()
    (args.folder/'source-change-audit.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='parts'})


if __name__=='__main__':main()
