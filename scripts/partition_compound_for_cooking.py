#!/usr/bin/env python3
"""Exact convex subdivision of convex pieces to make PhysX cooking reliable."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,copy
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import ConvexHull,QhullError
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/compound-reaching'

def hull(points):
    points=np.unique(points,axis=0);center=points.mean(0);scale=np.ptp(points,axis=0).max();normalized=(points-center)/scale
    ch=ConvexHull(normalized,qhull_options='C-1e-8');faces=ch.simplices.copy()
    for f,plane in zip(faces,ch.equations):
        a,b,c=normalized[f]
        if np.dot(np.cross(b-a,c-a),plane[:3])<0:f[1],f[2]=f[2],f[1]
    result=trimesh.Trimesh(points,faces,process=True);result.remove_unreferenced_vertices();return result

def split(mesh,normal,offset):
    p=np.array(mesh.vertices);d=p@normal-offset;edges=mesh.edges_unique;mask=d[edges[:,0]]*d[edges[:,1]]< -1e-24;edges=edges[mask];a=p[edges[:,0]];b=p[edges[:,1]];alpha=-d[edges[:,0]]/(d[edges[:,1]]-d[edges[:,0]]);cross=a+alpha[:,None]*(b-a);cross-=(cross@normal-offset)[:,None]*normal
    return hull(np.r_[p[d<=0],cross]),hull(np.r_[p[d>=0],cross])

def partition(mesh,limit=24,depth=0):
    if len(mesh.vertices)<=limit:return [mesh]
    if depth>20:raise RuntimeError('Subdivision did not converge')
    p=np.array(mesh.vertices);_,axes=np.linalg.eigh(np.cov(p.T));axes=np.r_[axes.T[::-1],np.eye(3)];best=None
    for normal in axes:
        projection=p@normal
        for offset in [(projection.min()+projection.max())/2,float(np.median(projection))]:
            try:a,b=split(mesh,normal,offset)
            except QhullError:continue
            score=(max(len(a.vertices),len(b.vertices)),len(a.vertices)+len(b.vertices))
            if best is None or score<best[0]:best=(score,a,b)
    if best is None:raise RuntimeError('No nondegenerate partition cut')
    _,a,b=best
    if abs(a.volume+b.volume-mesh.volume)>max(1e-12,abs(mesh.volume)*1e-6):raise RuntimeError('Subdivision changed volume')
    return partition(a,limit,depth+1)+partition(b,limit,depth+1)

def main():
    src=OUT/'decomposition.json';old=json.loads(src.read_text());rows=[];checks=[]
    for row in old['parts']:
        data=json.loads((ROOT/row['parts_file']).read_text());pieces=[];maxerr=0.;maxoutside=0.
        for i,part in enumerate(data['parts']):
            parent=hull(np.array(part['vertices']));children=partition(parent);error=abs(sum(h.volume for h in children)-parent.volume);maxerr=max(maxerr,error);planes=ConvexHull(parent.vertices).equations
            for h in children:
                maxoutside=max(maxoutside,float((h.vertices@planes[:,:3].T+planes[:,3]).max()));pieces.append(dict(vertices=h.vertices.tolist(),faces=h.faces.tolist(),parent_part=i))
        path=OUT/'geometry'/row['link'].replace('/','-')/'cook24-parts.json';path.write_text(json.dumps(dict(link=row['link'],parts=pieces,partition_vertex_limit=24,partition_method='Balanced half-space cuts with normalized QHull coplanar tolerance 1e-8; volume and parent containment checked',source_parts=row['parts_file'])))
        updated=dict(row);updated.update(parts=len(pieces),parts_file=str(path.relative_to(ROOT)),maximum_part_vertices=max(len(p['vertices']) for p in pieces),maximum_part_triangles=max(len(p['faces']) for p in pieces));rows.append(updated);checks.append(dict(link=row['link'],parents=len(data['parts']),children=len(pieces),maximum_volume_sum_error_m3=maxerr,maximum_child_vertex_outside_parent_plane_m=maxoutside));print(checks[-1],flush=True)
    report=dict(old);report.update(parts=rows,subdivision=checks,partition_vertex_limit=24,passed=old['passed'] and all(r['maximum_child_vertex_outside_parent_plane_m']<1e-8 for r in checks));(OUT/'decomposition-cook24.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
