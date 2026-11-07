#!/usr/bin/env python3
"""Certify a common strict interior for source, cooked vertices and planes."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


def common_interior(source,vertices,planes,scale):
    equations=np.r_[ConvexHull(source).equations,ConvexHull(vertices).equations,np.array(planes)]
    largest=max(scale);normals=equations[:,:3]*largest/scale;lengths=np.linalg.norm(normals,axis=1);normals/=lengths[:,None];offset=equations[:,3]/lengths
    fit=linprog([0,0,0,-1],A_ub=np.c_[normals,np.ones(len(normals))],b_ub=-offset,bounds=[(None,None)]*3+[(0,None)],method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    if not fit.success or fit.x[3]<=0:raise ValueError('No common strict interior')
    return fit.x[:3]*largest/scale,float(fit.x[3]*largest)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args();payload=json.loads((args.folder/'payload.json').read_text());cooked={p['name']:p for p in json.loads((args.folder/'decoded.json').read_text())['parts']};records=[]
    for part in payload['parts']:
        native=cooked[f"cook_{part['index']}_{part['variant']}"];source=np.array([[p[a] for a in 'xyz'] for p in part['vertices']]);scale=np.array([part['scale'][a] for a in 'xyz']);point,radius=common_interior(source,native['vertices'],native['planes'],scale);part['interior']=dict(zip('xyz',map(float,point)));records.append(dict(index=part['index'],radius_m=radius))
    (args.folder/'payload.json').write_text(json.dumps(payload));(args.folder/'interiors.json').write_text(json.dumps(dict(parts=records,minimum_radius_m=min(p['radius_m'] for p in records)),indent=2)+'\n');print('common interiors',len(records),'minimum radius m',min(p['radius_m'] for p in records))


if __name__=='__main__':main()
