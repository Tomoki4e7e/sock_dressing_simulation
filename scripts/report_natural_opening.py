#!/usr/bin/env python3
"""Measure the actual ordered cloth boundary; do not infer aperture from camera width."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def records(path):
    lines=path.read_text().splitlines()
    rows=[]
    for i,line in enumerate(lines):
        try: rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1: raise
    return rows


def ordered_ring(row):
    opening=set(row['opening_particle_indices'])
    faces=np.asarray(row['particle_surface_triangles']).reshape(-1,3)
    counts=Counter(tuple(sorted((int(a),int(b)))) for t in faces
        for a,b in [(t[0],t[1]),(t[1],t[2]),(t[2],t[0])])
    neighbours={p:[] for p in opening}
    for (a,b),n in counts.items():
        if n==1 and a in opening and b in opening:
            neighbours[a].append(b);neighbours[b].append(a)
    if not opening or any(len(n)!=2 for n in neighbours.values()):
        raise ValueError('Opening is not one closed boundary ring')
    ring=[min(opening)];previous=-1
    while True:
        next_point=next(p for p in neighbours[ring[-1]] if p!=previous)
        if next_point==ring[0]:break
        if next_point in ring:raise ValueError('Opening contains multiple cycles')
        previous=ring[-1];ring.append(next_point)
    if len(ring)!=len(opening):raise ValueError('Opening contains multiple cycles')
    return ring


def measure(row,ring):
    points=np.asarray(row['particles_world'])[ring]
    points=points-points.mean(axis=0)
    cross=np.cross(points,np.roll(points,-1,axis=0)).sum(axis=0)*.5
    area=float(np.linalg.norm(cross))
    edges=np.roll(points,-1,axis=0)-points
    perimeter=float(np.linalg.norm(edges,axis=1).sum())
    normal=cross/area if area>1e-12 else np.zeros(3)
    projected=edges-np.outer(edges@normal,normal)
    turns=np.arctan2(np.cross(np.roll(projected,1,axis=0),projected)@normal,
        np.sum(np.roll(projected,1,axis=0)*projected,axis=1))
    opposite=np.linalg.norm(points[:len(ring)//2]-points[len(ring)//2:],axis=1) if len(ring)%2==0 else np.array([np.nan])
    safety=row['dressing_qa']['foot_collision_safety']
    return {'command':row['frame']+1,'area_mm2':area*1e6,
        'minimum_opposite_arc_distance_mm':float(np.min(opposite))*1000,
        'hydraulic_radius_mm':2*area/perimeter*1000,
        'perimeter_mm':perimeter*1000,'area_perimeter_index':4*np.pi*area/perimeter**2,
        'concave_turns':int(np.count_nonzero(turns < -1e-3)),
        'attached_sides':sum(bool(g['attached']) for g in row['grasp_state']),
        'anchor_error_mm':safety['maximum_original_grasp_anchor_error_m']*1000,
        'natural_bending':safety.get('natural_opening_rim_bending')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--candidate-label',default='V38')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    inputs={'V37':args.baseline}
    if args.candidate:inputs[args.candidate_label]=args.candidate
    groups={};summary={}
    for name,path in inputs.items():
        rows=records(path);ring=ordered_ring(rows[0]);values=[measure(r,ring) for r in rows]
        groups[name]=values
        areas=[v['area_mm2'] for v in values]
        summary[name]={'frames':len(rows),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'minimum_area_mm2':min(areas),'mean_area_mm2':float(np.mean(areas)),
            'final_area_mm2':areas[-1],'mean_area_perimeter_index':float(np.mean([v['area_perimeter_index'] for v in values])),
            'minimum_hydraulic_radius_mm':min(v['hydraulic_radius_mm'] for v in values),
            'mean_hydraulic_radius_mm':float(np.mean([v['hydraulic_radius_mm'] for v in values])),
            'minimum_opposite_arc_distance_mm':min(v['minimum_opposite_arc_distance_mm'] for v in values),
            'minimum_area_command':values[int(np.argmin(areas))]['command']}
    report={'metric':'magnitude of ordered boundary area vector; maximum oriented projected area, not a proof of foot passage or dressing',
        'comparison':'Independent replays, not identical intermediate states. Only completed diagnostic rows are included.',
        'summary':summary,'rows':groups}
    (args.output/'opening-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    fields=['version','command','area_mm2','perimeter_mm','area_perimeter_index','concave_turns','attached_sides','anchor_error_mm','minimum_opposite_arc_distance_mm','hydraulic_radius_mm']
    with (args.output/'opening-comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for name,values in groups.items():writer.writerows(dict(version=name,**v) for v in values)
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(args.output/'.plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
    for name,values in groups.items():
        x=[v['command'] for v in values]
        axes[0].plot(x,[v['area_mm2'] for v in values],label=name)
        axes[1].plot(x,[v['hydraulic_radius_mm'] for v in values],label=name)
        axes[2].plot(x,[v['minimum_opposite_arc_distance_mm'] for v in values],label=name)
    axes[0].set_ylabel('Ordered rim area vector (mm²)')
    axes[1].set_ylabel('2 area / perimeter (mm)');axes[2].set_ylabel('Min opposite-arc distance (mm)');axes[2].set_xlabel('Command number')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.tight_layout();fig.savefig(args.output/'opening-comparison.png',dpi=160)
    print(json.dumps(summary))


if __name__=='__main__':main()
