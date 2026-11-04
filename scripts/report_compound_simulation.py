#!/usr/bin/env python3
"""Compare isolated PhysX solver contacts with the affine convex input."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from report_compound_contacts import transformed_planes,sphere_penetration


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);args=ap.parse_args();raw=json.loads(args.source.read_text());rows=[];totals=Counter();by_variant={}
    for p in raw['parts']:
        local=np.array(p['vertices_local']);linear=Rotation.from_quat(p['world_rotation']).as_matrix()@np.diag(p['world_scale']);planes=transformed_planes(ConvexHull(local).equations,linear);points=local@linear.T;queries=p['contacts'];centers=np.array([q['center_shifted'] for q in queries]);radii=np.array([q['radius_m'] for q in queries]);expected=sphere_penetration(points,planes,centers,radii);counts=Counter();checks=[]
        for q,e in zip(queries,expected):
            depth=max([0.]+[-c['separation_m'] for c in q['contacts']]);error=abs(depth-max(0.,float(e)));observed=any(c['separation_m']<=0 for c in q['contacts']);mismatch=bool(abs(e)>1e-6 and observed!=(e>0));movement=float(np.linalg.norm(np.array(q['center_after'])-q['center_shifted']));counts['tests']+=1;counts['contact_callbacks']+=bool(q['contacts']);counts['presence_mismatch_over_1um']+=mismatch;counts['depth_error_over_01mm']+=error>1e-4;counts[q['kind']+'_tests']+=1;counts[q['kind']+'_presence_mismatch']+=mismatch
            if q['kind']!='interior':counts['surface_tests']+=1;counts['surface_presence_mismatch_over_1um']+=mismatch;counts['surface_depth_error_over_01mm']+=error>1e-4
            checks.append(dict(sample=q['sample'],kind=q['kind'],approach_step=q.get('approach_step',-1),radius_m=q['radius_m'],expected_depth_m=float(e),observed_depth_m=float(depth),contact_count=len(q['contacts']),presence_mismatch=mismatch,depth_error_m=error,movement_m=movement))
        rows.append(dict(index=p['index'],variant=p['variant'],counts=dict(counts),maximum_depth_error_m=max(c['depth_error_m'] for c in checks),checks=checks));totals.update(counts);by_variant.setdefault(p['variant'],Counter()).update(counts)
    result=dict(source=str(args.source),totals=dict(totals),by_variant={k:dict(v) for k,v in by_variant.items()},parts=rows,physics_executed=True,isolated_scene_only=True,main_scene_physics_executed=False,robot_motion_executed=False,geometry_gate_passed=False)
    if rows and all(p['index']<0 for p in rows):result['surface_contact_calibration_passed']=not totals['surface_presence_mismatch_over_1um'] and not totals['surface_depth_error_over_01mm']
    args.source.with_suffix('.summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['totals'],indent=2))


if __name__=='__main__':main()
