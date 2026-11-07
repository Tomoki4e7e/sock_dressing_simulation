#!/usr/bin/env python3
"""Measure Euclidean surface errors for the remaining shifted vertex rays."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--variant',default='pca1');args=ap.parse_args();raw=json.loads(args.source.read_text());summary=json.loads(args.source.with_suffix('.summary.json').read_text());checks=[];gates={}
    if len(raw['parts'])!=len(summary['parts']):raise ValueError('Raw and summary part counts differ')
    for part,report in zip(raw['parts'],summary['parts']):
        assert (part['index'],part['variant'])==(report['index'],report['variant'])
        reported={r['sample']:r for r in report['rays']}
        for ray in part['rays']:
            count=gates.setdefault(part['variant'],{}).setdefault(ray['kind'],Counter());count['rays']+=1;count['world_misses']+=not ray['hit'];count['shifted_misses']+=not ray['shifted_hit'];r=reported.get(ray['sample'])
            if r is not None:
                count['shifted_hit_mismatch']+=(r['shifted_interval'] is not None)!=r['shifted_hit'];count['shifted_distance_over_01mm']+=r['shifted_error_m'] is not None and r['shifted_error_m']>1e-4
                count['world_pose_hit_mismatch']+=(r['intervals']['pose'] is not None)!=r['hit'];count['world_pose_distance_over_01mm']+=r['errors_m']['pose'] is not None and r['errors_m']['pose']>1e-4
        if part['variant']!=args.variant:continue
        bad=[r for r in report['rays'] if r['shifted_error_m'] is not None and r['shifted_error_m']>1e-4]
        if not bad:continue
        local=np.array(part['vertices_local']);linear=Rotation.from_quat(part['world_rotation']).as_matrix()@np.diag(part['world_scale']);points=local@linear.T;surface=trimesh.Trimesh(points,ConvexHull(points).simplices,process=False);rays={r['sample']:r for r in part['rays']}
        for b in bad:
            ray=rays[b['sample']];direction=np.array(ray['direction_shifted']);direction/=np.linalg.norm(direction);hit=np.array(ray['origin_shifted'])+direction*ray['shifted_distance_m'];distance=trimesh.proximity.closest_point(surface,hit[None])[1][0]
            checks.append(dict(index=part['index'],sample=ray['sample'],kind=ray['kind'],radial_error_m=b['shifted_error_m'],hit_surface_distance_m=float(distance)))
    result=dict(source=str(args.source),variant=args.variant,rays=len(checks),parts=len(set(r['index'] for r in checks)),maximum_hit_surface_distance_m=max((r['hit_surface_distance_m'] for r in checks),default=0),surface_distance_over_01mm=sum(r['hit_surface_distance_m']>1e-4 for r in checks),checks=checks,ray_gates=gates,geometry_gate_passed=False)
    output=args.output or args.source.with_suffix('.residuals.json');output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
