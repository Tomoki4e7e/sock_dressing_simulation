#!/usr/bin/env python3
"""Compare outside ClosestPoint witnesses with input and ray-plane hulls."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from probe_compound_planes import solid_vertex_distance


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);ap.add_argument('--cycle',type=int,default=1);ap.add_argument('--cooked',type=Path);args=ap.parse_args();raw=json.loads((args.folder/f'cycle-{args.cycle}.raw.json').read_text());payload=json.loads((args.folder/f'cycle-{args.cycle}.payload.json').read_text());recovered=json.loads((args.folder/f'cycle-{args.cycle}.summary.json').read_text());results=[]
    cooked={p['name']:p for p in json.loads(args.cooked.read_text())['parts']} if args.cooked else {}
    for row,shape,plane in zip(raw['parts'],payload['parts'],recovered['parts']):
        scales=np.array([shape['scale'][a] for a in 'xyz']);original=np.array(row['vertices']);supports=row.get('supports',[])
        if not supports:continue
        points=np.array([q['point'] for q in supports]);origins=np.array([q['query'] for q in supports]);normal=np.array([q['normal'] for q in supports]) if 'normal' in supports[0] else origins-points;normal/=np.linalg.norm(normal,axis=1)[:,None];offset=np.sum(normal*points,axis=1)
        consistency=(normal@points.T-offset[:,None])/np.linalg.norm(normal/scales,axis=1)[:,None];physical=points*scales;expected=original*scales
        support_hull=physical[ConvexHull(physical).vertices];input_to_support=solid_vertex_distance(expected,support_hull);support_to_input=solid_vertex_distance(support_hull,expected)
        ranges=[]
        for multiplier in [1024,65536]:
            selected=np.array([q['point'] for q in supports if q['range_multiplier']==multiplier])*scales;hull=selected[ConvexHull(selected).vertices];ranges.append(hull)
        drift=max(solid_vertex_distance(ranges[0],ranges[1]).max(),solid_vertex_distance(ranges[1],ranges[0]).max())
        rec=dict(index=row['index'],variant=row['variant'],support_queries=len(supports),support_hull_vertices=support_hull.tolist(),input_outside_support_hull_m=float(input_to_support.max()),support_hull_outside_input_m=float(support_to_input.max()),support_range_hull_drift_m=float(drift),maximum_projection_support_violation_m=float(np.maximum(consistency,0).max()),ray_reconstruction_stable=plane['stable'],ray_hull_hausdorff_m=plane.get('reconstructed_solid_hausdorff_m'))
        if 'reconstructed_vertices' in plane:
            ray_hull=np.array(plane['reconstructed_vertices'])*scales;rec['ray_hull_outside_support_hull_m']=float(solid_vertex_distance(ray_hull,support_hull).max());rec['support_hull_outside_ray_hull_m']=float(solid_vertex_distance(support_hull,ray_hull).max())
        projections=[q for q in row.get('vertex_projections',[]) if q['offset_query_units']==0]
        if projections:
            displacement=[float(np.linalg.norm((np.array(q['query'])-q['point'])*scales)) for q in projections]
            rec['input_vertex_closest_displacement_m']=max(displacement)
            rec['vertex_closest_displacements_m']=displacement
        native=cooked.get(f"cook_{row['index']}_{row['variant']}")
        if native:
            native_vertices=np.array(native['vertices'])*scales
            rec['serialized_hull_to_support_m']=float(solid_vertex_distance(native_vertices,support_hull).max())
            rec['support_hull_to_serialized_m']=float(solid_vertex_distance(support_hull,native_vertices).max())
            rec['serialized_support_hausdorff_m']=max(rec['serialized_hull_to_support_m'],rec['support_hull_to_serialized_m'])
        results.append(rec);print(row['index'],row['variant'],'input->support',rec['input_outside_support_hull_m'],'support->input',rec['support_hull_outside_input_m'],'range drift',rec['support_range_hull_drift_m'],'projection inconsistency',rec['maximum_projection_support_violation_m'],flush=True)
    distances=[p['serialized_support_hausdorff_m'] for p in results if 'serialized_support_hausdorff_m' in p]
    summary=dict(parts=results,scope='Outside closest-point witnesses, not direct native vertex extraction',physics_executed=False)
    if distances:
        summary['serialized_comparison']=dict(cases=len(distances),maximum_hull_distance_m=max(distances),hull_mismatches_over_1um=sum(d>1e-6 for d in distances))
        print(summary['serialized_comparison'])
    (args.folder/'support-summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
