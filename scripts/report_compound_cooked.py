#!/usr/bin/env python3
"""Compare actual serialized convex vertices and planes to their cook input."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from probe_compound_planes import solid_vertex_distance


def compare_cooked(decoded,shape):
    vertices=np.array(decoded['vertices']);source=np.array([[v[a] for a in 'xyz'] for v in shape['vertices']]);scale=np.array([shape['scale'][a] for a in 'xyz']);planes=np.array(decoded['planes']);physical=vertices*scale;expected=source*scale
    outward=solid_vertex_distance(physical,expected);inward=solid_vertex_distance(expected,physical)
    normal_lengths=np.linalg.norm(planes[:,:3],axis=1)
    options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9}
    fit=linprog([0,0,0,-1],A_ub=np.c_[planes[:,:3],normal_lengths],b_ub=-planes[:,3],bounds=[(None,None)]*4,method='highs',options=options)
    bounded=bool(fit.success and fit.x[3]>0);plane_hausdorff=None
    if bounded:
        # A polyhedron can have a finite inradius and still be unbounded.
        for direction in np.r_[np.eye(3),-np.eye(3)]:
            bound=linprog(direction,A_ub=planes[:,:3],b_ub=-planes[:,3],bounds=[(None,None)]*3,method='highs',options=options)
            if not bound.success:bounded=False;break
    if bounded:
        plane_vertices=HalfspaceIntersection(planes,fit.x[:3]).intersections*scale
        plane_hausdorff=float(max(solid_vertex_distance(plane_vertices,expected).max(),solid_vertex_distance(expected,plane_vertices).max()))
    residual=(vertices@planes[:,:3].T+planes[:,3])/np.linalg.norm(planes[:,:3]/scale,axis=1)
    return dict(index=shape['index'],variant=shape['variant'],input_vertices=len(source),cooked_vertices=len(vertices),cooked_vertex_hull_hausdorff_m=float(max(inward.max(),outward.max())),input_outside_cooked_vertex_hull_m=float(inward.max()),cooked_vertex_hull_outside_input_m=float(outward.max()),serialized_plane_hull_closed=bounded,plane_interior_lp_status=fit.message,serialized_plane_hull_hausdorff_m=plane_hausdorff,cooked_vertices_outside_serialized_planes_m=float(np.maximum(residual,0).max()),cooking_options=decoded['cooking_options'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);args=ap.parse_args();payload=json.loads((args.folder/'payload.json').read_text())['parts'];decoded=json.loads((args.folder/'decoded.json').read_text())['parts'];shapes={f"cook_{p['index']}_{p['variant']}":p for p in payload if not p.get('primitiveBox')};rows=[]
    for d in decoded:rows.append(compare_cooked(d,shapes.pop(d['name'])))
    if shapes:raise ValueError('Missing cooked parts: '+str(list(shapes)[:10]))
    summary={}
    for variant in sorted({p['variant'] for p in rows}):
        subset=[p for p in rows if p['variant']==variant and p['index']>=0]
        plane_distances=[p['serialized_plane_hull_hausdorff_m'] for p in subset if p['serialized_plane_hull_closed']]
        summary[variant]=dict(parts=len(subset),vertex_hull_discrepancies_over_01mm=sum(p['cooked_vertex_hull_hausdorff_m']>1e-4 for p in subset),maximum_vertex_hull_distance_m=max(p['cooked_vertex_hull_hausdorff_m'] for p in subset),plane_hull_discrepancies_over_01mm=sum(d>1e-4 for d in plane_distances),plane_hulls_not_closed=sum(not p['serialized_plane_hull_closed'] for p in subset),maximum_plane_hull_distance_m=max(plane_distances,default=None),parts_with_vertex_reduction=sum(p['cooked_vertices']<p['input_vertices'] for p in subset))
    result=dict(parts=rows,summary=summary,physics_executed=False,operational_geometry_changed=False,scope='Serialized cook output under the same Unity version; Player correspondence requires independent witnesses')
    (args.folder/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
