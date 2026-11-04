#!/usr/bin/env python3
"""Compare residual rays and sphere contacts with affine, unrounded hulls."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import trimesh
from audit_compound_rays import ray_interval
from sock_dressing_simulation.convex_distance import point_convex_distance


def affine_ray_interval(planes,origin,direction,linear,translation):
    # Keep the local direction's length: the interval remains in world metres.
    return ray_interval(planes,np.linalg.solve(linear,np.asarray(origin)-translation),np.linalg.solve(linear,direction))


def transformed_planes(planes,linear):
    normals=planes[:,:3]@np.linalg.inv(linear);lengths=np.linalg.norm(normals,axis=1)
    return np.c_[normals/lengths[:,None],planes[:,3]/lengths]


def sphere_penetration(points,planes,centers,radii):
    """Signed overlap depth of a sphere and a closed convex solid."""
    offsets=np.max(centers@planes[:,:3].T+planes[:,3],axis=1)
    signed=offsets.copy();outside=offsets>0
    if np.any(outside):signed[outside]=point_convex_distance(centers[outside],points)
    return np.asarray(radii)-signed


def main():
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('--compact',action='store_true');args=ap.parse_args();raw=json.loads(args.source.read_text());rows=[];totals=Counter();ray_kinds={}
    for control in raw['controls']:
        expected=max(0.,.7-control['x'])
        assert control['overlap']==(expected>0) and abs(control['depth']-expected)<1e-6,control
    for row in raw['parts']:
        local=np.array(row['vertices_local']);planes=ConvexHull(local).equations;M=np.array(row['world_matrix']);translation=np.array(row['world_position']);R=Rotation.from_quat(row['world_rotation']).as_matrix();linear=R@np.diag(row['world_scale']);world=np.array(row['vertices_world']);center=world.mean(0);world_planes=ConvexHull(world-center).equations;records=[];counts=Counter()
        for ray in row['rays']:
            ray_counts=Counter();kind=ray.get('kind','world_sphere' if ray['sample']<96 else 'world_vertex');ray_counts['rays']+=1
            origin=np.array(ray['origin_world']);direction=np.array(ray['direction_world']);direction/=np.linalg.norm(direction)
            intervals=dict(float_vertices=ray_interval(world_planes,origin-center,direction),matrix=affine_ray_interval(planes,origin,direction,M[:3,:3],M[:3,3]),pose=affine_ray_interval(planes,origin,direction,linear,translation))
            errors={}
            for kind,interval in intervals.items():
                counts[kind+'_input_misses']+=interval is None
                counts[kind+'_hit_mismatch']+=(interval is not None)!=ray['hit']
                error=None if not ray['hit'] or interval is None else abs(ray['distance_m']-interval[0]);errors[kind]=error
                counts[kind+'_distance_over_01mm']+=error is not None and error>1e-4
            shifted=affine_ray_interval(planes,ray['origin_shifted'],np.array(ray['direction_shifted'])/np.linalg.norm(ray['direction_shifted']),linear,np.zeros(3));shifted_error=None if not ray['shifted_hit'] or shifted is None else abs(ray['shifted_distance_m']-shifted[0]);counts['shifted_pose_hit_mismatch']+=(shifted is not None)!=ray['shifted_hit'];counts['shifted_pose_distance_over_01mm']+=shifted_error is not None and shifted_error>1e-4
            ray_counts['world_pose_hit_mismatch']+=(intervals['pose'] is not None)!=ray['hit'];ray_counts['world_pose_distance_over_01mm']+=errors['pose'] is not None and errors['pose']>1e-4;ray_counts['shifted_pose_hit_mismatch']+=(shifted is not None)!=ray['shifted_hit'];ray_counts['shifted_pose_distance_over_01mm']+=shifted_error is not None and shifted_error>1e-4
            ray_kinds.setdefault(row['variant'],{}).setdefault(ray.get('kind','world_sphere' if ray['sample']<96 else 'world_vertex'),Counter()).update(ray_counts)
            bad=any(v for k,v in ray_counts.items() if k!='rays')
            if not args.compact or bad:records.append(dict(sample=ray['sample'],kind=ray.get('kind','residual'),hit=ray['hit'],intervals=intervals,errors_m=errors,shifted_hit=ray['shifted_hit'],shifted_interval=shifted,shifted_error_m=shifted_error))
        queries=row['contacts'];centers=np.array([r['center_world'] for r in queries])-translation;radii=np.array([r['radius_m'] for r in queries]);expected=sphere_penetration(local@linear.T,transformed_planes(planes,linear),centers,radii)
        shifted_expected=sphere_penetration(local@linear.T,transformed_planes(planes,linear),np.array([r['center_shifted'] for r in queries]),radii)
        contacts=[]
        for q,e,se in zip(queries,expected,shifted_expected):
            e=float(e);se=float(se);error=abs((q['depth_m'] if q['overlap'] else 0)-max(0.,e));shifted_error=abs((q['shifted_depth_m'] if q['shifted_overlap'] else 0)-max(0.,se))
            counts['contact_tests']+=1
            if q.get('compute_penetration_executed',True):
                counts['compute_penetration_tests']+=1;counts['contact_error_over_01mm']+=error>1e-4;counts['shifted_contact_error_over_01mm']+=shifted_error>1e-4
            # A boolean difference within 1 micron of touching is recorded but
            # is not called a macroscopic shape discrepancy.
                counts['contact_boolean_mismatch_over_1um']+=abs(e)>1e-6 and q['overlap']!=(e>0)
            scene_bad=False
            if 'scene_overlap' in q:
                counts['scene_overlap_tests']+=1;scene_bad=bool(abs(e)>1e-6 and q['scene_overlap']!=(e>0));shifted_scene_bad=bool(abs(se)>1e-6 and q['shifted_scene_overlap']!=(se>0));counts['scene_overlap_mismatch_over_1um']+=scene_bad;counts['shifted_scene_overlap_mismatch_over_1um']+=shifted_scene_bad
                counts[q['kind']+'_scene_tests']+=1;counts[q['kind']+'_scene_mismatch']+=scene_bad
                scene_bad=scene_bad or shifted_scene_bad
            if not args.compact or scene_bad or q.get('compute_penetration_executed',True) and (error>1e-4 or shifted_error>1e-4):
                contacts.append(dict(sample=q['sample'],kind=q['kind'],radius_m=q['radius_m'],expected_depth_m=e,observed_depth_m=q['depth_m'] if q['overlap'] else 0,overlap=q['overlap'],depth_error_m=error,shifted_expected_depth_m=se,shifted_observed_depth_m=q['shifted_depth_m'] if q['shifted_overlap'] else 0,shifted_depth_error_m=shifted_error,scene_overlap=q.get('scene_overlap'),shifted_scene_overlap=q.get('shifted_scene_overlap')))
        reconstructed=local@linear.T+translation
        rows.append(dict(index=row['index'],variant=row['variant'],counts=dict(counts),maximum_float_vertex_rounding_m=float(np.linalg.norm(world-reconstructed,axis=1).max()),maximum_matrix_pose_vertex_difference_m=float(np.linalg.norm((local@M[:3,:3].T+M[:3,3])-reconstructed,axis=1).max()),rays=records,contacts=contacts));totals.update(counts)
    by_variant={}
    for row in rows:
        by_variant.setdefault(row['variant'],Counter()).update(row['counts'])
    result=dict(source=str(args.source),box_sphere_controls_passed=True,totals=dict(totals),by_variant={key:dict(value) for key,value in by_variant.items()},by_variant_ray_kind=ray_kinds,parts=rows,geometry_gate_passed=False,physics_executed=False,scope='Residual diagnostic; affine input hull is not an extracted PhysX cooked hull')
    if rows and all(row['index']<0 for row in rows):
        result['compute_penetration_calibration_passed']=not totals['contact_error_over_01mm'] and not totals['contact_boolean_mismatch_over_1um']
        result['scene_overlap_calibration_passed']=None if not totals['scene_overlap_tests'] else not totals['scene_overlap_mismatch_over_1um'] and not totals['shifted_scene_overlap_mismatch_over_1um']
    args.source.with_suffix('.summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['totals'],indent=2))


if __name__=='__main__':main()
