#!/usr/bin/env python3
"""Summarize independent unit-coordinate and world-coordinate PhysX probes."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
import trimesh
from audit_compound_rays import ray_interval
from sock_dressing_simulation.convex_distance import point_convex_distance


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('source',type=Path)
    args=ap.parse_args()
    raw=json.loads(args.source.read_text())
    payload=json.loads(args.source.with_suffix('.payload.json').read_text())
    summaries=defaultdict(lambda:defaultdict(float));rows=[]
    for row,source in zip(raw['parts'],payload['parts']):
        assert (row['index'],row['variant'])==(source['index'],source['variant'])
        local=np.array(row['vertices_local']);world=np.array(row['vertices_world']);center=world.mean(0)
        planes=ConvexHull(local).equations
        try:world_planes=ConvexHull(world-center).equations
        except QhullError:world_planes=None
        scales=np.array([source['scale'][a] for a in 'xyz'])
        counts=defaultdict(int);maxima=defaultdict(float);issues=[]
        for j,ray in enumerate(row['rays']):
            kind=ray['kind'];counts[kind+'_rays']+=1
            o=np.array(ray['origin_local']);d=np.array(ray['direction_local'])
            if np.linalg.norm(d)==0:
                counts[kind+'_invalid_direction']+=1
                continue
            d/=np.linalg.norm(d)
            interval=ray_interval(planes,o,d)
            counts[kind+'_local_miss']+=not ray['local_hit']
            counts[kind+'_world_miss']+=not ray['world_hit']
            counts[kind+'_input_miss']+=interval is None
            if interval is not None and ray['local_hit']:
                error=abs(ray['local_distance']-interval[0])*np.linalg.norm(d*scales)
                counts[kind+'_local_error_over_01mm']+=error>1e-4
                maxima[kind+'_local_error_m']=max(maxima[kind+'_local_error_m'],float(error))
            if world_planes is not None:
                world_direction=np.array(ray['direction_world']);world_direction/=np.linalg.norm(world_direction)
                world_interval=ray_interval(world_planes,np.array(ray['origin_world'])-center,world_direction)
                counts[kind+'_world_input_miss']+=world_interval is None
                counts[kind+'_expected_world_hit_miss']+=world_interval is not None and not ray['world_hit']
                if world_interval is not None and ray['world_hit']:
                    error=abs(ray['world_distance_m']-world_interval[0])
                    counts[kind+'_world_error_over_01mm']+=error>1e-4
                    maxima[kind+'_world_error_m']=max(maxima[kind+'_world_error_m'],float(error))
                for field,label in [('local_hit_world','unit_cook'),('world_hit_point','world')]:
                    if ray['local_hit' if label=='unit_cook' else 'world_hit']:
                        hit=np.array(ray[field])-center
                        error=abs(float(np.max(world_planes[:,:3]@hit+world_planes[:,3])))
                        maxima[label+'_boundary_plane_error_m']=max(maxima[label+'_boundary_plane_error_m'],error)
                        counts[label+'_boundary_plane_error_over_01mm']+=error>1e-4
            if 'local_closest_to_hit' in ray and ray['local_hit']:
                disagreement=np.linalg.norm((np.array(ray['local_hit_point'])-ray['local_closest_to_hit'])*scales)
                maxima['local_ray_closest_disagreement_m']=max(maxima['local_ray_closest_disagreement_m'],float(disagreement))
                counts['local_ray_closest_disagreement_over_01mm']+=disagreement>1e-4
            if 'world_closest_to_hit' in ray and ray['world_hit']:
                disagreement=np.linalg.norm(np.array(ray['world_hit_point'])-ray['world_closest_to_hit'])
                maxima['world_ray_closest_disagreement_m']=max(maxima['world_ray_closest_disagreement_m'],float(disagreement))
                counts['world_ray_closest_disagreement_over_01mm']+=disagreement>1e-4
            if not ray['local_hit'] or not ray['world_hit']:
                issues.append(dict(sample=j,kind=kind,local_hit=ray['local_hit'],world_hit=ray['world_hit'],input_interval=interval))
        if world_planes is not None:
            surface=trimesh.Trimesh(world-center,ConvexHull(world-center).simplices,process=False)
            for hit_field,flag,label in [('local_hit_world','local_hit','unit_cook'),('world_hit_point','world_hit','world')]:
                hit_points=np.array([r[hit_field] for r in row['rays'] if r[flag]])
                if len(hit_points):
                    distances=point_convex_distance(hit_points-center,world-center,surface=True)
                    maxima[label+'_hit_surface_distance_m']=float(distances.max())
                    counts[label+'_hit_surface_distance_over_01mm']=int(np.sum(distances>1e-4))
        if 'vertex_closest_local' in row:
            error=np.linalg.norm((local-np.array(row['vertex_closest_local']))*scales,axis=1)
            maxima['source_vertex_closest_error_m']=float(error.max())
            counts['source_vertex_closest_error_over_01mm']=int(np.sum(error>1e-4))
        for label in ['local','world']:
            if 'bounds_'+label+'_center' not in row:continue
            bc=np.array(row['bounds_'+label+'_center']);half=np.array(row['bounds_'+label+'_size'])/2
            hits=np.array([r[label+'_hit_point'] for r in row['rays'] if r[label+'_hit']])
            if len(hits):
                outside=np.maximum(abs(hits-bc)-half,0)
                if label=='local':outside*=scales
                distances=np.linalg.norm(outside,axis=1)
                maxima[label+'_hit_outside_collider_bounds_m']=float(distances.max())
                counts[label+'_hit_outside_collider_bounds_over_01mm']=int(np.sum(distances>1e-4))
        record=dict(index=row['index'],variant=row['variant'],counts=dict(counts),maxima=dict(maxima),misses=issues)
        rows.append(record)
        summary=summaries[row['variant']];summary['parts']+=1
        for key,value in counts.items():summary[key]+=value
        for key,value in maxima.items():summary['max_'+key]=max(summary['max_'+key],value)
    result=dict(summaries=dict(summaries),parts=rows,physics_executed=False,scope='Static diagnostic, not an installation acceptance gate')
    args.source.with_suffix('.summary.json').write_text(json.dumps(result,indent=2,default=lambda value: value.item())+'\n')
    print(json.dumps(result['summaries'],indent=2))


if __name__=='__main__':main()
