#!/usr/bin/env python3
"""Audit saved PhysX rays against the actual float input hull, including exits.

An entry distance alone is not a ray/convex intersection: it must not exceed
the exit distance. Distances along nearly tangent rays amplify plane errors.
Keep those diagnostics separate from perpendicular boundary displacement.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.convex_distance import point_convex_distance


def ray_interval(planes, origin, direction):
    """Return the forward ray interval, or None for a miss (unit planes)."""
    if not np.all(np.isfinite(origin)) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction)==0:
        raise ValueError('Ray needs a finite origin and nonzero finite direction')
    offsets = planes[:, :3] @ origin + planes[:, 3]
    slopes = planes[:, :3] @ direction
    parallel = abs(slopes) < 1e-12
    if np.any(offsets[parallel] > 1e-12):
        return None
    incoming, outgoing = slopes < -1e-12, slopes > 1e-12
    enter = max(0., float(np.max(-offsets[incoming] / slopes[incoming], initial=-np.inf)))
    leave = float(np.min(-offsets[outgoing] / slopes[outgoing], initial=np.inf))
    return (enter, leave) if enter <= leave else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT/'artifacts/compound-reaching/native-compound.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/compound-reaching/followup/ray-audit.json')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = []
    for i, row in enumerate(data['parts']):
        points = np.array(row['vertices_world'])
        center = points.mean(0)
        hull = ConvexHull(points-center)
        planes = hull.equations
        issues = []
        maximum_boundary_plane_error = 0.
        counts = dict(rays=0, physx_misses=0, input_ray_misses=0,
                      both_miss=0, physx_only_misses=0, hit_plane_error_over_01mm=0,
                      interval_error_over_01mm=0, sphere_bad=0, vertex_bad=0)
        hits=[]
        for j, probe in enumerate(row['physx_radial_probes']):
            origin = np.array(probe['origin_world'])-center
            direction = np.array(probe['direction_world'])
            direction /= np.linalg.norm(direction)
            interval = ray_interval(planes, origin, direction)
            counts['rays'] += 1
            counts['input_ray_misses'] += interval is None
            counts['physx_misses'] += not probe['hit']
            counts['both_miss'] += interval is None and not probe['hit']
            counts['physx_only_misses'] += interval is not None and not probe['hit']
            error = None if interval is None or not probe['hit'] else abs(probe['distance_m']-interval[0])
            plane_error = None
            if probe['hit']:
                hit = origin+direction*probe['distance_m']
                hits.append(hit)
                # Outside: a lower bound on distance. Inside: exact distance
                # to the nearest supporting plane. Do not call this Hausdorff.
                plane_error = abs(float(np.max(planes[:, :3]@hit+planes[:, 3])))
                maximum_boundary_plane_error = max(maximum_boundary_plane_error, plane_error)
                counts['hit_plane_error_over_01mm'] += plane_error > 1e-4
            bad = interval is None or not probe['hit'] or error > 1e-4
            counts['interval_error_over_01mm'] += error is not None and error > 1e-4
            counts['sphere_bad' if j < 96 else 'vertex_bad'] += bad
            if bad:
                issues.append(dict(sample=j, kind='sphere' if j < 96 else 'vertex',
                                   input_interval_m=interval, physx_hit=probe['hit'],
                                   distance_error_m=error, boundary_plane_error_m=plane_error))
        singular = np.linalg.svd(points-center, compute_uv=False)
        # A plane residual is only a lower bound outside the hull. Query the
        # actual closed triangle surface as a separate Euclidean diagnostic.
        exact=point_convex_distance(np.asarray(hits),points-center,surface=True) if hits else np.array([])
        rows.append(dict(index=i, link=row['link'], counts=counts,
                         thickness_ratio=float(singular[-1]/singular[0]),
                         maximum_boundary_plane_error_m=maximum_boundary_plane_error,
                         maximum_hit_surface_distance_m=float(exact.max(initial=0)),
                         hits_with_surface_distance_over_01mm=int(np.sum(exact>1e-4)),issues=issues))
    totals = {key: sum(r['counts'][key] for r in rows) for key in rows[0]['counts']}
    result = dict(input=str(args.input), totals=totals,
                  parts_with_boundary_plane_error_over_01mm=sum(r['maximum_boundary_plane_error_m']>1e-4 for r in rows),
                  maximum_hit_surface_distance_m=max(r['maximum_hit_surface_distance_m'] for r in rows),
                  hits_with_surface_distance_over_01mm=sum(r['hits_with_surface_distance_over_01mm'] for r in rows),
                  parts=rows, passed=False,
                  scope='Diagnostic of float input hull; not a replacement geometry acceptance gate')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='parts'}, indent=2))


if __name__ == '__main__':
    main()
