#!/usr/bin/env python3
"""Supplement containment with pose-validated rendered skin vertices.

The selected skin includes the ankle transition. Its aggregate ratio is not the
toe/forefoot/heel acceptance criterion and does not require ankle pull-up.
This is offline analysis: it never connects to or changes the simulator.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from foot_containment_demo import closed_faces, inside_points


def validate_pose(row, geometry):
    live = row['dressing_qa']['foot_collision_safety']['foot_collider_geometry']
    for reference in geometry['regions']:
        collider = next(c for c in live if c['region'] == reference['region'])
        for field in ('position', 'rotation', 'scale'):
            if not np.allclose(collider[field], reference['reference_'+field],
                               rtol=0, atol=1e-6):
                raise ValueError('Recorded human pose differs from skin reconstruction: '+field)


def audit(directory, skin_path, geometry_path):
    directory, skin_path, geometry_path = map(Path, (directory, skin_path, geometry_path))
    skin_bytes = skin_path.read_bytes()
    skin, geometry = json.loads(skin_bytes), json.loads(geometry_path.read_text())
    if hashlib.sha256(skin_bytes).hexdigest() != geometry['source_skin_sha256']:
        raise ValueError('Skin reconstruction does not match the collision geometry source')
    if not skin['recorded_bone_pose_validated'] or skin['maximum_baked_skin_error_m'] > 1e-5:
        raise ValueError('Skin reconstruction is not validated against rendered geometry')
    points = np.array([[v['world'][a] for a in 'xyz'] for v in skin['vertices']])
    records = []
    for line in (directory/'live_frame_diagnostics.jsonl').read_text().splitlines():
        row = json.loads(line)
        validate_pose(row, geometry)
        try:
            vertices, faces, cap = closed_faces(row)
            inside, ambiguous = inside_points(points, vertices[faces])
            anatomy = {s['name']:np.mean(s['samples_world'], axis=0)
                       for s in row['dressing_qa']['anatomical_foot_conformity']['sections']}
            leg_axis = anatomy['calf']-anatomy['ankle']
            leg_axis /= np.linalg.norm(leg_axis)
            cuff_heights = (vertices[np.unique(cap)]-anatomy['ankle']) @ leg_axis
            records.append({'command':row['frame']+1, 'valid':not bool(ambiguous.any()),
                'inside_vertices':int(inside.sum()), 'total_vertices':len(points),
                'aggregate_ratio':float(inside.mean()),
                'ambiguous_vertex_indices':np.flatnonzero(ambiguous).tolist(),
                'outside_vertex_indices':np.flatnonzero(~inside & ~ambiguous).tolist(),
                'optional_cuff_mean_height_above_ankle_m':float(cuff_heights.mean()),
                'optional_cuff_min_height_above_ankle_m':float(cuff_heights.min()),
                'optional_cuff_max_height_above_ankle_m':float(cuff_heights.max())})
        except ValueError as error:
            records.append({'command':row['frame']+1, 'valid':False, 'reason':str(error)})
    return {'role':'supplementary rendered-skin audit; not the regional acceptance criterion',
        'limitation':'Selected vertices include ankle transition; exposed ankle is permitted. Vertex counts are not surface area fractions. World vertices assume the unchanged human skin pose; each frame checks the reference foot collider transform.',
        'physics_or_rendering_modified':False,
        'optional_pull_up_reference':'Cuff displacement from ankle section centre along the ankle-to-calf section axis; not a success requirement.',
        'source_skin_sha256':geometry['source_skin_sha256'],
        'maximum_baked_skin_error_m':skin['maximum_baked_skin_error_m'],
        'reference_foot_collider_pose_matches_for_every_recorded_command':True,
        'vertices':skin['vertices'], 'rows':records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--skin', type=Path, default=Path('artifacts/phase4/foot-conforming-fixed-motion/human-skin-reconstruction.json'))
    parser.add_argument('--geometry', type=Path, default=Path('RCareUnity/Assets/SockDressing/Resources/FootSkinHullGeometry.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.skin, args.geometry)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'commands':len(result['rows']), 'last':result['rows'][-1]}))
