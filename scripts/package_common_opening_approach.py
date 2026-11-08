#!/usr/bin/env python3
"""Publish only a complete, same-start, 70-state/two-arm IK collection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
        default=ROOT/'artifacts/common-opening-approach/negative-wrist-start')
    args = parser.parse_args(); out = args.root.resolve()
    report = json.loads((out/'summary.json').read_text())
    initial = yaml.safe_load((out/'initial.yaml').read_text())
    assert report['all_passed'] and report['counts'] == dict(right=70, left=70)
    assert len(report['rows']) == 140 and report['initial_sha256'] == digest(out/'initial.yaml')
    native = json.loads((ROOT/report['native_initial']['file']).read_text())
    assert native['passed'] and digest(ROOT/report['native_initial']['file']) == report['native_initial']['sha256']
    for field in ('right_q_rad', 'left_q_rad', 'translation_m'):
        np.testing.assert_array_equal(native['initial_configuration'][field], initial[field])
    conditions = set(); seen = set()
    for row in report['rows']:
        assert row['passed'] and row['validation']['passed'] and row['fixed_frame_audit']['passed']
        key = row['id'], row['side']; assert key not in seen; seen.add(key)
        conditions.add((row['stage'], row['pitch_deg'], row['roll_deg']))
        path = ROOT/row['trajectory']; assert digest(path) == row['trajectory_sha256']
        route = json.loads(path.read_text()); side = route['side']
        assert route['initial_sha256'] == report['initial_sha256']
        assert route['condition']['state_sha256'] == digest(ROOT/route['condition']['state'])
        q = np.asarray(route['route']['q_rad']); raw = np.asarray(route['route']['raw_q_rad'])
        np.testing.assert_array_equal(q[0], initial[side+'_q_rad'])
        assert row['validation']['endpoint_position_error_m'] <= 1e-4
        assert row['validation']['physical_tip_midplane_error_m'] <= .0002
        assert row['validation']['minimum_joint_margin_deg'] >= 5-1e-6
        assert row['validation']['maximum_velocity_fraction'] <= .1+1e-9
        assert row['validation']['maximum_acceleration_fraction'] <= .1+1e-9
        if side == 'right':
            # The fixed shoulder model has a certified barrier at joint1=110°.
            # PCHIP and the linear fallback stay within the raw scalar ranges.
            assert np.rad2deg(raw[:, 0]).min() > 110.
        csv_path = ROOT/row['csv']['file']; assert digest(csv_path) == row['csv']['sha256']
        with csv_path.open(newline='') as stream:
            records = list(csv.reader(stream))
        values = np.asarray(records[1:], dtype=float)
        assert values.shape[1] == 20 and len(values) == row['csv']['samples']
        assert np.isfinite(values).all() and np.all(np.diff(values[:, 0]) > 0)
        np.testing.assert_array_equal(values[0, 1:8], initial['right_q_rad'])
        np.testing.assert_array_equal(values[0, 8:15], initial['left_q_rad'])
        fixed = values[:, 8:15] if side == 'right' else values[:, 1:8]
        np.testing.assert_array_equal(fixed, np.repeat([initial[('left' if side == 'right' else 'right')+'_q_rad']], len(values), axis=0))
    assert len(conditions) == 70
    verified = yaml.safe_load((out/'verified-initial.yaml').read_text())
    for field in ('right_q_rad', 'left_q_rad', 'translation_m'):
        np.testing.assert_array_equal(verified[field], initial[field])
    destination = ROOT/'config/common_opening_initial_pose.yaml'
    destination.write_text(yaml.safe_dump(verified, allow_unicode=True, sort_keys=False))
    note = '''70 saved opening-stage / ankle-posture states, with one right and one left opening-approach IK trajectory per state.
The identical initial arm poses and human translation apply to every route.
Left starts at the accepted A route's penultimate waypoint. Right starts about 28 mm from its reflection across the reference foot sagittal plane.
The opposite arm stays at the shared initial pose. Torso, head, base and human placement stay fixed.
Target: physical finger-tip midplane 2 mm outside a material point in the opening 0--15 mm band. The calibrated TCP is 10 mm behind the finger tips.
CSV columns: time_s, 7 right-arm joint angles [rad], 7 left-arm joint angles [rad], right gap [m], left gap [m], moving TCP X/Y/Z [m]. Times are nonuniform; a jaw-width adjustment at the initial pose is included when needed.
JSON files include the raw arm path, timed samples, opening target and independent interpolated collision validation.
The collision tests use the registered full robot geometry, posed human/leg skin and frozen cloth. They are sampled tests, not a continuous collision certificate.
The common initial pose was restored and checked in Unity. Complete dynamic trajectory replay, grasp closure, frictional holding and pull-up are not validated. This collection is not marked learning-ready.
Existing reference placement and the grounded-left-foot floor allowance are retained.
Repository-relative shape paths and SHA256 hashes in the JSON refer to the already saved 70 shapes; shape assets are not duplicated in this archive.
'''
    (out/'collection-notes.txt').write_text(note)
    archive = out/'ik-trajectories-70-states.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in ('verified-initial.yaml', 'reference.json', 'summary.json', 'summary.csv', 'states.json', 'collection-notes.txt'):
            z.write(out/name, name)
        for row in report['rows']:
            z.write(ROOT/row['trajectory'], f'json/{row["side"]}/{row["id"]}.json')
            z.write(ROOT/row['csv']['file'], f'csv/{row["side"]}/{row["id"]}.csv')
    index = dict(all_passed=True, conditions=70, routes=140, initial_configuration=str(destination.relative_to(ROOT)),
        initial_configuration_sha256=digest(destination), source_initial_sha256=report['initial_sha256'],
        archive=str(archive.relative_to(ROOT)), archive_sha256=digest(archive),
        csv_units='seconds, radians, metres', moving_arms='one at a time', native_initial_passed=True,
        full_dynamic_replay_validated=False, learning_ready=False)
    (out/'collection.json').write_text(json.dumps(index, indent=2)+'\n')
    (out/'route-search-progress.json').write_text(json.dumps(dict(passed=140, completed=140, requested=140,
        complete=True, source='Final validated collection', rows=[dict(id=r['id'], side=r['side'], passed=True,
        file=r['trajectory']) for r in report['rows']]), indent=2)+'\n')
    print('VERIFIED 70 states / 140 routes; published', destination, flush=True)
    print(archive, 'bytes', archive.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
