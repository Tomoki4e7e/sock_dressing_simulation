#!/usr/bin/env python3
"""Seed right-arm endpoints on the reachable side of the shoulder barrier.

Goals remain candidates. search_common_opening_starts.py routes independently
validates the entire interpolated arm motion before accepting any trajectory.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from search_centered_reaching import REGISTERED_MANIFEST, URDF, ROBOT, write
from search_fixed_torso_starts import context
from sock_dressing_simulation.common_opening_approach import (
    inventory, load_state, guided_approach_targets, edge_samples, sha256,
)
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics, ControlledCollisionScene
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', help='Default: right-arm conditions not yet accepted')
    parser.add_argument('--max-targets', type=int, default=256)
    args = parser.parse_args(); out = args.output.resolve()
    initial = yaml.safe_load((out/'initial.yaml').read_text())
    certificate_path = ROOT/'artifacts/apron-reaching/shoulder-barrier.json'
    certificate = json.loads(certificate_path.read_text())
    if (not certificate['barrier_confirmed'] or certificate['urdf_sha256'] != sha256(URDF)
            or ROOT/certificate['robot_scene'] != ROBOT
            or certificate['required_clearance_m'] != .005):
        raise ValueError('The shoulder component certificate does not match the fixed planning model')
    rows = inventory(ROOT)
    for row in rows:
        if args.ids and row['id'] not in args.ids:
            continue
        prior = out/'routes/right'/f'{row["id"]}.json'
        if not args.ids and prior.exists() and json.loads(prior.read_text())['passed']:
            continue
        state = load_state(ROOT, row)
        k, base, *_ = context(row['scene_number'], state, REGISTERED_MANIFEST,
            left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
        base.use_spatial_aabb = True
        base.park_left(initial['left_q_rad'], initial['left_gap_m'])
        model = ControlledArmKinematics(k, 'right', initial['right_q_rad'], initial['right_gap_m'])
        model.lower = model.lower+np.deg2rad(5); model.upper = model.upper-np.deg2rad(5)
        boundary = certificate['joint1_slice_deg']
        if np.rad2deg(initial['right_q_rad'][0]) > boundary:
            model.lower[0] = max(model.lower[0], np.deg2rad(boundary+.01))
        else:
            model.upper[0] = min(model.upper[0], np.deg2rad(boundary-.01))
        checker = ControlledCollisionScene(base, 'right', initial['right_q_rad'], initial['right_gap_m'])
        delta = np.asarray(initial['translation_m'])
        skin = SkinBody(state['source']['skin_surface'])
        cloth = ClothScreen(state['particles_world'], state['triangles'])
        candidates = []
        for index, target in enumerate(guided_approach_targets(
                state, 'right', model, np.asarray(initial['right_q_rad']), delta)):
            if index >= args.max_targets:
                break
            R = np.asarray(target['rotation']); p = np.asarray(target['tcp_target_m'])-delta
            q = np.asarray(target['seed_q_rad']); gap = .04
            if not skin.inspect(k.hand.bank(gap), R, p, .001)[0] or cloth.intersections(k.hand.bank(gap), R, p):
                continue
            if not checker.is_clear(q, gap, delta):
                continue
            straight = all(checker.is_clear(value, gap, delta)
                for value in edge_samples(initial['right_q_rad'], q, 1.))
            candidates.append(dict(index=index, target=target, q=q.tolist(), straight_coarse_passed=straight))
            if straight:
                break
        if not candidates:
            print('NO COMPONENT GOAL', row['id'], flush=True)
            continue
        best = next((c for c in candidates if c['straight_coarse_passed']), candidates[0])
        value = dict(condition=row, side='right', passed=False, goal_only=True,
            initial_sha256=sha256(out/'initial.yaml'), route=dict(target=best['target'], q_rad=[best['q']]),
            search=dict(generator=str(Path(__file__).relative_to(ROOT)),
                generator_sha256=sha256(__file__), certificate=str(certificate_path.relative_to(ROOT)),
                certificate_sha256=sha256(certificate_path), source_robot_scene_sha256=sha256(ROBOT),
                geometry_manifest_sha256=sha256(REGISTERED_MANIFEST), candidates=candidates,
                reason='Same side of the certified shoulder barrier; final interpolated route remains unvalidated'))
        write(out/'seed-routes/right'/f'{row["id"]}.json', value)
        print('GOAL SEEDED', row['id'], best['index'], best['straight_coarse_passed'], flush=True)


if __name__ == '__main__':
    main()
