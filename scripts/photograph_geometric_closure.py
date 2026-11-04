#!/usr/bin/env python3
"""Replay five accepted geometric candidates as frozen native Unity snapshots."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from photograph_grasp_candidates import photograph
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.grasp_geometry import RightHand, HANDEDNESS
from sock_dressing_simulation.pose_patterns import initialize_pattern, read_patterns
from sock_dressing_simulation.scenario import scenario_from_config

# Maximum area, two different cuff sites/ankle angles, deeper band, threshold edge.
CHOICES = [(22, 6), (12, 9), (3, 8), (30, 26), (5, 11)]


def capture(number, index, port, out, player):
    directory = out / f'pose-{number:02d}-candidate-{index:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / 'report.json'
    if destination.exists():
        return json.loads(destination.read_text())
    source = ROOT / f'artifacts/geometric-closure/pose-{number:02d}/report.json'
    report = json.loads(source.read_text())
    row = next(r for r in report['candidates'] if r['candidate_index'] == index)
    if not row['accepted']:
        raise ValueError('Only accepted candidates may be selected')
    initial = json.loads((ROOT / row['initial_state_source']).read_text())
    prior = json.loads((ROOT / f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text())
    hand = RightHand(ROOT / prior['settings']['urdf'])
    config = load_config(ROOT / 'config/pose_patterns_toe_contact.yaml')
    settings = config.pop('pose_pattern')
    workbook = settings.pop('workbook')
    settings.pop('number')
    # Restore archived settled vertices rather than running a new settling trial.
    settings['toe_contact'] = None
    config['rcareworld'].update(executable=player, graphics=True,
                               port=port, log_level=0)
    config['assets']['output_dir'] = str(directory / 'assets')
    pattern = next(p for p in read_patterns(Path(workbook)) if p.number == number)
    scenario = scenario_from_config(config, seed=0)
    prepared = _prepare(config, scenario)
    c = row['candidate']
    R = np.asarray(c['rotation_matrix'])
    p = np.asarray(c['grasp_center_unity_world_m'])
    expected_cloth = np.asarray(initial['initial_particles_world'])
    result = dict(number=number, candidate_index=index, pitch_deg=report['pitch_deg'],
                  roll_deg=report['roll_deg'], geometric_report=str(source.relative_to(ROOT)),
                  candidate=row, stages=[], frozen_cloth=True,
                  physical_closure_or_robot_ik_executed=False, player=player)
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']), directory / 'assets/sock.obj',
                 initial_joints=scenario.initial_joints)
        env.apply_scenario(scenario)
        def advance(count):
            if count:
                env._env.Simulate(.02, int(count))
                env._env.step(simulate=False)
        env.advance_physics = advance
        initialize_pattern(env, pattern, **settings)
        attr = env.sock_cloth
        # The geometric evaluation used each trial's saved initial vertices.
        # Restore those exact vertices; all subsequent calls disable physics.
        # End the contact experiment first: initialization is prohibited while
        # its fitted material is active. Reinstall the same skin for preview.
        attr._send_data('EndSockToeContact')
        attr.set_pattern_held(False)
        env._env.step(2, simulate=False)
        normal = np.asarray(prior['samples'][c['sample_index']]['normal'])
        normal /= np.linalg.norm(normal)
        attr.initialize_preinsert(expected_cloth, normal)
        attr.request_particles()
        env._env.step(2, simulate=False)
        error = float(np.linalg.norm(attr.particles() - expected_cloth, axis=1).max())
        if error > 5e-6:
            raise RuntimeError(f'Cloth state mismatch: {error}')
        result['cloth_readback_error_m'] = error
        attr._send_data('InstallSockRigidSkinContact', env.robot.id, 2)
        env._env.step(2, simulate=False)
        attr._send_data('GetSockRigidSkinContact')
        env._env.step(2, simulate=False)
        result['native_skin_report'] = copy.deepcopy(attr.data['sock_rigid_skin_contact'])
        state = json.loads((ROOT / json.loads((ROOT / f'artifacts/grasp-cloth-refilter/pose-{number:02d}/report.json').read_text())['source_state']).read_text())
        expected_skin = np.asarray(state['source']['skin_surface']['vertices'])
        native_skin = np.asarray(result['native_skin_report']['vertices_world'])
        result['skin_readback_error_m'] = float(cKDTree(expected_skin).query(native_skin)[0].max())
        if result['skin_readback_error_m'] > 5e-6:
            raise RuntimeError('Native skin does not match the evaluated initial state')
        attr._send_data('SetSockInspectionMaterial', True, env.robot.id)
        env._env.step(2, simulate=False)
        stages = [('approach-start', 'approach_start_placements', row['opening_gap_m'],
                   p - R[:, 0] * report['settings']['approach_distance_m']),
                  ('open', 'open_placements', row['opening_gap_m'], p),
                  ('closed', 'closed_placements', row['closing_gap_m'], p)]
        for stage, key, gap, center in stages:
            attr._send_data('SetSockGraspPreview', json.dumps(dict(label=stage, parts=row[key])))
            env._env.step(2, simulate=False)
            readback = copy.deepcopy(attr.data['sock_grasp_preview'])
            L = R @ HANDEDNESS
            base = center - L @ hand.grasp_center_urdf
            expected = {}
            for link, raw in hand.meshes.values():
                T = hand.link_transform(link, hand.joint_for_gap(gap))
                expected[link] = (raw @ T[:3, :3].T + T[:3, 3]) @ L.T + base
            mesh_error = max(float(cKDTree(expected[r['link']]).query(
                np.asarray(r['vertices_world']))[0].max()) for r in readback['parts'])
            if mesh_error > 1e-5:
                raise RuntimeError(f'Native hand geometry mismatch: {mesh_error}')
            images = []
            views = [('overview', R[:, 2] + R[:, 0] * .3, p - R[:, 0] * .04, .30, 42),
                     ('gap-side', R[:, 2] + R[:, 0] * .10, p, .085, 32),
                     ('gap-opposite', -R[:, 2] + R[:, 0] * .10, p, .085, 32),
                     ('tip', R[:, 0] + R[:, 2] * .15, p, .095, 32)]
            for view, direction, target, distance, fov in views:
                images.append(photograph(env, directory / f'{stage}-{view}.png', target,
                                         direction, R[:, 1], distance, fov))
            result['stages'].append(dict(stage=stage, gap_m=gap, images=images,
                                         native_hand_readback_error_m=mesh_error))
            print(f'pose {number} candidate {index} {stage}: captured', flush=True)
        attr.request_particles()
        env._env.step(simulate=False)
        result['cloth_drift_during_photography_m'] = float(np.linalg.norm(
            attr.particles() - expected_cloth, axis=1).max())
        if result['cloth_drift_during_photography_m'] > 5e-6:
            raise RuntimeError('Cloth changed while photographing')
        attr._send_data('RemoveSockGraspPreview')
        attr._send_data('SetSockInspectionMaterial', False, env.robot.id)
        env._env.step(simulate=False)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    shutil.rmtree(directory / 'assets')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=15770)
    parser.add_argument('--number', type=int)
    parser.add_argument('--player', default='scripts/run_geometric_inspection_player.sh')
    parser.add_argument('--output', default='artifacts/geometric-closure-photos-clean')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset, (number, index) in enumerate(CHOICES):
        if args.number is None or number == args.number:
            rows.append(capture(number, index, args.port + offset, out, args.player))
    (out / 'report.json').write_text(json.dumps(dict(candidates=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
