#!/usr/bin/env python3
"""Compare free-tail gravity using identical native cloth starting states."""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'.deps/RCareWorld/pyrcareworld')]
from prepare_held_stage_shapes import local_runtime_locks, write, verify_stage_player
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.circumference_fit import begin_fit, read_fit, hold_fit
from sock_dressing_simulation.held_ankle_poses import audit_actual_skin
from sock_dressing_simulation.toe_contact import select_toe_support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Native fit directory; failed trials are allowed as diagnostic seeds')
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--port', type=int, default=18700)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/cde-tail-gravity-v2')
    args = parser.parse_args()
    if not 0 < args.scale <= 1 or args.steps < 55:
        parser.error('Use a gravity scale in (0,1] and at least 55 steps')
    verify_stage_player()
    saved = json.loads((args.reference/'report.json').read_text())
    geometry = json.loads((args.reference/'fitted-geometry.json').read_text())
    pattern = saved['pattern']
    out = args.output.resolve()/f'gravity-{args.scale:g}'
    out.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT/'config/pose_patterns.yaml')
    config.pop('pose_pattern')
    os.environ['SOCK_STAGE_GRAVITY'] = '1'
    os.environ['SOCK_STAGE_PLAYER_LOG'] = str(out/'player.log')
    config['rcareworld'].update(graphics=False, port=args.port, log_level=0,
                              executable=str(ROOT/'scripts/run_stage_preparation_player.sh'))
    config['assets']['output_dir'] = str(out/'assets')
    local_runtime_locks(out/'runtime-locks')
    scenario = scenario_from_config(config, seed=0)
    prepared = _prepare(config, scenario)
    source = geometry['source']
    depth = np.asarray(source['rest_positions'])[:,2]
    depth -= depth.min()
    support = select_toe_support(np.asarray(geometry['particles_world']), source)
    boundary = support['material_max_m']
    masks = dict(cuff=depth <= .030+1e-6, wrapped=(depth > .030+1e-6)&(depth <= boundary+1e-6),
                 tail=depth > boundary+1e-6)
    samples, positions = [], []
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']), out/'assets/sock.obj', initial_joints=scenario.initial_joints)
        env.apply_scenario(scenario)
        attr = env.sock_cloth
        attr.set_pattern_foot_pose(pattern['pitch_deg'], pattern['roll_deg'])
        attr._send_data('GetSockPatternSkin')
        env._env.step(simulate=False)
        skin_error = float(np.linalg.norm(np.asarray(attr.data['sock_pattern_skin']['vertices'])-
                                         source['skin_surface']['vertices'],axis=1).max())
        if skin_error > 5e-6:
            raise ValueError('Diagnostic foot pose differs from the saved fit')
        env.initialize_preinsert(saved['stage'])
        attr.restore_prepared_state((args.reference/'initial-prepared-state.json').read_text())
        env._env.step(simulate=False)
        begin_fit(env, stage_contact=True)
        attr._send_data('SetSockHeldStageCircumferenceScale', float(saved['rest_circumference_scale']))
        env._env.step(simulate=False)
        attr.restore_prepared_state((args.reference/'fitted-prepared-state.json').read_text())
        attr.request_particles()
        env._env.step(simulate=False)
        initial = read_fit(env)
        initial_error = float(np.linalg.norm(attr.particles()-geometry['particles_world'], axis=1).max())
        if not attr.data.get('prepared_state_restore_report', {}).get('ok') or initial_error > 5e-6:
            raise ValueError('Diagnostic seed restoration failed')
        attr._send_data('ConfigureSockHeldStageFitTailGravity', float(boundary), args.scale)
        env._env.step(simulate=False)
        def record(row):
            samples.append(copy.deepcopy(row))
            positions.append(attr.particles().copy())
            if len(samples) % 10 == 0:
                print(f'tail gravity {args.scale:g}: {len(samples)*5}/{args.steps} steps', flush=True)
        held = hold_fit(env, steps=args.steps, reference_center=initial['cuff_center_world'], on_sample=record)
        policy = copy.deepcopy(attr.data.get('sock_held_stage_tail_gravity', {}))
        if (policy.get('scale') != args.scale or policy.get('affected_particles') != int(masks['tail'].sum())
                or policy.get('contact_gravity_model') != 'force_consistent_tail_gravity_v2'):
            raise ValueError('Native tail-only gravity policy was not applied')
        metrics = {}
        times = np.array([r['physics_time_s'] for r in samples[-11:]])
        final_positions = np.array(positions[-11:])
        for name, mask in masks.items():
            speeds = np.linalg.norm(np.diff(final_positions[:,mask],axis=0),axis=2)/np.diff(times)[:,None]
            metrics[name] = dict(particles=int(mask.sum()), maximum_speed_m_s=float(speeds.max()),
                rms_speed_m_s=float(np.sqrt(np.mean(speeds**2))),
                maximum_excursion_m=float(np.linalg.norm(final_positions[:,mask]-final_positions[0,mask],axis=2).max()))
        body, surface = audit_actual_skin(attr.particles(), source['skin_surface'], geometry['edges'], geometry['triangles'])
        result = dict(number=pattern['number'], stage=saved['stage'], gravity_policy=policy,
                      reference=str(args.reference.resolve()), initial_readback_error_m=initial_error,
                      skin_readback_error_m=skin_error,
                      measured_duration_s=float(samples[-1]['physics_time_s']-initial['physics_time_s']),
                      metrics=metrics, assessment=held['assessment'], body=body, surface=surface,
                      passed=bool(held['assessment']['passed'] and body['geometry_ok'] and surface['surface_ok']),
                      scope='Tail-gravity causal comparison from the same failed seed; not a full native fit regeneration')
        write(out/'report.json', result)
        write(out/'telemetry.json', samples)
        attr.request_prepared_state()
        env._env.step(simulate=False)
        (out/'fitted-prepared-state.json').write_text(attr.data['prepared_sock_state']+'\n')
        write(out/'fitted-geometry.json',dict(source=source,particles_world=attr.particles().tolist(),
            edges=geometry['edges'],triangles=geometry['triangles'],rest_edge_lengths=attr.data['particle_rest_edge_lengths']))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
