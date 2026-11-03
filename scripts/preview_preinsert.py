#!/usr/bin/env python3
"""Render A-E partial dressing in the native Player and save restart states."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config, resolve_package_path
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.preinsert import audit_preinsert
from sock_dressing_simulation.sock_cloth import validate_prepared_state_readback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/preinsert_normal.yaml')
    parser.add_argument('--stages', nargs='+', choices=list('ABCDE'), default=list('ABCDE'))
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/sock-pose-pattern/preinsert-normal')
    parser.add_argument('--above-heel-m', type=float, default=.05)
    args = parser.parse_args()
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Keep generated assets local; the usual asset directory may be shared.
    config['assets']['output_dir'] = str(output/'assets')
    config['rcareworld']['graphics'] = True
    config['camera']['fov'] = 50.
    config.setdefault('preinsert', {}).update(stage=args.stages[0], above_heel_m=args.above_heel_m)
    scenario = scenario_from_config(config, seed=0)
    prepared = _prepare(config, scenario)
    records, thumbnails = [], []
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']), resolve_package_path(config['assets']['output_dir'])/'sock.obj',
                 initial_joints=scenario.initial_joints)
        application = env.apply_scenario(scenario)
        for index, stage in enumerate(args.stages):
            try:
                report = (application['preinsert'] if index == 0 else
                          env.initialize_preinsert(stage, above_heel_m=args.above_heel_m,
                              clearance_m=config['preinsert'].get('clearance_m', .015)))
            except Exception:
                (output/'failed-source.json').write_text(json.dumps(env.sock_cloth.data.get('sock_preinsert_source'), indent=2))
                raise
            source = copy.deepcopy(env.sock_cloth.data['sock_preinsert_source'])
            env.sock_cloth.request_physics_clock()
            env._env.step(simulate=False)
            initial_clock = copy.deepcopy(env.sock_cloth.data['cloth_physics_clock'])
            env.observe()
            env.sock_cloth.request_physics_clock()
            env._env.step(simulate=False)
            report['first_observation_preserves_physics_clock'] = initial_clock == env.sock_cloth.data['cloth_physics_clock']
            if not report['first_observation_preserves_physics_clock']:
                raise RuntimeError('Initial observation advanced preinsert physics')
            env.sock_cloth.request_particles()
            env.sock_cloth.request_prepared_state()
            env._env.step(simulate=False)
            particles = env.sock_cloth.particles()
            report['geometry_audit'] = audit_preinsert(particles, source,
                env.sock_cloth.data['particle_edges'], env.sock_cloth.data['particle_rest_edge_lengths'])
            center = (particles.min(axis=0)+particles.max(axis=0))*.5
            # Two consistent anatomical views, independent of robot-mounted framing.
            toe, heel, calf, ankle = [np.asarray(source[k]) for k in ('toe', 'heel', 'calf', 'ankle')]
            axis = (toe-heel)/np.linalg.norm(toe-heel)
            side = np.cross(axis, calf-ankle); side /= np.linalg.norm(side)
            up = np.cross(side, axis); up /= np.linalg.norm(up)
            images = []
            env.sock_cloth._send_data("SetSockInspectionMaterial", True, env.robot.id)
            env._env.step(simulate=False)
            for name, direction in [('side', side), ('top', up)]:
                position = center + direction*.65
                forward = center-position
                yaw = np.degrees(np.arctan2(forward[0], forward[2]))
                pitch = -np.degrees(np.arctan2(forward[1], np.linalg.norm(forward[[0, 2]])))
                env.recording_camera.SetTransform(position=position.tolist(), rotation=[float(pitch), float(yaw), 0.])
                env._env.step(simulate=False)
                camera = env._capture_recording_camera(simulate=False)
                image = Image.fromarray(camera['rgb'])
                image.save(output/f'{stage}-{name}.png')
                image.thumbnail((480, 360))
                images.append(image)
            env.sock_cloth._send_data("SetSockInspectionMaterial", False, env.robot.id)
            env._env.step(simulate=False)
            tile = Image.new('RGB', (960, 390), 'white')
            for i, image in enumerate(images): tile.paste(image, (480*i, 30))
            ImageDraw.Draw(tile).text((12, 8), f'{stage}: cuff path {report["insertion_path_m"]*100:.1f} cm', fill='black')
            thumbnails.append(tile)
            snapshot = env.sock_cloth.data.get('prepared_sock_state')
            if not snapshot:
                raise RuntimeError('Native prepared state capture unavailable')
            env.sock_cloth.restore_prepared_state(snapshot)
            env.sock_cloth.request_prepared_state()
            env._env.step(simulate=False)
            restored = env.sock_cloth.data.get('prepared_state_restore_report', {})
            readback = validate_prepared_state_readback(snapshot, env.sock_cloth.data['prepared_sock_state'])
            if not restored.get('ok') or not readback['ok']:
                raise RuntimeError('Preinsert state restore validation failed')
            report['state_restore_verified'] = True
            (output/f'{stage}-prepared-state.json').write_text(snapshot+'\n')
            (output/f'{stage}-geometry.json').write_text(json.dumps(dict(source=source,
                particles_world=particles.tolist(), triangles=env.sock_cloth.data['particle_surface_triangles'],
                edges=env.sock_cloth.data['particle_edges'], rest_edge_lengths=env.sock_cloth.data['particle_rest_edge_lengths'],
                report=report), indent=2)+'\n')
            records.append(report)
            print(f'{stage}: initialized {len(particles)} particles; native readback error {report["maximum_readback_error_m"]:.2g} m', flush=True)
    montage = Image.new('RGB', (960, 390*len(thumbnails)), 'white')
    for i, tile in enumerate(thumbnails): montage.paste(tile, (0, 390*i))
    montage.save(output/'A-E-overview.jpg')
    (output/'report.json').write_text(json.dumps(dict(records=records, physics_settling_validated=False), indent=2)+'\n')
    print(output/'A-E-overview.jpg')
    if not all(record['geometry_audit']['geometry_ok'] and record['surface_audit']['surface_ok'] for record in records):
        raise RuntimeError('Preinsert geometry audit failed; see report.json and preview images')


if __name__ == '__main__':
    main()
