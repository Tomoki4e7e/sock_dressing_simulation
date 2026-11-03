#!/usr/bin/env python3
"""Reproduce workbook sock/ankle patterns in Unity with held visual geometry."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config, resolve_package_path
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.pose_patterns import initialize_pattern, read_patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/pose_patterns.yaml')
    parser.add_argument('--numbers', nargs='+', type=int, help='Workbook serial numbers; default all 50')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/sock-pose-pattern/held-gravity')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--amplitude-m', type=float, choices=[.01, .015, .02])
    parser.add_argument('--seed', type=int, help='Reproducible local defect placement')
    parser.add_argument('--port', type=int, default=13000)
    parser.add_argument('--hold-steps', type=int, default=5)
    args = parser.parse_args()
    if args.hold_steps < 1: parser.error('--hold-steps must be positive')
    if args.seed is not None and args.seed < 0: parser.error('--seed must be nonnegative')
    config = load_config(args.config)
    settings = config.pop('pose_pattern')
    workbook = resolve_package_path(settings.pop('workbook'))
    settings.pop('number')
    if settings.get('held_reference') and (args.amplitude_m is not None or args.seed is not None):
        parser.error('Held-reference mode preserves shape; amplitude/seed options are disabled')
    if args.amplitude_m is not None: settings['amplitude_m'] = args.amplitude_m
    if args.seed is not None: settings['seed'] = args.seed
    patterns = read_patterns(workbook)
    if args.numbers:
        unknown = set(args.numbers)-{p.number for p in patterns}
        if unknown: parser.error(f'Unknown serial numbers: {sorted(unknown)}')
        patterns = [p for p in patterns if p.number in args.numbers]
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = dict(workbook=str(workbook), workbook_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
        settings=settings, patterns=[asdict(p) for p in patterns],
        conventions=dict(pitch_zero='dorsum vertically up; toe axis horizontal',
            positive_pitch='raise toes', positive_roll='right-hand about ankle-to-toes',
            positive_shift='towards heel/calf along insertion path',
            positive_diagonal='right-hand about local dorsal axis of opening',
            rotation_order='pitch then roll; diagonal deforms sock opening only'),
        mode='held_reference_ankle_only' if settings.get('held_reference') else 'visual_deformed_pattern',
        applied_only=['pitch_deg', 'roll_deg'] if settings.get('held_reference') else 'all_workbook_conditions',
        unique_angle_pairs=len({(p.pitch_deg, p.roll_deg) for p in patterns}))
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    if args.manifest_only:
        print(out/'manifest.json'); return
    config['rcareworld'].update(graphics=not args.headless, port=args.port)
    config['assets']['output_dir'] = str(out/'assets')
    config['camera']['fov'] = 50.
    scenario = scenario_from_config(config, seed=0); prepared = _prepare(config, scenario)
    records = []; tiles = []; failures = []
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']), out/'assets/sock.obj', initial_joints=scenario.initial_joints)
        env.apply_scenario(scenario)
        env._read_only_native_observation = True
        for pattern in patterns:
            started = time.monotonic(); directory = out/f'{pattern.number:02d}-{pattern.sock_id}'
            directory.mkdir(exist_ok=True)
            try:
                if not args.headless:
                    env.sock_cloth._send_data('SetSockInspectionMaterial', True, env.robot.id); env._env.step(simulate=False)
                report = initialize_pattern(env, pattern, **settings)
                attr = env.sock_cloth; source = env._pose_pattern_source
                before = attr.particles().copy()
                env.advance_physics(args.hold_steps)
                attr.request_particles(); env._env.step(simulate=False)
                error = float(np.max(np.linalg.norm(attr.particles()-before, axis=1)))
                report['held_physics_steps'] = args.hold_steps
                report['maximum_hold_motion_m'] = error
                if report.get('gravity_drape'):
                    attr._send_data('GetSockPatternGravityDrape'); env._env.step(simulate=False)
                    drape = dict(attr.data['sock_pattern_gravity_drape'])
                    report['post_hold_gravity_drape'] = drape
                    report['maximum_supported_motion_m'] = drape['maximum_fixed_error_m']
                    if drape['maximum_fixed_error_m'] > 5e-6: raise RuntimeError('Supported region moved')
                    from sock_dressing_simulation.held_ankle_poses import audit_actual_skin
                    body, surface = audit_actual_skin(attr.particles(), source['skin_surface'], attr.data['particle_edges'], attr.data['particle_surface_triangles'])
                    report.update(geometry_audit=body, surface_audit=surface)
                    if not body['geometry_ok'] or not surface['surface_ok']: raise RuntimeError(f'Drape intersection after hold: {body}; {surface}')
                elif error > 5e-6: raise RuntimeError(f'Held shape moved {error} m')
                state = dict(mode='held_visual_shape', pattern=asdict(pattern), settings=settings,
                    particles_world=attr.particles().tolist(), opening_normal=report['opening_inward_normal_world'],
                    source=source, triangles=attr.data['particle_surface_triangles'],
                    edges=attr.data['particle_edges'], rest_edge_lengths=attr.data['particle_rest_edge_lengths'])
                (directory/'visual-state.json').write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
                runtime = dict(config); runtime['pose_pattern'] = dict(settings, number=pattern.number, workbook=str(workbook))
                (directory/'runtime-config.yaml').write_text(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False))
                if not args.headless:
                    attr._send_data('SetSockInspectionMaterial', True, env.robot.id); env._env.step(simulate=False)
                    center = (before.min(axis=0)+before.max(axis=0))*.5
                    forward = np.asarray(report['native_pose']['toe_direction'])
                    dorsal = np.asarray(report['native_pose']['dorsal_direction'])
                    horizontal = forward.copy(); horizontal[1] = 0
                    side = np.cross(horizontal, [0., 1., 0.]); side /= np.linalg.norm(side)
                    thumbs = []
                    for name, direction in [('side', side), ('dorsal', dorsal)]:
                        position = center+direction*.60
                        look = center-position
                        yaw = np.degrees(np.arctan2(look[0], look[2]))
                        pitch = -np.degrees(np.arctan2(look[1], np.linalg.norm(look[[0, 2]])))
                        env.recording_camera.SetTransform(position=position.tolist(), rotation=[float(pitch), float(yaw), 0.])
                        env._env.step(simulate=False)
                        # Complete a render after material/camera changes before
                        # consuming RGB; this must not advance cloth physics.
                        env._capture_recording_camera(simulate=False)
                        image = Image.fromarray(env._capture_recording_camera(simulate=False)['rgb'])
                        image.save(directory/(name+'.png')); image.thumbnail((400, 300)); thumbs.append(image.copy())
                    attr._send_data('SetSockInspectionMaterial', False, env.robot.id); env._env.step(simulate=False)
                    tile = Image.new('RGB', (800, 330), 'white')
                    for i, image in enumerate(thumbs): tile.paste(image, (400*i, 30))
                    ImageDraw.Draw(tile).text((8, 8), f'{pattern.number:02d} {pattern.sock_id} {report["stage"]} pitch={pattern.pitch_deg:g} roll={pattern.roll_deg:g}', fill='black')
                    tile.save(directory/'views.jpg'); tiles.append(tile)
                report['wall_elapsed_s'] = time.monotonic()-started
                (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
                (directory/'failure.json').unlink(missing_ok=True)
                (directory/'contact-diagnostic.json').unlink(missing_ok=True)
                records.append(report)
                print(f'{pattern.number:02d}: PASS pitch={report["measured_pitch_deg"]:.1f} clearance={report["clearance_m"]*1000:g}mm cuff={report["cuff_perimeter_m"]*100:.1f}cm defects={report["local_feature_count"]} hold_error={error:.2g}m', flush=True)
            except Exception as error:
                gravity_debug = getattr(env, '_gravity_drape_debug', None)
                if gravity_debug is not None:
                    (directory/'gravity-diagnostic.json').write_text(json.dumps(gravity_debug, indent=2)+'\n')
                diagnostic = getattr(env, '_held_reference_collision', None)
                if diagnostic is not None:
                    (directory/'contact-diagnostic.json').write_text(json.dumps(diagnostic, indent=2)+'\n')
                    env._held_reference_collision = None
                failures.append(dict(number=pattern.number, error=str(error)))
                (directory/'failure.json').write_text(json.dumps(failures[-1], ensure_ascii=False, indent=2)+'\n')
                print(f'{pattern.number:02d}: FAIL {error}', flush=True)
            if not args.headless:
                env.sock_cloth._send_data('SetSockInspectionMaterial', False, env.robot.id); env._env.step(simulate=False)
            (out/'report.json').write_text(json.dumps(dict(records=records, failures=failures,
                passed=len(records)==len(patterns) and not failures), ensure_ascii=False, indent=2)+'\n')
    for first in range(0, len(tiles), 10):
        page = Image.new('RGB', (1600, 330*5), 'white')
        for i, tile in enumerate(tiles[first:first+10]): page.paste(tile, ((i%2)*800, (i//2)*330))
        page.save(out/f'overview-{first//10+1}.jpg')
    if failures: raise SystemExit(2)


if __name__ == '__main__': main()
