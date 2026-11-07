#!/usr/bin/env python3
"""Render and verify the C--E surfaces prepared with A's method in Unity."""
import os
os.environ.setdefault('__GLX_VENDOR_LIBRARY_NAME', 'mesa')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '.deps/RCareWorld/pyrcareworld')]
from prepare_held_stage_shapes import local_runtime_locks, write
from prepare_bulged_cuffs import write_obj
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.pose_patterns import read_patterns
from sock_dressing_simulation.held_stage_shapes import DEFAULT_ROOT, load_stage_state, STAGES
from sock_dressing_simulation.held_ankle_poses import foot_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number', type=int, required=True)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--root', type=Path, default=ROOT / DEFAULT_ROOT)
    parser.add_argument('--compare-root', type=Path, help='Previous frozen shapes, captured with the exact same cameras')
    parser.add_argument('--port', type=int, default=18400)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    pattern = next(p for p in read_patterns(ROOT / 'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx') if p.number == args.number)
    out = args.root.resolve() / f'preview-{args.number:02d}'
    out.mkdir(parents=True, exist_ok=True)
    images, records = [], []
    for index, stage in enumerate(args.stages):
        state = load_stage_state(args.root, pattern, stage)
        directory = out / stage
        directory.mkdir(exist_ok=True)
        rest, faces = np.array(state['source']['rest_positions']), np.array(state['triangles']).reshape(-1, 3)
        previous = None
        if args.compare_root:
            old = json.loads((args.compare_root/f'pose-{args.number:02d}-{stage}/state.json').read_text())
            if np.shape(old['source']['rest_positions']) != rest.shape or not np.allclose(old['source']['rest_positions'],rest,atol=1e-9,rtol=0):
                raise ValueError('Comparison requires matching material meshes')
            previous = np.array(old['particles_world'])
        write_obj(directory / 'rest.obj', rest, faces, native_import=True)
        config = load_config(ROOT / 'config/pose_patterns.yaml')
        config.pop('pose_pattern')
        config['camera']['fov'] = 45.
        config['rcareworld'].update(graphics=not args.headless, port=args.port+index, log_level=0,
                                   executable=str(ROOT / 'scripts/run_stage_preparation_player.sh'))
        os.environ['SOCK_STAGE_PLAYER_LOG'] = str(directory / 'player.log')
        config['assets']['output_dir'] = str(directory / 'assets')
        local_runtime_locks(directory / 'runtime-locks')
        scenario = scenario_from_config(config, seed=0)
        prepared = _prepare(config, scenario)
        with SockDressingEnv(config) as env:
            env.load(Path(prepared['runtime_urdf']), directory / 'rest.obj', initial_joints=scenario.initial_joints)
            env.apply_scenario(scenario)
            attr = env.sock_cloth
            attr.set_pattern_foot_pose(pattern.pitch_deg, pattern.roll_deg)
            attr.request_preinsert_source()
            attr._send_data('GetSockPatternSkin')
            env._env.step(2, simulate=False)
            native_rest = np.array(attr.data['sock_preinsert_source']['rest_positions'])
            distances, order = cKDTree(rest).query(native_rest)
            if len(order) != len(rest) or len(set(order)) != len(rest) or distances.max() > 5e-6:
                raise ValueError('Native refined cloth topology differs')
            expected = np.array(state['particles_world'])[order]
            attr.initialize_preinsert(expected, state['opening_normal'])
            env._env.step(2, simulate=False)
            attr.set_pattern_held(True)
            attr.request_particles()
            env._env.step(2, simulate=False)
            cloth_error = float(np.linalg.norm(attr.particles()-expected, axis=1).max())
            skin_error = float(np.linalg.norm(np.array(attr.data['sock_pattern_skin']['vertices'])-
                                             state['source']['skin_surface']['vertices'], axis=1).max())
            record = dict(stage=stage, cloth_error_m=cloth_error, skin_error_m=skin_error,
                          passed=max(cloth_error, skin_error) < 5e-6)
            write(directory / 'native-validation.json', record)
            if not record['passed']:
                raise ValueError(f'Native surface readback failed: {record}')
            records.append(record)
            if args.headless:
                continue
            attr._send_data('SetSockInspectionMaterial', True, env.robot.id)
            env._env.step(simulate=False)
            foot = np.array(state['source']['surfaces'][0]['vertices'])
            bounds = np.vstack([expected, foot] + ([previous] if previous is not None else []))
            center = (bounds.min(0)+bounds.max(0))*.5
            basis = foot_basis(state['source'])
            horizontal = basis[:, 0].copy()
            horizontal[1] = 0
            side = np.cross(horizontal, [0., 1., 0.])
            side /= np.linalg.norm(side)
            for name, direction in [('side', side), ('dorsal', basis[:, 1])]:
                position = center + direction*.55
                look = center-position
                yaw = float(np.degrees(np.arctan2(look[0], look[2])))
                pitch = float(-np.degrees(np.arctan2(look[1], np.linalg.norm(look[[0, 2]]))))
                env.recording_camera.SetTransform(position=position.tolist(), rotation=[pitch, yaw, 0.])
                env._env.step(simulate=False)
                variants = [('before-', previous[order]), ('', expected)] if previous is not None else [('', expected)]
                for prefix, shape in variants:
                    attr.set_pattern_held(False)
                    env._env.step(simulate=False)
                    attr.initialize_preinsert(shape,state['opening_normal'])
                    env._env.step(simulate=False)
                    attr.set_pattern_held(True)
                    env._env.step(simulate=False)
                    env._capture_recording_camera(simulate=False)
                    image = Image.fromarray(env._capture_recording_camera(simulate=False)['rgb'])
                    image.save(directory / (prefix+name+'.png'))
                    image.thumbnail((400,300) if previous is not None else (480,360))
                    images.append((stage+' '+prefix+name, image.copy()))
            if previous is not None:
                record['comparison_uses_same_camera'] = True
                write(directory/'native-validation.json',record)
    write(out / 'native-validation.json', dict(passed=all(r['passed'] for r in records), records=records))
    if images:
        columns, width, height = (4,400,330) if args.compare_root else (2,480,390)
        canvas = Image.new('RGB', (columns*width, height*len(args.stages)), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (label, image) in enumerate(images):
            x, y = (i % columns)*width, (i // columns)*height
            canvas.paste(image, (x, y+25))
            draw.text((x+8, y+6), label, fill='black')
        canvas.save(out / 'C-E-overview.jpg')
    print(out, flush=True)


if __name__ == '__main__':
    main()
