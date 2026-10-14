#!/usr/bin/env python3
"""Fit a right-mounted camera against an annotated real first frame."""
from pathlib import Path
import argparse
import copy
import json
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
import yaml

import replay_foot_contact as replay

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'ShareSet/share_folder/data/data_center_general_depth_n5_sock/train/2026-01-13-14-42-36_30deg'
BASE = ROOT / 'config/autonomous30_current_ankle_total25deg_physics10_camera_centered.yaml'


def mask_features(mask):
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if len(xs) < 30:
        raise ValueError('Not enough mask pixels')
    return dict(center=[float(xs.mean()), float(ys.mean())],
                bbox=[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                width=int(xs.max() - xs.min() + 1), height=int(ys.max() - ys.min() + 1))


def reference():
    # Centers of the visible metallic gripping faces, in the unscaled real image.
    polygons = [[[243, 574], [382, 607], [394, 640], [231, 623]],
                [[650, 617], [810, 589], [813, 632], [657, 644]]]
    pads = np.asarray([np.mean(points, axis=0) for points in polygons])
    return dict(image=str(REFERENCE / 'camera_right/0.png'), pad_polygons=polygons,
        pad_centers=pads.tolist(),
        foot=mask_features(Image.open(REFERENCE / 'camera_right_mask/foot_mask/0.png')),
        sock=mask_features(Image.open(REFERENCE / 'camera_right_mask/sock_mask/0.png')))


def project(points, position, quaternion, camera):
    relative = Rotation.from_quat(quaternion).inv().apply(np.asarray(points) - position)
    f = camera['height'] / (2 * np.tan(np.deg2rad(camera['fov']) / 2))
    z = relative[:, 2]
    xy = np.column_stack((camera['width'] / 2 + f * relative[:, 0] / z,
                          camera['height'] / 2 - f * relative[:, 1] / z))
    return xy, z


def fit_camera(env, directory, target, iterations, seed_position=None, seed_rotation=None, forecast=None, shape=False):
    out = Path(directory) / 'composition'
    out.mkdir(exist_ok=True)
    scene = env.config['scene']
    env.sock_cloth.request_scene_geometry()
    env.sock_cloth.request_physics_clock()
    env._env.step(simulate=False)
    geometry = copy.deepcopy(env.sock_cloth.data['scene_geometry'])
    before = copy.deepcopy(env.sock_cloth.data['cloth_physics_clock'])
    pad_world = np.asarray([geometry['left_grasp_position'], geometry['right_grasp_position']])
    if forecast:
        previous = [json.loads(line) for line in (Path(forecast) / 'live_frame_diagnostics.jsonl').read_text().splitlines()][0]['scene_geometry']
        pad_world = np.asarray([previous['left_grasp_position'], previous['right_grasp_position']])
    desired = np.asarray(target['pad_centers'])
    parameters = np.array((seed_position or scene['camera_local_position']) + (seed_rotation or scene['camera_local_rotation']), dtype=float)
    start = parameters.copy()
    samples = []

    def measure(p, save=False):
        env.camera.SetTransform(position=p[:3].tolist(), rotation=p[3:].tolist(), scale=[1., 1., 1.], is_world=False)
        env._env.step(simulate=False)
        data = env.camera.data
        xy, z = project(pad_world, np.asarray(data['position']), data['quaternion'], env.config['camera'])
        # Preserve correspondence and prefer small camera motions in the underdetermined fit.
        residual = np.r_[((xy - desired) / [1280., 960.]).ravel(), (p[:3] - start[:3]) * .025,
                         (p[3:] - start[3:]) * .00008, np.maximum(.1 - z, 0) * 10]
        capture = None
        if shape:
            capture = env._capture_camera(simulate=False)
            terms = [((xy - desired) / [38.4, 28.8]).ravel()]
            for feature, key, weight in [('sock', 'sock_mask', 1.), ('foot', 'leg_mask', .25)]:
                try:
                    value = mask_features(capture[key])
                    terms.append(weight * np.r_[(np.asarray(value['center']) - target[feature]['center']) / [38.4, 28.8],
                        (value['width'] / target[feature]['width'] - 1) / .15,
                        .5 * (value['height'] / target[feature]['height'] - 1) / .15])
                except ValueError:
                    terms.append(np.full(4, 100.))
            terms.extend([(p[:3] - start[:3]) * .1, (p[3:] - start[3:]) * .001, np.maximum(.1 - z, 0) * 100])
            residual = np.concatenate(terms)
        if save:
            if capture is None:
                capture = env._capture_camera(simulate=False)
            def features(mask):
                try:
                    return mask_features(mask)
                except ValueError:
                    return dict(visible=False)
            sample = dict(position=p[:3].tolist(), rotation=p[3:].tolist(), pad_pixels=xy.tolist(), pad_depth_m=z.tolist(),
                world_position=list(env.camera.data['position']), world_quaternion=list(env.camera.data['quaternion']),
                foot=features(capture['leg_mask']), sock=features(capture['sock_mask']))
            index = len(samples)
            img = Image.fromarray(capture['rgb'])
            draw = ImageDraw.Draw(img)
            for x, y in xy:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline='red', width=3)
            img.save(out / f'trial_{index:02d}.png')
            sample['image'] = str(out / f'trial_{index:02d}.png')
            samples.append(sample)
            print('composition ' + json.dumps(sample), flush=True)
        return residual

    residual = measure(parameters, True)
    for _ in range(iterations):
        eps = np.array([.003] * 3 + [.4] * 3)
        jacobian = []
        for axis in range(6):
            probe = parameters.copy(); probe[axis] += eps[axis]
            jacobian.append((measure(probe) - residual) / eps[axis])
        delta = np.linalg.lstsq(np.column_stack(jacobian), -residual, rcond=None)[0]
        delta = np.clip(delta, [-.035] * 3 + [-6.] * 3, [.035] * 3 + [6.] * 3)
        proposed = parameters + delta
        proposed[:3] = np.clip(proposed[:3], start[:3] - .35, start[:3] + .35)
        proposed[3:] = np.clip(proposed[3:], [-165., -85., -85.], [165., 85., 85.])
        candidate = measure(proposed)
        if np.linalg.norm(candidate) > np.linalg.norm(residual):
            for factor in [.25, .0625, 0.]:
                proposed = parameters + delta * factor
                if np.linalg.norm(measure(proposed)) <= np.linalg.norm(residual):
                    break
        parameters = proposed
        residual = measure(parameters, True)
        if not shape and np.max(np.abs(residual[:4])) < .012:
            break
    measure(parameters)
    scene['camera_local_position'] = parameters[:3].tolist()
    scene['camera_local_rotation'] = parameters[3:].tolist()
    env._camera_mount_report.update(local_position=parameters[:3].tolist(), local_rotation=parameters[3:].tolist())
    env.config['inference']['camera_alignment_applied'] = True
    env.sock_cloth.request_physics_clock()
    env._env.step(simulate=False)
    after = copy.deepcopy(env.sock_cloth.data['cloth_physics_clock'])
    if before != after:
        raise RuntimeError('Camera trials advanced physics')
    frame = env._capture_camera(simulate=False)
    Image.fromarray(frame['rgb']).save(out / 'before_policy.png')
    Image.fromarray(frame['leg_mask'].astype(np.uint8) * 255).save(out / 'foot_before_policy.png')
    # An independent overview helps inspect chair/floor clearance and camera placement.
    import pyrcareworld.attributes as attributes
    overview = env._env.attrs.get(1400) or env._env.InstanceObject(name='Camera', id=1400, attr_type=attributes.CameraAttr)
    overview.SetTransform(position=[1.25, 1.2, 1.25], rotation=[0., 0., 0.], scale=[1., 1., 1.])
    overview.LookAt([-0.12, .65, -.1])
    overview.GetRGB(1280, 960, 70.)
    env._env.step(simulate=False)
    overview_rgb = env._decode(overview.data['rgb'], 'RGB')
    Image.fromarray(overview_rgb).save(out / 'overview.png')
    env.sock_cloth.request_physics_clock()
    env._env.step(simulate=False)
    after = copy.deepcopy(env.sock_cloth.data['cloth_physics_clock'])
    if before != after:
        raise RuntimeError('Overview capture advanced physics')
    report = dict(reference=target, samples=samples, camera_position=parameters[:3].tolist(), camera_rotation=parameters[3:].tolist(),
        clock_before=before, clock_after=after, scene_geometry=geometry, forecast_grasp_source=str(forecast) if forecast else None)
    (out / 'fit.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def configuration(base, delta, camera_position, camera_rotation):
    pose = copy.deepcopy(base['scene']['initial_pose_contract']['locked_pose_baseline'])
    for key in ('human_root_position', 'human_anchor_position', 'chair_position', 'right_toe_position'):
        pose[key] = (np.asarray(pose[key]) + delta).tolist()
    return dict(extends=str(BASE), scene=dict(camera_parent_link='head/see3cam_right/camera_color_frame',
        camera_local_position=camera_position, camera_local_rotation=camera_rotation,
        human_position=pose['human_root_position'], initial_pose_contract=dict(locked_pose_baseline=pose,
            expected_human_position=pose['human_root_position'], expected_chair_position=pose['chair_position'],
            away_from_robot_m=None, down_m=None, up_m=None, right_from_robot_m=None, translate_to_foot_to_sock=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--translation', nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument('--position', nargs=3, type=float, default=[0., -.04, .1])
    parser.add_argument('--rotation', nargs=3, type=float, default=[60., 0., 0.])
    parser.add_argument('--iterations', type=int, default=18)
    parser.add_argument('--compare-pitches', nargs='+', type=float)
    parser.add_argument('--forecast-from', type=Path)
    parser.add_argument('--shape-after-command', action='store_true', help='Exploratory frozen camera fit after the first live action; requires a subsequent fresh validation.')
    args = parser.parse_args()
    root = ROOT / 'artifacts/phase4/real-right-camera-composition-20261013'
    root.mkdir(exist_ok=True)
    if (root / args.name).exists() or (root / f'{args.name}.yaml').exists():
        raise FileExistsError(f'Choose a new candidate name; existing data is preserved: {args.name}')
    from sock_dressing_simulation.config import load_config
    # Preparation uses a known visible view; candidate transforms are fitted only after it freezes.
    config = configuration(load_config(BASE), np.asarray(args.translation), [0., -.04, .1], [60., 0., 0.])
    path = root / f'{args.name}.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    target = reference()
    (root / 'reference.json').write_text(json.dumps(target, indent=2) + '\n')
    annotated = Image.open(target['image']).convert('RGB'); draw = ImageDraw.Draw(annotated)
    for polygon in target['pad_polygons']:
        draw.polygon([tuple(p) for p in polygon], outline='red', width=3)
    annotated.save(root / 'reference_annotated.png')
    original = replay.run_demo

    def run(config, **kwargs):
        factory = kwargs['environment_factory']

        class Environment(factory):
            def command(self, command, previous=None):
                result = super().command(command, previous)
                if self._cloth_contact_rollout_started:
                    self._composition_command_seen = True
                return result

            def observe(self):
                if args.shape_after_command and getattr(self, '_composition_command_seen', False) and not getattr(self, '_composition_shape_done', False):
                    self._composition_shape_done = True
                    trial = root / args.name / 'shape_after_command'
                    trial.mkdir(exist_ok=True)
                    fit_camera(self, trial, target, 12, shape=True)
                    (root / args.name / 'config.yaml').write_text(yaml.safe_dump(self.config, sort_keys=False))
                value = super().observe()
                if self._cloth_contact_rollout_started:
                    camera = value['camera']['raw']
                    (root / args.name / 'composition/camera_after_observation.json').write_text(json.dumps(dict(
                        position=camera['position'], quaternion=camera['quaternion']), indent=2) + '\n')
                return value

            def begin_cloth_contact_rollout(self):
                super().begin_cloth_contact_rollout()
                if args.compare_pitches:
                    import shutil
                    trials = []
                    for pitch in args.compare_pitches:
                        trial = root / args.name / 'camera_search' / f'pitch_{pitch:g}'
                        trial.mkdir(parents=True, exist_ok=True)
                        position = [0., -.2, 0.] if pitch < 0 else [0., -.04, .1]
                        report = fit_camera(self, trial, target, args.iterations, position, [pitch, 0., 0.])
                        sample = report['samples'][-1]
                        score = 0.
                        for feature in ('foot', 'sock'):
                            value = sample[feature]
                            if 'center' not in value:
                                score += 1000
                                continue
                            score += float(np.sum(((np.asarray(value['center']) - target[feature]['center']) / [38.4, 28.8]) ** 2))
                            score += ((value['width'] / target[feature]['width'] - 1) / .15) ** 2
                            score += .25 * ((value['height'] / target[feature]['height'] - 1) / .15) ** 2
                        downward = np.degrees(np.arcsin(-Rotation.from_quat(sample['world_quaternion']).apply([0., 0., 1.])[1]))
                        if downward < 51:
                            score += 1000
                        trials.append(dict(pitch=pitch, score=score, directory=str(trial), report=report))
                        print('camera search score ' + json.dumps(dict(pitch=pitch, score=score, downward_angle=downward)), flush=True)
                    best = min(trials, key=lambda item: item['score'])
                    (root / args.name / 'camera_search/results.json').write_text(json.dumps(trials, indent=2) + '\n')
                    report = best['report']
                    self.camera.SetTransform(position=report['camera_position'], rotation=report['camera_rotation'], scale=[1., 1., 1.], is_world=False)
                    self._env.step(simulate=False)
                    self.config['scene'].update(camera_local_position=report['camera_position'], camera_local_rotation=report['camera_rotation'])
                    self._camera_mount_report.update(local_position=report['camera_position'], local_rotation=report['camera_rotation'])
                    shutil.copytree(Path(best['directory']) / 'composition', root / args.name / 'composition', dirs_exist_ok=True)
                else:
                    fit_camera(self, root / args.name, target, args.iterations, args.position, args.rotation, args.forecast_from)
                (root / args.name / 'config.yaml').write_text(yaml.safe_dump(self.config, sort_keys=False))

        kwargs['environment_factory'] = Environment
        return original(config, **kwargs)

    replay.run_demo = run
    sys.argv = ['replay_foot_contact', '--config', str(path), '--autonomous', '--device', 'cuda', '--steps', '1',
        '--physics-video', '--surface-mode', 'local', '--transport', 'local', '--fixed-grasp-offsets', '--post-preparation-material',
        '--player-executable', '/mnt/robot_ssd/sock_dressing_simulation_work/grasp_following_20261012_v2/player-contact-diagnostics/Player.x86_64',
        '--output-root', str(root), '--candidate-name', args.name, '--port', '5370']
    replay.main()


if __name__ == '__main__':
    main()
