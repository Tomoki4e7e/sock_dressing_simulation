#!/usr/bin/env python3
"""Compare ankle-to-toe size while translating only human/chair world Y and Z."""
import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import yaml

import replay_foot_contact as replay
from match_real_camera_composition import ROOT, REFERENCE, project
from sock_dressing_simulation.config import load_config

BASE = ROOT / 'config/autonomous30_current_ankle_total25deg_physics10_real_composition.yaml'
OUTPUT = ROOT / 'artifacts/phase4/real-right-camera-foot-size-20261014'
PLAYER = '/mnt/robot_ssd/sock_dressing_simulation_work/grasp_following_20261012_v2/player-contact-diagnostics/Player.x86_64'
# Manually annotated ankle boundary on the specified unscaled real frame.
REAL_ANKLE_BOUNDARY = [[490., 61.], [619., 73.]]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()) + '\n')


def foot_measurement(mask, ankle, toe_hint):
    """Axis length from ankle to distal outline, and maximum transverse width.

    Native ankle/toe landmarks define the simulator's anatomical axis. The
    real axis uses the annotated ankle and distal renderer/SAM mask pixels.
    Border-truncated outlines are reported and cannot pass full visibility.
    """
    ankle = np.asarray(ankle, dtype=float)
    axis = np.asarray(toe_hint, dtype=float) - ankle
    axis /= np.linalg.norm(axis)
    normal = np.array([axis[1], -axis[0]])
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    points = np.column_stack([xs, ys]).astype(float)
    axial = (points - ankle) @ axis
    points = points[axial >= 0]
    axial = axial[axial >= 0]
    if len(points) < 30:
        raise ValueError('Too few foot outline pixels beyond ankle boundary')
    transverse = (points - ankle) @ normal
    # Two-pixel slices avoid combining different axial portions into one width.
    bands = np.floor(axial / 2).astype(int)
    unique = np.unique(bands)
    widths = [(float(np.ptp(transverse[bands == b]) + 1), int(b)) for b in unique if np.sum(bands == b) >= 3]
    width, band = max(widths)
    selected = bands == band
    left = points[selected][np.argmin(transverse[selected])]
    right = points[selected][np.argmax(transverse[selected])]
    tip_pixels = points[axial >= axial.max() - 5]
    tip = tip_pixels.mean(axis=0)
    length = float(axial.max() + 1)
    height, image_width = np.asarray(mask).shape
    ankle_visible = bool(0 <= ankle[0] < image_width and 0 <= ankle[1] < height)
    touches_border = bool(np.any((points[:, 0] <= 0) | (points[:, 0] >= image_width - 1) |
                                (points[:, 1] <= 0) | (points[:, 1] >= height - 1)))
    return dict(ankle=ankle.tolist(), axis=axis.tolist(), tip=tip.tolist(),
                width_px=width, length_px=length, area_px=int(len(points)), width_endpoints=[left.tolist(), right.tolist()],
                center=points.mean(axis=0).tolist(), ankle_visible=ankle_visible,
                outline_truncated=touches_border, width_length_complete=ankle_visible and not touches_border,
                angle_degrees=float(np.degrees(np.arctan2(axis[0], axis[1]))))


def annotate(rgb, measurement, path):
    image = Image.fromarray(np.asarray(rgb).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    ankle = np.array(measurement['ankle'])
    axis = np.array(measurement['axis'])
    normal = np.array([axis[1], -axis[0]])
    draw.line([tuple(ankle - normal * 110), tuple(ankle + normal * 110)], fill='orange', width=3)
    draw.line([tuple(ankle), tuple(ankle + axis * measurement['length_px'])], fill='cyan', width=3)
    draw.line([tuple(p) for p in measurement['width_endpoints']], fill='red', width=3)
    for point in [measurement['ankle'], measurement['tip']]:
        x, y = point
        draw.ellipse((x-5, y-5, x+5, y+5), outline='yellow', width=3)
    draw.text((20, 20), f"width {measurement['width_px']:.1f}px  length {measurement['length_px']:.1f}px", fill='yellow')
    image.save(path)


def real_reference():
    mask = np.asarray(Image.open(REFERENCE / 'camera_right_mask/foot_mask/0.png')) > 0
    ys, xs = np.nonzero(mask)
    distal = ys >= ys.max() - 5
    toe = [float(np.mean(xs[distal])), float(np.mean(ys[distal]))]
    value = foot_measurement(mask, np.mean(REAL_ANKLE_BOUNDARY, axis=0), toe)
    value.update(image=str(REFERENCE / 'camera_right/0.png'), ankle_boundary=REAL_ANKLE_BOUNDARY,
                 annotation_method='Manual ankle boundary; distal foot mask outline. Orange=ankle cut, cyan=length, red=maximum transverse width.')
    return value


def configuration(dy, dz):
    base = load_config(BASE)
    pose = copy.deepcopy(base['scene']['initial_pose_contract']['locked_pose_baseline'])
    for key in ('human_root_position', 'chair_position', 'human_anchor_position', 'right_toe_position'):
        pose[key] = (np.asarray(pose[key]) + [0., dy, dz]).tolist()
    return dict(extends=str(BASE), scene=dict(human_position=pose['human_root_position'],
        initial_pose_contract=dict(locked_pose_baseline=pose,
            expected_human_position=pose['human_root_position'], expected_chair_position=pose['chair_position'],
            away_from_robot_m=None, down_m=None, up_m=None, right_from_robot_m=None,
            translate_to_foot_to_sock=False)))


def capture(env, directory, stage, target, observed_frame=None):
    directory = Path(directory)
    env.sock_cloth.request_dressing_qa()
    env.sock_cloth.request_scene_geometry()
    env.sock_cloth.request_visual_diagnostics(int(env.config['scene']['stable_ids']['chair']))
    env.sock_cloth.request_physics_clock()
    env._env.step(simulate=False)
    before = copy.deepcopy(env.sock_cloth.data['cloth_physics_clock'])
    qa = copy.deepcopy(env.sock_cloth.data['dressing_qa'])
    geometry = copy.deepcopy(env.sock_cloth.data['scene_geometry'])
    visual = copy.deepcopy(env.sock_cloth.data.get('visual_diagnostics', []))
    pose_checks = {item['role']: bool(item['valid']) for item in visual
                   if item.get('role') in ('human_task_pose', 'human_chair_lock')}
    if set(pose_checks) != {'human_task_pose', 'human_chair_lock'} or not all(pose_checks.values()):
        raise RuntimeError('Native human/chair preparation pose validation failed: ' + json.dumps(pose_checks))
    ankle = next(v['position'] for v in qa['foot_collision_safety']['foot_collider_geometry'] if v['region'] == 'ankle')
    frame = observed_frame if observed_frame is not None else env._capture_camera(simulate=False)
    camera = copy.deepcopy(frame['raw'])
    points, depths = project([ankle, geometry['right_toe_position']], np.asarray(camera['position']), camera['quaternion'], env.config['camera'])
    measured = foot_measurement(frame['leg_mask'], points[0], points[1])
    # ID images retain occlusion, unlike the amodal renderer leg mask. Match
    # the native EncodeIDAsColor values in Gamma or Linear project color space.
    visible = np.zeros(frame['leg_mask'].shape, dtype=bool)
    for instance_id in [env.config['scene']['human_id'], *env.config['scene']['human_leg_mask_ids']]:
        color = np.array([(int(instance_id) * multiplier + offset) % 256
                          for multiplier, offset in [(16807, 187), (48271, 79), (95849, 233)]])
        linear = color / 255.
        gamma = np.round(np.where(linear <= .0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - .055) * 255)
        for expected in [color, gamma]:
            visible |= np.max(np.abs(frame['instance_mask'].astype(float) - expected), axis=-1) <= 2
    yy, xx = np.mgrid[:frame['leg_mask'].shape[0], :frame['leg_mask'].shape[1]]
    axial = (xx - measured['ankle'][0]) * measured['axis'][0] + (yy - measured['ankle'][1]) * measured['axis'][1]
    foot_pixels = frame['leg_mask'] & (axial >= 0)
    def visible_fraction(region):
        count = np.count_nonzero(region)
        return float(np.count_nonzero(region & visible) / count) if count else 0.
    measured.update(visible_foot_fraction=visible_fraction(foot_pixels),
                    visible_ankle_fraction=visible_fraction(foot_pixels & (axial < 20)),
                    visible_toe_fraction=visible_fraction(foot_pixels & (axial >= measured['length_px'] - 16)))
    measured.update(width_relative_error=measured['width_px'] / target['width_px'] - 1,
                    length_relative_error=measured['length_px'] / target['length_px'] - 1,
                    area_relative_error=measured['area_px'] / target['area_px'] - 1,
                    center_error_px=(np.asarray(measured['center']) - target['center']).tolist(),
                    angle_error_degrees=measured['angle_degrees'] - target['angle_degrees'])
    Image.fromarray(frame['rgb']).save(directory / f'{stage}.png')
    Image.fromarray(frame['leg_mask'].astype(np.uint8) * 255).save(directory / f'{stage}_leg_mask.png')
    Image.fromarray(frame['instance_mask']).save(directory / f'{stage}_instance_mask.png')
    annotate(frame['rgb'], measured, directory / f'{stage}_annotated.png')
    # Independent view for floor clearance, rigid posture, and physical placement.
    import pyrcareworld.attributes as attributes
    overview = env._env.attrs.get(1400) or env._env.InstanceObject(name='Camera', id=1400, attr_type=attributes.CameraAttr)
    overview.SetTransform(position=[1.25, 1.2, 1.25], rotation=[0., 0., 0.], scale=[1., 1., 1.])
    overview.LookAt([-.12, .65, -.1])
    overview.GetRGB(1280, 960, 70.)
    env._env.step(simulate=False)
    Image.fromarray(env._decode(overview.data['rgb'], 'RGB')).save(directory / f'{stage}_overview.png')
    env.sock_cloth.request_physics_clock()
    env._env.step(simulate=False)
    after = copy.deepcopy(env.sock_cloth.data['cloth_physics_clock'])
    if before != after:
        raise RuntimeError('Foot measurement capture advanced physics')
    report = dict(stage=stage, measured=measured, reference=target,
                  clock_before=before, clock_after=after, dressing_qa=qa, scene_geometry=geometry,
                  visual_diagnostics=visual, camera_world_position=camera['position'], camera_world_quaternion=camera['quaternion'],
                  preparation_pose_checks=pose_checks,
                  landmark_depth_m=depths.tolist(), native_ankle_world=ankle,
                  initial_pose_contract=env.config['scene']['initial_pose_contract'],
                  camera_local_position=env.config['scene']['camera_local_position'],
                  camera_local_rotation=env.config['scene']['camera_local_rotation'])
    write_json(directory / f'{stage}.json', report)
    print('foot size ' + json.dumps(dict(stage=stage, measured=measured)), flush=True)
    return report


class PreviewComplete(Exception):
    """Leave the environment context after successful preparation, before policy creation."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--dy', type=float, default=0.)
    parser.add_argument('--dz', type=float, default=0.)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    if abs(args.dy) > .150001 or abs(args.dz) > .200001:
        parser.error('Translation must stay within Y±15cm and Z±20cm from the current composition')
    if not args.name.replace('_', '').replace('-', '').isalnum():
        parser.error('Use an alphanumeric candidate name')
    OUTPUT.mkdir(exist_ok=True)
    directory = OUTPUT / args.name
    path = OUTPUT / f'{args.name}.yaml'
    if directory.exists() or path.exists():
        raise FileExistsError(f'Existing data is preserved; choose a new candidate name: {args.name}')
    path.write_text(yaml.safe_dump(configuration(args.dy, args.dz), sort_keys=False))
    target = real_reference()
    write_json(OUTPUT / 'reference.json', target)
    annotate(np.asarray(Image.open(target['image']).convert('RGB')), target, OUTPUT / 'reference_annotated.png')
    original = replay.run_demo

    def run(config, **kwargs):
        factory = kwargs['environment_factory']

        class Environment(factory):
            def begin_cloth_contact_rollout(self):
                super().begin_cloth_contact_rollout()
                capture(self, directory, 'before_policy', target)
                if args.preview_only:
                    write_json(directory / 'preview.json', dict(preparation_completed=True, frames_inferred=0,
                        frames_replayed=0, post_preparation_physics_steps=0, delta_world_m=[0., args.dy, args.dz],
                        prepared_state_restored=False, automatic_camera_alignment=False))
                    raise PreviewComplete()

            def command(self, command, previous=None):
                result = super().command(command, previous)
                if self._cloth_contact_rollout_started:
                    self._foot_size_command_seen = True
                return result

            def observe(self):
                value = super().observe()
                if getattr(self, '_foot_size_command_seen', False) and not getattr(self, '_foot_size_after_saved', False):
                    self._foot_size_after_saved = True
                    capture(self, directory, 'first_inference', target, value['camera'])
                return value

        kwargs['environment_factory'] = Environment
        return original(config, **kwargs)

    replay.run_demo = run
    sys.argv = ['replay_foot_contact', '--config', str(path), '--autonomous', '--device', 'cuda', '--steps', '1',
        '--surface-mode', 'local', '--transport', 'local', '--fixed-grasp-offsets', '--post-preparation-material',
        '--player-executable', PLAYER, '--output-root', str(OUTPUT), '--candidate-name', args.name, '--port', '5380']
    if not args.preview_only:
        sys.argv.append('--physics-video')
    try:
        replay.main()
    except PreviewComplete:
        print('Preview completed without policy inference or rollout physics.', flush=True)


if __name__ == '__main__':
    main()
