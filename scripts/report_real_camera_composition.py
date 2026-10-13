#!/usr/bin/env python3
"""Compare a real camera frame with the first live simulated command."""
from pathlib import Path
import argparse
import json
import shutil

import numpy as np
from PIL import Image, ImageDraw
import yaml

from match_real_camera_composition import ROOT, BASE, mask_features, project
from sock_dressing_simulation.config import load_config

OLD_EPISODE = ROOT / 'artifacts/phase4/current-ankle-total25deg-physics10-camera-centered-20261013/autonomous30/data_sock_sim_smoke/train/phase4_20261013T064852Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    fit = json.loads((directory / 'composition/fit.json').read_text())
    reference = fit['reference']
    result = json.loads((directory / 'result.json').read_text())
    episode = Path(result['metadata']).parent
    metadata = json.loads((episode / 'metadata.json').read_text())
    config = load_config(directory / 'config.yaml')
    baseline = load_config(BASE)
    live = [json.loads(line) for line in (directory / 'live_frame_diagnostics.jsonl').read_text().splitlines()]
    world = live[0]['scene_geometry']
    pose_path = directory / 'composition/camera_after_observation.json'
    camera_pose = json.loads(pose_path.read_text()) if pose_path.exists() else dict(position=fit['samples'][-1]['world_position'], quaternion=fit['samples'][-1]['world_quaternion'])
    pads, depths = project([world['left_grasp_position'], world['right_grasp_position']], np.asarray(camera_pose['position']), camera_pose['quaternion'], config['camera'])
    desired_pads = np.asarray(reference['pad_centers'])
    pad_errors = pads - desired_pads
    measured = dict(
        foot=mask_features(Image.open(episode / 'camera_right_mask/leg_mask/0.png')),
        sock=mask_features(Image.open(episode / 'camera_right_mask/sock_mask/0.png')))
    comparison = {}
    agreements = {}
    for name, features in measured.items():
        target = reference[name]
        error = np.asarray(features['center']) - target['center']
        width_error = features['width'] / target['width'] - 1
        height_error = features['height'] / target['height'] - 1
        agreements[name + '_position'] = bool(np.all(abs(error) <= [38.4, 28.8]))
        agreements[name + '_width'] = abs(width_error) <= .15
        comparison[name] = dict(reference=target, measured=features, center_error_px=error.tolist(), width_relative_error=width_error, height_relative_error=height_error)
    def angle(points):
        d = np.asarray(points)[1] - np.asarray(points)[0]
        return float(np.degrees(np.arctan2(d[1], d[0])))
    tilt_error = angle(pads) - angle(desired_pads)
    agreements['pad_positions'] = bool(np.all(abs(pad_errors) <= [38.4, 28.8]))
    agreements['pad_tilt'] = abs(tilt_error) <= 3
    span_error = np.linalg.norm(pads[1] - pads[0]) / np.linalg.norm(desired_pads[1] - desired_pads[0]) - 1
    agreements['pad_span'] = bool(abs(span_error) <= .15)
    comparison['pads'] = dict(reference=desired_pads.tolist(), measured=pads.tolist(), errors_px=pad_errors.tolist(), tilt_error_degrees=tilt_error, span_relative_error=float(span_error), camera_depth_m=depths.tolist())
    sync = [json.loads(line) for line in (directory / 'physics-video-sync.jsonl').read_text().splitlines()]
    initial = metadata['rollout_initial_cloth_physics_clock']
    locks = metadata['human_chair_lock_qa_by_frame']
    expected = config['scene']['initial_pose_contract']['locked_pose_baseline']
    delta = np.asarray(expected['human_root_position']) - baseline['scene']['initial_pose_contract']['locked_pose_baseline']['human_root_position']
    checks = dict(
        one_live_inference=metadata['frames'] == metadata['frames_inferred'] == 1 and metadata['frames_replayed'] == 0,
        stop_max_steps=metadata['stop_reason'] == 'max_steps',
        no_reference=metadata['reference_action_projection']['path'] is None and metadata['reference_action_projection']['blend'] == 0,
        same_checkpoint=metadata['checkpoint_sha256'] == json.loads((OLD_EPISODE / 'metadata.json').read_text())['checkpoint_sha256'],
        ten_physics_steps=len(sync) == 10 and all(row['command'] == 1 for row in sync),
        continuous_clock=all(row['clock_before']['fixed_steps'] == initial['fixed_steps'] + i + 1 for i, row in enumerate(sync)),
        recording_read_only=all(row['clock_before'] == row['clock_after'] for row in sync),
        camera_trials_read_only=fit['clock_before'] == fit['clock_after'],
        fixed_camera_for_policy=bool(np.allclose(config['scene']['camera_local_position'], fit['camera_position']) and
                                     np.allclose(config['scene']['camera_local_rotation'], fit['camera_rotation'])),
        right_head_mount=config['scene']['camera_parent_link'] == 'head/see3cam_right/camera_color_frame',
        resolution_fov_unchanged=config['camera'] == baseline['camera'],
        initial_arm_foot_scenario_unchanged=config['scenario'] == baseline['scenario'],
        human_chair_rotation_scale_unchanged=all(config['scene'].get(key) == baseline['scene'].get(key)
            for key in ('human_rotation', 'human_scale', 'chair_rotation', 'chair_scale', 'robot_position', 'robot_rotation')),
        cloth_parameters_unchanged=config['obi']['expected'] == load_config(OLD_EPISODE.parents[2] / 'config.yaml')['obi']['expected'],
        rigid_common_translation=all(np.allclose(np.asarray(expected[key]) - baseline['scene']['initial_pose_contract']['locked_pose_baseline'][key], delta, atol=1e-8) for key in expected),
        actual_positions_match=bool(locks) and all(np.allclose(locks[0][key], expected[key], atol=.002, rtol=0) for key in expected),
        chair_above_floor=expected['chair_position'][1] - .05 > 0,
        camera_above_floor=camera_pose['position'][1] > 0,
        pad_depth_above_near_limit=bool(np.all(depths >= .1)))
    images = [Path(reference['image']), OLD_EPISODE / 'camera_right/0.png', episode / 'camera_right/0.png']
    composite = Image.new('RGB', (1920, 520), 'white'); draw = ImageDraw.Draw(composite)
    for index, (path, label) in enumerate(zip(images, ['Real reference', 'Previous simulation', 'Adjusted simulation: first command'])):
        composite.paste(Image.open(path).convert('RGB').resize((640, 480)), (index * 640, 40))
        draw.text((index * 640 + 10, 12), label, fill='black')
    composite.save(directory / 'comparison.jpg')
    report = dict(recording_valid=all(checks.values()), composition_matches=all(agreements.values()),
        checks=checks, composition_checks=agreements, comparison=comparison,
        translation_world_m=delta.tolist(), camera_local_position=config['scene']['camera_local_position'], camera_local_rotation=config['scene']['camera_local_rotation'],
        actual_camera_pose=camera_pose, physical_steps=len(sync), physical_duration_s=sync[-1]['clock_after']['simulated_time_s'] - initial['simulated_time_s'],
        unmatched=[name for name, passed in agreements.items() if not passed],
        notes=['Reference pad polygons are manually annotated metallic faces; native grasp-frame centers are used as corresponding simulation landmarks.',
               'Real SAM masks and simulation renderer masks have different boundaries around the gripped cuff; raw mask discrepancies are retained.',
               'Camera/robot obstruction and human/chair posture must also be inspected in the saved overview and RGB images.',
               'Sock height is reported as an additional discrepancy; the acceptance criteria separately test mask width and centroid.'])
    review = directory / 'visual_review.json'
    if review.exists():
        report['visual_review'] = json.loads(review.read_text())
    (directory / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.publish:
        if not report['recording_valid']:
            raise RuntimeError('Recording validation failed; do not publish as a completed candidate')
        destination = directory.parent
        shutil.copyfile(episode / 'camera_right/0.png', destination / 'first_inference.png')
        shutil.copyfile(directory / 'composition/before_policy.png', destination / 'before_policy.png')
        shutil.copyfile(directory / 'comparison.jpg', destination / 'comparison.jpg')
        shutil.copyfile(directory / 'comparison.json', destination / 'validation.json')
        shutil.copyfile(directory / 'composition/overview.png', destination / 'overview.png')
        scene = dict(camera_parent_link=config['scene']['camera_parent_link'], camera_local_position=config['scene']['camera_local_position'],
            camera_local_rotation=config['scene']['camera_local_rotation'], human_position=expected['human_root_position'],
            initial_pose_contract=dict(locked_pose_baseline=expected, expected_human_position=expected['human_root_position'], expected_chair_position=expected['chair_position'],
                away_from_robot_m=None, down_m=None, up_m=None, right_from_robot_m=None, translate_to_foot_to_sock=False))
        output = ROOT / 'config/autonomous30_current_ankle_total25deg_physics10_real_composition.yaml'
        output.write_text(yaml.safe_dump(dict(extends=BASE.name, scene=scene), sort_keys=False))
        shutil.copyfile(output, destination / 'final_config.yaml')
        (destination / 'selected.json').write_text(json.dumps(dict(directory=str(directory), episode=str(episode)), indent=2) + '\n')
    print(json.dumps(dict(recording_valid=report['recording_valid'], composition_matches=report['composition_matches'], comparison=comparison, unmatched=report['unmatched']), indent=2))


if __name__ == '__main__':
    main()
