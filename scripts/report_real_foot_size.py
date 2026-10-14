#!/usr/bin/env python3
"""Validate and publish a fresh, one-command fixed-camera foot-size candidate."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw
import yaml

from adjust_real_foot_size import BASE, OUTPUT, ROOT, configuration, write_json
from sock_dressing_simulation.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    result = json.loads((directory / 'result.json').read_text())
    episode = Path(result['metadata']).parent
    metadata = json.loads((episode / 'metadata.json').read_text())
    before = json.loads((directory / 'before_policy.json').read_text())
    after = json.loads((directory / 'first_inference.json').read_text())
    config = load_config(directory / 'config.yaml')
    baseline = load_config(BASE)
    original_pose = baseline['scene']['initial_pose_contract']['locked_pose_baseline']
    pose = config['scene']['initial_pose_contract']['locked_pose_baseline']
    delta = np.array(pose['human_root_position']) - original_pose['human_root_position']
    sync = [json.loads(line) for line in (directory / 'physics-video-sync.jsonl').read_text().splitlines()]
    clock = metadata['rollout_initial_cloth_physics_clock']
    old_episode = ROOT / 'artifacts/phase4/real-right-camera-composition-20261013/candidate06'
    old_result = json.loads((old_episode / 'result.json').read_text())
    old_metadata = json.loads(Path(old_result['metadata']).read_text())
    base_capture = json.loads((OUTPUT / 'probe00/before_policy.json').read_text())
    base_foot = base_capture['dressing_qa']['foot_collision_safety']['foot_collider_geometry']
    adjusted_foot = before['dressing_qa']['foot_collision_safety']['foot_collider_geometry']
    camera_keys = ['camera_parent_link', 'camera_local_position', 'camera_local_rotation']
    checks = dict(
        inference_one_replay_zero=metadata['frames'] == metadata['frames_inferred'] == 1 and metadata['frames_replayed'] == 0,
        stop_max_steps=metadata['stop_reason'] == 'max_steps',
        seed_zero=metadata['seed'] == 0,
        same_checkpoint=metadata['checkpoint_sha256'] == old_metadata['checkpoint_sha256'],
        no_reference=metadata['reference_action_projection']['path'] is None and metadata['reference_action_projection']['blend'] == 0,
        camera_mount_unchanged=all(config['scene'][key] == baseline['scene'][key] for key in camera_keys),
        camera_intrinsics_unchanged=config['camera'] == baseline['camera'],
        same_initial_arm_foot_pose=config['scenario'] == baseline['scenario'],
        same_prepared_grasp_positions=all(np.allclose(before['scene_geometry'][key], base_capture['scene_geometry'][key], atol=.002)
            for key in ('left_grasp_position', 'right_grasp_position')),
        foot_shape_rotation_unchanged=all(a['region'] == b['region'] and np.allclose(a['size'], b['size'], atol=1e-5) and
            abs(np.dot(a['rotation'], b['rotation'])) > 1 - 1e-5 for a, b in zip(base_foot, adjusted_foot)) and len(base_foot) == len(adjusted_foot),
        actual_foot_rigid_translation=all(np.allclose(np.array(b['position']) - a['position'], delta, atol=.002) for a, b in zip(base_foot, adjusted_foot)),
        same_cloth_parameters=config['obi']['expected'] == load_config(old_episode / 'config.yaml')['obi']['expected'],
        common_translation=bool(all(np.allclose(np.array(pose[key]) - original_pose[key], delta, atol=1e-8) for key in original_pose)),
        x_unchanged=bool(abs(delta[0]) < 1e-8),
        bounded_y_z=bool(abs(delta[1]) <= .150001 and abs(delta[2]) <= .200001),
        ten_physics_steps=len(sync) == 10 and all(row['command'] == 1 for row in sync),
        continuous_physics_clock=all(row['clock_before']['fixed_steps'] == clock['fixed_steps'] + i + 1 for i, row in enumerate(sync)),
        read_only_recording=all(row['clock_before'] == row['clock_after'] for row in sync),
        read_only_measurements=before['clock_before'] == before['clock_after'] and after['clock_before'] == after['clock_after'],
        camera_world_pose_unchanged=bool(np.allclose(before['camera_world_position'], after['camera_world_position'], atol=1e-5) and
            abs(np.dot(before['camera_world_quaternion'], after['camera_world_quaternion'])) > 1 - 1e-5),
        actual_pose_locked=bool(metadata['human_chair_lock_qa_by_frame']) and all(
            np.allclose(metadata['human_chair_lock_qa_by_frame'][0][key], pose[key], atol=.002, rtol=0) for key in original_pose),
        chair_above_floor=pose['chair_position'][1] - .05 > 0,
        image_matches_episode=bool(np.array_equal(np.asarray(Image.open(directory / 'first_inference.png')),
                                                np.asarray(Image.open(episode / 'camera_right/0.png')))))
    features = after['measured']
    composition_checks = dict(width_within_ten_percent=abs(features['width_relative_error']) <= .1,
                             length_within_ten_percent=abs(features['length_relative_error']) <= .1,
                             full_ankle_to_toe_visible=features['width_length_complete'],
                             toe_visible_in_id_image=features['visible_toe_fraction'] >= .5,
                             ankle_visible_in_id_image=features['visible_ankle_fraction'] >= .5)
    report = dict(recording_valid=all(checks.values()), size_matched=all(composition_checks.values()), checks=checks,
        size_checks=composition_checks, before_policy=before['measured'], first_inference=features,
        reference=after['reference'], delta_from_current_composition_world_m=delta.tolist(),
        physical_steps=len(sync), physical_duration_s=sync[-1]['clock_after']['simulated_time_s'] - clock['simulated_time_s'],
        episode=str(episode), unmatched=[name for name, value in composition_checks.items() if not value],
        notes=['Real ankle boundary is manually annotated; native ankle collider origin and fixed toe define the simulation axis.',
               'Length is ankle-to-distal-outline distance along the axis. Width is the maximum transverse two-pixel slice.',
               'If the ankle is outside the frame or the foot outline hits the border, length is an estimate and size matching cannot pass.',
               'Position and shape differences are reported separately from size. Cloth appearance is outside this adjustment.',
               'Foot pixel area is an additional size indicator; it does not replace the agreed width/length acceptance criteria.',
               'Visible toe/ankle fractions use the occlusion-preserving native ID image; the renderer leg mask itself is amodal.'])
    review = directory / 'visual_review.json'
    if review.exists():
        report['visual_review'] = json.loads(review.read_text())
    write_json(directory / 'validation.json', report)
    composite = Image.new('RGB', (1920, 520), 'white')
    draw = ImageDraw.Draw(composite)
    paths = [OUTPUT / 'reference_annotated.png', OUTPUT / 'previous_annotated.png', directory / 'first_inference_annotated.png']
    for index, (path, label) in enumerate(zip(paths, ['Real: ankle to toe', 'Previous simulation', 'Adjusted: one live command'])):
        composite.paste(Image.open(path).convert('RGB').resize((640, 480)), (index * 640, 40))
        draw.text((index * 640 + 12, 12), label, fill='black')
    composite.save(directory / 'comparison.jpg')
    if args.publish:
        if not report['recording_valid']:
            raise RuntimeError('Fresh validation failed; candidate must not be published')
        for name in ['before_policy.png', 'before_policy_annotated.png', 'first_inference.png',
                     'first_inference_annotated.png', 'first_inference_overview.png', 'comparison.jpg', 'validation.json']:
            shutil.copyfile(directory / name, OUTPUT / name)
        selected = configuration(float(delta[1]), float(delta[2]))
        selected['extends'] = BASE.name
        path = ROOT / 'config/autonomous30_current_ankle_total25deg_physics10_real_foot_size.yaml'
        path.write_text(yaml.safe_dump(selected, sort_keys=False))
        shutil.copyfile(path, OUTPUT / 'final_config.yaml')
        write_json(OUTPUT / 'selected.json', dict(directory=str(directory), episode=str(episode), size_matched=report['size_matched']))
    print(json.dumps(dict(recording_valid=report['recording_valid'], size_matched=report['size_matched'],
                         first_inference=features, unmatched=report['unmatched']), indent=2))


if __name__ == '__main__':
    main()
