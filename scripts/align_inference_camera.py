"""Calibrate the mounted camera on a frozen, freshly prepared scene."""
from pathlib import Path
import json

import numpy as np
from PIL import Image
import yaml

from search_left_camera_toe_alignment import leg_axis_measurement, toe_tip_x_from_mask


def align_camera(environment, directory, output_config, base_config):
    directory = Path(directory) / 'camera_alignment'
    directory.mkdir(exist_ok=True)
    scene = environment.config['scene']
    original = np.asarray(scene['camera_local_rotation'], dtype=float)
    angles = original.copy()
    samples = []

    def clock():
        environment.sock_cloth.request_dressing_qa()
        environment._env.step(simulate=False)
        return dict(environment.sock_cloth.data['dressing_qa']['foot_collision_safety']['cloth_physics_clock'])

    initial_clock = clock()

    def measure(rotation):
        environment.camera.SetTransform(position=list(scene['camera_local_position']),
            rotation=rotation.tolist(), scale=[1., 1., 1.], is_world=False)
        environment._env.step(simulate=False)
        camera = environment._capture_camera(simulate=False)
        mask = camera['leg_mask']
        axis = leg_axis_measurement(mask)
        error = toe_tip_x_from_mask(mask) - (mask.shape[1] - 1) / 2
        sample = dict(rotation=rotation.tolist(), signed_error_px=error, **axis)
        index = len(samples)
        Image.fromarray(camera['rgb']).save(directory / f'{index:02d}.png')
        Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f'{index:02d}_leg.png')
        sample['image'] = str(directory / f'{index:02d}.png')
        samples.append(sample)
        print('camera alignment ' + json.dumps(sample), flush=True)
        return np.array([error, axis['leg_axis_ridge_degrees']])

    residual = measure(angles)
    for _ in range(12):
        if abs(residual[0]) <= 25 and abs(residual[1]) <= 3:
            break
        columns = []
        for axis in (1, 2):
            perturbed = angles.copy()
            perturbed[axis] += .5
            columns.append((measure(perturbed) - residual) / .5)
        correction = np.linalg.lstsq(np.column_stack(columns), -residual, rcond=None)[0]
        angles[[1, 2]] += np.clip(correction, -8, 8)
        angles[1] = np.clip(angles[1], original[1] - 75, original[1] + 75)
        angles[2] = np.clip(angles[2], -60, 60)
        residual = measure(angles)
    final_clock = clock()
    passed = bool(abs(residual[0]) <= 25 and abs(residual[1]) <= 3 and initial_clock == final_clock)
    report = dict(passed=passed, original_rotation=original.tolist(), rotation=angles.tolist(),
        clock_before=initial_clock, clock_after=final_clock, samples=samples,
        center_tolerance_px=25, vertical_tolerance_degrees=3)
    (directory / 'alignment.json').write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise RuntimeError('Camera alignment failed; see camera_alignment/alignment.json')
    scene['camera_local_rotation'] = angles.tolist()
    environment._camera_mount_report['local_rotation'] = angles.tolist()
    environment.config['inference']['camera_alignment_applied'] = True
    output_config = Path(output_config)
    output_config.write_text(yaml.safe_dump(dict(extends=Path(base_config).name,
        scene=dict(camera_local_rotation=angles.tolist())), sort_keys=False))
    return report
