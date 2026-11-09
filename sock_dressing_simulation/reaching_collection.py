"""Synchronized collection on the validated frozen-cloth reaching replay clock."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReachingCollector:
    def __init__(self, path, mapping, plan, source, root, config, head_cameras=False, contact_only=False, image_size=(1280,960)):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / 'camera_right').mkdir()
        self.head_cameras = bool(head_cameras)
        if self.head_cameras:
            (self.path / 'camera_left').mkdir()
            (self.path / 'camera_overview').mkdir()
            self.camera_poses = (self.path / 'camera_poses.jsonl').open('w')
        self.signals = {name: (self.path / (name + '.csv')).open('w', newline='')
                        for name in ('angle', 'torque', 'external_torque')}
        self.writers = {name: csv.writer(stream) for name, stream in self.signals.items()}
        self.frames_stream = (self.path / 'frames.csv').open('w', newline='')
        self.frames_writer = csv.DictWriter(self.frames_stream, fieldnames=[
            'frame', 'step', 'time_s', 'stage', 'stage_time_s', 'cloth_fixed_steps',
            'cloth_time_s', 'inverse_dynamics_step', 'torque_source'])
        self.frames_writer.writeheader()
        self.telemetry = (self.path / 'telemetry_50hz.jsonl').open('w')
        self.frames = 0
        self.steps = 0
        self.initial_clock = None
        self.initial_step = None
        self.torque_sources = set()
        self.closed = False
        self.metadata = dict(
            schema='shareset-reaching-episode-v1', source_route=source.stem,
            trajectory=str(source.relative_to(root)), trajectory_sha256=sha256(source),
            joint_names=list(mapping.names), fixed_step_s=.02, sample_stride=5,
            image_rate_hz=10, image_size=list(image_size), seed=0,
            shared_initial_configuration=(str((source.parent.parent / 'candidate-initial.yaml').relative_to(root))
                                          if contact_only else plan['shared_initial_configuration']),
            shared_initial_sha256=sha256(source.parent.parent / 'candidate-initial.yaml' if contact_only
                                        else root / plan['shared_initial_configuration']),
            angle_units=['rad' if i in mapping.revolute_action_indices else 'm' for i in range(18)],
            effort_units=['N*m' if i in mapping.revolute_action_indices else 'N' for i in range(18)],
            mimic_angle='0.5 * measured finger_joint',
            mimic_effort='duplicate of its independently actuated source DOF, not an independent sensor',
            effort_sign='Unity reduced-coordinate joint axis; no angular unit conversion for effort',
            camera_fov=float(config['camera']['fov']),
            robot_joint_gravity=config.get('scene', {}).get('robot_joint_gravity'),
            effort_scope='Measured motor drive effort in the frozen-cloth, high-stiffness replay; no cloth friction or retention load',
            cloth_frozen=True, dynamic_grasp_validated=False, learning_ready=False,
            reaching_learning_ready=False, completed=False,
            scope='Left-arm reaching and geometric closure with frozen cloth; no retention or pulling validation',
            left_foot_floor_allowance_m=plan.get('left_foot_floor_allowance_m', 0.))
        if contact_only:
            import yaml
            initial = yaml.safe_load((source.parent.parent / 'candidate-initial.yaml').read_text())
            self.metadata.update(condition=plan['condition'], final_closure_performed=False,
                scope='Paired-arm contact pregrasp with frozen cloth, without final closure, retention or pulling',
                endpoint_task='paired_contact_pregrasp', translation_m=plan['translation_m'],
                left_foot_floor_allowance_m=initial.get('left_foot_floor_allowance_m', 0.))
        else:
            self.metadata.update(**{key: plan[key] for key in
                ('dataset_inputs', 'geometry_record', 'workbook_pattern', 'applied_opening_stage')})
        if self.head_cameras:
            self.metadata.update(schema='shareset-reaching-head-stereo-v2',
                camera_source='Dry-AIREC physical head See3CAM right',
                camera_left_source='Dry-AIREC physical head See3CAM left',
                camera_overview_source='World-fixed confirmation camera; excluded from learning',
                camera_profile='head-stereo-zero-offset-v2')
        if contact_only:
            self.metadata['schema'] = 'paired-contact-pregrasp-episode-v1'

    def append(self, env, mapping, record):
        signals = env.robot_signals()
        if not signals['torque_available']:
            raise RuntimeError('Measured joint effort unavailable; zero filling is forbidden')
        if not env.robot.data.get('inverse_dynamics_available'):
            raise RuntimeError('Native inverse dynamics telemetry is unavailable')
        arrays = {key: np.asarray(signals[key], dtype=float) for key in self.writers}
        if any(v.shape != (18,) or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Invalid 18-column signals')
        step = int(record['step']) + 1
        if step != self.steps + 1:
            raise ValueError('Replay telemetry steps are not consecutive')
        if self.initial_step is None:
            self.initial_step = record['cloth_fixed_steps'] - 1
            self.initial_clock = record['cloth_time_s'] - .02
        if record['cloth_fixed_steps'] - self.initial_step != step:
            raise RuntimeError('Physics advanced outside the replay')
        self.steps = step
        self.torque_sources.add(signals['torque_source'])
        row = dict(step=step, time_s=step * .02, stage=record['stage'],
                   stage_time_s=record['stage_time_s'], cloth_time_s=record['cloth_time_s'],
                   cloth_fixed_steps=record['cloth_fixed_steps'],
                   inverse_dynamics_step=env.robot.data.get('inverse_dynamics_step'),
                   torque_source=signals['torque_source'],
                   external_torque_source=signals['external_torque_source'],
                   **{key: value.tolist() for key, value in arrays.items()})
        self.telemetry.write(json.dumps(row) + '\n')
        if step % 5:
            return
        before = int(record['cloth_fixed_steps'])
        telemetry_sample = copy.deepcopy(env.robot.data) if self.frames == 0 else None
        camera = env.recording_camera
        width,height=self.metadata['image_size']
        camera.GetRGB(width, height, self.metadata['camera_fov'])
        if self.head_cameras:
            from .head_cameras import capture_head_cameras
            head_images, head_report = capture_head_cameras(env, self.metadata['camera_fov'],width,height)
            # RGB payloads are one-step responses and disappear on the next
            # exchange. Decode all three views in this same response.
            image = env._decode(camera.data['rgb'], 'RGB')
            self.camera_poses.write(json.dumps(dict(frame=self.frames, step=step, **head_report)) + '\n')
        env.sock_cloth._send_data('GetSockCompoundReplay', env.robot.id)
        env._env.step(simulate=False)
        if int(env.sock_cloth.data['sock_compound_replay']['cloth_fixed_steps']) != before:
            raise RuntimeError('Camera capture advanced physics')
        if not self.head_cameras:image = env._decode(camera.data['rgb'], 'RGB')
        if image.shape != (height, width, 3):
            raise ValueError('Camera dimensions differ from episode metadata')
        if self.head_cameras:
            Image.fromarray(image).save(self.path / 'camera_overview' / f'{self.frames}.png')
            for side, value in head_images.items():
                Image.fromarray(value).save(self.path / ('camera_' + side) / f'{self.frames}.png')
        else:
            Image.fromarray(image).save(self.path / 'camera_right' / f'{self.frames}.png')
        for key, values in arrays.items():
            self.writers[key].writerow(values.tolist())
        self.frames_writer.writerow(dict(frame=self.frames, **{key: row[key] for key in
            ('step', 'time_s', 'stage', 'stage_time_s', 'cloth_fixed_steps', 'cloth_time_s',
             'inverse_dynamics_step', 'torque_source')}))
        if self.frames == 0:
            self.metadata['camera_transform'] = {key: copy.deepcopy(camera.data[key]) for key in
                ('position', 'rotation', 'quaternion', 'local_position', 'local_rotation') if key in camera.data}
            if self.head_cameras:
                self.metadata['overview_camera_transform'] = self.metadata['camera_transform']
                self.metadata['camera_transform'] = head_report['cameras']['right']
                self.metadata['camera_left_transform'] = head_report['cameras']['left']
                self.metadata['stereo_baseline_m'] = head_report['baseline_m']
            (self.path / 'native-telemetry-sample.json').write_text(json.dumps(
                telemetry_sample, default=lambda v: v.tolist(), indent=2) + '\n')
        self.frames += 1

    def finish(self, result):
        self.metadata.update(completed=bool(result['passed']), replay_passed=bool(result['passed']),
                             physics_steps_unchanged=bool(result['final_steps'] - result['initial_steps'] == self.steps))
        if self.metadata.get('endpoint_task') == 'paired_contact_pregrasp':
            self.metadata['contact_policy'] = result.get('policy')

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        for stream in list(self.signals.values()) + [self.frames_stream, self.telemetry]:
            stream.close()
        if self.head_cameras:self.camera_poses.close()
        self.metadata.update(frames=self.frames, physics_steps=self.steps,
                             torque_sources=sorted(self.torque_sources))
        if typ is not None:
            self.metadata.update(completed=False, error=str(value))
        (self.path / 'metadata.json').write_text(json.dumps(self.metadata, indent=2) + '\n')
