#!/usr/bin/env python3
"""Validate autonomous commands, physical timing, limits, and decoded videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.joints import JointMap


def validate(directory, commands=40):
    config = load_config(directory / 'config.yaml')
    mapping = JointMap.from_config(config)
    settings = config['real_speed']
    limits = json.loads((directory / 'speed-limits.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    metadata = json.loads(Path(result['metadata']).read_text())
    motion = [json.loads(line) for line in (directory / 'joint-motion.jsonl').read_text().splitlines()]
    sync_path = directory / 'physics-video-sync.jsonl'
    sync = [json.loads(line) for line in sync_path.read_text().splitlines()] if sync_path.exists() else []
    video = json.loads((directory / 'physics-video.json').read_text())
    if not motion or not sync or video['last_clock'] is None:
        report = dict(ok=False, commands=metadata['frames'], physical_steps=len(motion),
                      stop_reason=metadata['stop_reason'], checks={'complete_recording': False})
        (directory/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
        return report
    dt = settings['fixed_step_s']
    substeps = round(settings['command_interval_s'] / dt)
    expected_steps = commands * substeps
    checks = {}
    checks['command_count'] = metadata['frames'] == metadata['frames_inferred'] == commands
    checks['complete'] = metadata['stop_reason'] == 'max_steps'
    projection = metadata['reference_action_projection']
    checks['autonomous'] = (metadata['policy_source'] == 'SAMDAMSARNNPolicy' and
                            metadata['frames_replayed'] == 0 and projection['path'] is None and
                            projection['blend'] == 0 and projection['cartesian_pull_m'] == 0 and
                            not config['inference']['policy_catch_up']['enabled'])
    checks['physical_steps'] = len(motion) == len(sync) == video['frames'] == expected_steps
    initial = metadata['rollout_initial_cloth_physics_clock']
    final = video['last_clock']
    duration = final['simulated_time_s'] - initial['simulated_time_s']
    checks['physical_duration'] = (final['fixed_steps'] - initial['fixed_steps'] == expected_steps and
                                    abs(duration - commands * settings['command_interval_s']) < 1e-5)
    checks['recording_read_only'] = all(row['clock_before'] == row['clock_after'] for row in sync)
    checks['command_interval'] = all(
        sum(row['command'] == i for row in motion) == substeps for i in range(1, commands + 1))
    checks['clock_continuity'] = all(
        row['clock']['fixed_steps'] == initial['fixed_steps'] + i + 1 and
        row['clock'] == sync[i]['clock_after'] for i, row in enumerate(motion))
    arrays = {key: np.array([row[key] for row in motion]) for key in
              ('policy_target', 'drive_target', 'target_velocity', 'target_acceleration',
               'actual_position', 'actual_velocity', 'displacement_velocity')}
    vmax, amax = np.asarray(limits['velocity_limit']), np.asarray(limits['acceleration_limit'])
    checks['finite_signals'] = all(np.all(np.isfinite(value)) for value in arrays.values())
    checks['target_velocity'] = bool(np.all(np.abs(arrays['target_velocity']) <= vmax + 1e-7))
    checks['target_acceleration'] = bool(np.all(np.abs(arrays['target_acceleration']) <= amax + 1e-6))
    checks['target_derivatives'] = bool(
        np.allclose(np.diff(arrays['drive_target'], axis=0)/dt, arrays['target_velocity'][1:], atol=1e-7) and
        np.allclose(np.diff(arrays['target_velocity'], axis=0)/dt, arrays['target_acceleration'][1:], atol=1e-6))
    checks['actual_velocity'] = bool(np.all(np.abs(arrays['actual_velocity']) <= vmax + 1e-5))
    checks['actual_displacement_velocity'] = bool(np.all(np.abs(arrays['displacement_velocity']) <= vmax + 1e-5))
    checks['joint_bounds'] = bool(np.all(arrays['drive_target'] >= mapping.lower - 1e-8) and
                                 np.all(arrays['drive_target'] <= mapping.upper + 1e-8))
    checks['mimic'] = all(np.allclose(arrays['drive_target'][:, mapping.names.index(name)],
        arrays['drive_target'][:, mapping.names.index(rule['source'])] * rule['multiplier'] + rule['offset'],
        atol=1e-9) for name, rule in mapping.mimic.items())
    stop_corrections = [correction for row in motion for correction in
                        (row.get('prismatic_stop_velocity_guard') or {}).get('corrections', [])]
    allowed_stop_indices = {int(mapping.simulator_indices[i]) for i, name in enumerate(mapping.names)
                            if i not in mapping.revolute_action_indices and name not in mapping.mimic}
    checks['hard_stop_guard_scope'] = all(
        correction['index'] in allowed_stop_indices and correction['constrained_velocity_m_s'] == 0 and
        ((correction['position_m'] <= correction['lower_m'] + 1e-7 and correction['raw_velocity_m_s'] < 0) or
         (correction['position_m'] >= correction['upper_m'] - 1e-7 and correction['raw_velocity_m_s'] > 0))
        for correction in stop_corrections)
    episode = Path(result['metadata']).parent
    predicted = np.loadtxt(episode/'predicted_action.csv', delimiter=',', ndmin=2)
    checks['live_targets'] = bool(predicted.shape == (commands, 18) and all(
        np.allclose(arrays['policy_target'][i*substeps], mapping.bound(predicted[i]), atol=1e-8)
        for i in range(min(commands, len(predicted)))))
    paths = {'side_physics': (directory/'side_physics_50fps.mp4', 50, expected_steps),
             'front': (episode/'demo.mp4', 5, commands),
             'inference': (episode/'inference_camera.mp4', 5, commands)}
    # Use the actual episode filename produced by run_demo.
    if not paths['inference'][0].exists():
        paths['inference'] = (episode/'inference_camera_demo.mp4', 5, commands)
    videos = {}
    for name, (path, fps, frames) in paths.items():
        if not path.exists():
            checks[name+'_video'] = False
            videos[name] = {'path': str(path), 'missing': True}
            continue
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=avg_frame_rate,nb_read_frames,duration,width,height',
            '-of', 'json', str(path)], text=True))['streams'][0]
        numerator, denominator = map(float, probe['avg_frame_rate'].split('/'))
        decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                                capture_output=True, text=True)
        checks[name+'_video'] = bool(int(probe['nb_read_frames']) == frames and
            abs(numerator/denominator - fps) < 1e-6 and abs(float(probe['duration']) - frames/fps) < .021 and
            decode.returncode == 0 and not decode.stderr.strip())
        videos[name] = dict(path=str(path), **probe, decode_error=decode.stderr)
    report = dict(ok=all(checks.values()), checks=checks, commands=metadata['frames'],
        physical_steps=len(motion), simulated_duration_s=duration,
        basis=settings['basis'], task_success=result['result'].get('task_success'),
        maximum_actual_velocity_ratio=float(np.max(np.abs(arrays['actual_velocity']) / vmax)),
        maximum_displacement_velocity_ratio=float(np.max(np.abs(arrays['displacement_velocity']) / vmax)),
        maximum_target_acceleration_ratio=float(np.max(np.abs(arrays['target_acceleration']) / amax)),
        prismatic_stop_correction_count=len(stop_corrections),
        maximum_raw_stop_velocity_m_s=max((abs(c['raw_velocity_m_s']) for c in stop_corrections), default=0),
        checkpoint_sha256=metadata.get('checkpoint_sha256'), videos=videos)
    (directory/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--commands', type=int, default=40)
    args = parser.parse_args()
    if args.commands <= 0:
        parser.error('--commands must be positive')
    report = validate(args.directory, commands=args.commands)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report['ok'] else 1)
