#!/usr/bin/env python3
"""Validate and package a camera-centered autonomous 30-command recording."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.config import load_config
from search_left_camera_toe_alignment import alignment_measurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    directory, baseline = args.directory.resolve(), args.baseline.resolve()
    result = json.loads((directory / 'result.json').read_text())
    episode = Path(result['metadata']).resolve().parent
    metadata = json.loads((episode / 'metadata.json').read_text())
    old = json.loads((baseline / 'metadata.json').read_text())
    config = load_config(directory / 'config.yaml')
    base_config = load_config(baseline.parents[2] / 'config.yaml')
    alignment = json.loads((directory / 'camera_alignment/alignment.json').read_text())
    alignment_source = ROOT / 'scripts/align_inference_camera.py'
    shutil.copyfile(alignment_source, directory / 'sources' / alignment_source.name)
    sync = [json.loads(line) for line in (directory / 'physics-video-sync.jsonl').read_text().splitlines()]
    initial = metadata['rollout_initial_cloth_physics_clock']
    checks = dict(
        camera_alignment=alignment['passed'],
        inferred_30=metadata['frames'] == metadata['frames_inferred'] == 30,
        replayed_zero=metadata['frames_replayed'] == 0 and metadata['replay_source'] is None,
        policy=metadata['policy_source'] == 'SAMDAMSARNNPolicy',
        same_checkpoint=metadata['checkpoint_sha256'] == old['checkpoint_sha256'],
        same_player=metadata['native_player_executable'] == old['native_player_executable'],
        same_physics=config['foot_conforming_rollout'] == base_config['foot_conforming_rollout'] and config['obi']['expected'] == base_config['obi']['expected'],
        stop_max_steps=metadata['stop_reason'] == 'max_steps',
        no_reference=config['inference']['reference_actions'] is None and config['inference']['reference_action_blend'] == 0 and config['inference']['reference_cartesian_pull_m'] == 0,
        physics_steps_300=len(sync) == 300,
        ten_per_command=Counter(row['command'] for row in sync) == Counter({i: 10 for i in range(1, 31)}),
        recording_read_only=all(row['clock_before'] == row['clock_after'] for row in sync),
        clock_continuity=all(row['clock_before']['fixed_steps'] == initial['fixed_steps'] + i + 1 for i, row in enumerate(sync)),
        preparation_camera_read_only=alignment['clock_before'] == alignment['clock_after'],
        ankle_unchanged=config['scenario']['foot'] == base_config['scenario']['foot'],
        scene_unchanged=all(config['scene'][key] == base_config['scene'][key] for key in ('human_position', 'human_rotation', 'initial_pose_contract', 'camera_local_position', 'camera_parent_link')),
    )
    duration = sync[-1]['clock_after']['simulated_time_s'] - initial['simulated_time_s']
    checks['physical_duration_6s'] = abs(duration - 6) < 1e-5
    positions = ('chair_position', 'human_anchor_position', 'human_root_position', 'right_toe_position')
    locks = metadata['human_chair_lock_qa_by_frame']
    checks['recorded_scene_unchanged'] = len(locks) == 30 and all(
        row['valid'] and row['locked'] and all(np.allclose(row[key], old['human_chair_lock_qa_by_frame'][0][key], atol=2e-3, rtol=0) for key in positions)
        for row in locks)
    for name in ('predicted_action', 'applied_action'):
        data = np.loadtxt(episode / f'{name}.csv', delimiter=',', ndmin=2)
        checks[name] = data.shape == (30, 18) and bool(np.isfinite(data).all())
    measurements = []
    reference_leg = np.asarray(Image.open(episode / 'camera_right_mask/leg_mask/0.png').convert('L')) > 0
    reference_ys, _ = np.nonzero(reference_leg)
    toe_band = reference_leg & (np.arange(reference_leg.shape[0])[:, None] >= reference_ys.max() - max(4, round((reference_ys.max() - reference_ys.min() + 1) * .08)) + 1)
    for index in range(30):
        mask = episode / 'camera_right_mask/leg_mask' / f'{index}.png'
        try:
            measurement = dict(frame=index, **alignment_measurement(mask))
            measurement['centered_vertical'] = measurement['absolute_error_px'] <= 25 and abs(measurement['leg_axis_ridge_degrees']) <= 3
            sock = np.asarray(Image.open(episode / 'camera_right_mask/sock_mask' / f'{index}.png').convert('L')) > 0
            occlusion = float(np.count_nonzero(toe_band & sock) / max(1, np.count_nonzero(toe_band)))
            measurement['toe_sock_overlap_fraction'] = occlusion
            measurement['visibility_estimate_source'] = 'initial toe projection overlapping current sock mask; verify on contact sheet'
            measurement['visible_toe_measurement_status'] = 'unmeasurable_due_to_sock_occlusion' if occlusion >= .8 else ('partially_occluded' if occlusion > .05 else 'visible')
            measurement['axis_measurement_status'] = 'shape_axis_valid' if measurement['leg_axis_valid'] else 'low_confidence_shape_axis; inspect image and fixed 3D foot axis'
        except (ValueError, OSError) as error:
            measurement = dict(frame=index, measurement_status='unmeasurable', reason=str(error))
        measurements.append(measurement)
    checks['alignment_measurements_complete'] = len(measurements) == 30 and all('absolute_error_px' in row for row in measurements)
    live = [json.loads(line) for line in (directory / 'live_frame_diagnostics.jsonl').read_text().splitlines()]
    axes = [row['dressing_qa']['foot_axis_to_ankle'] for row in live]
    checks['foot_axis_unchanged'] = len(axes) == 30 and all(np.allclose(axis, axes[0], atol=1e-6, rtol=0) for axis in axes)

    videos = dict(front=episode / 'demo.mp4', inference=episode / 'inference_camera.mp4', side_physics=directory / 'side_physics_50fps.mp4')
    multiview = directory.parent / 'demo_multiview.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(videos['inference']), '-i', str(videos['side_physics']),
        '-filter_complex', '[0:v]scale=640:480,setsar=1[a];[1:v]select=eq(mod(n+1\\,10)\\,0),setpts=N/(5*TB),scale=640:480,setsar=1[b];[a][b]hstack=inputs=2[v]',
        '-map', '[v]', '-r', '5', '-frames:v', '30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(multiview)], check=True)
    videos['multiview'] = multiview
    video_reports = {}
    for name, path in videos.items():
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
            'stream=width,height,avg_frame_rate,nb_read_frames,duration', '-of', 'json', str(path)], capture_output=True, text=True, check=True)
        info = json.loads(probe.stdout)['streams'][0]
        decode = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'], capture_output=True, text=True)
        expected = 300 if name == 'side_physics' else 30
        checks[name + '_video'] = int(info['nb_read_frames']) == expected and abs(float(info['duration']) - 6) < .01 and decode.returncode == 0 and not decode.stderr
        video_reports[name] = dict(path=str(path), **info, decode_errors=decode.stderr)
    canvas = Image.new('RGB', (1280, 520), 'white')
    for index, source in enumerate((baseline / 'prompt_frame.png', episode / 'prompt_frame.png')):
        canvas.paste(Image.open(source).convert('RGB').resize((640, 480)), (640 * index, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), 'Before', fill='black')
    draw.text((652, 12), 'After: camera yaw/roll corrected', fill='black')
    canvas.save(directory.parent / 'camera_before_after.jpg')
    capture = cv2.VideoCapture(str(videos['inference']))
    previews = Image.new('RGB', (640 * 3, 480), 'white')
    for index, frame in enumerate((0, 14, 29)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, rgb = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {frame}')
        previews.paste(Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).resize((640, 480)), (index * 640, 0))
    capture.release()
    previews.save(directory.parent / 'rollout_preview.jpg')
    capture = cv2.VideoCapture(str(videos['inference']))
    contact_sheet = Image.new('RGB', (320 * 6, 260 * 5), 'white')
    draw = ImageDraw.Draw(contact_sheet)
    for frame in range(30):
        ok, rgb = capture.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {frame}')
        x, y = (frame % 6) * 320, (frame // 6) * 260
        contact_sheet.paste(Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)).resize((320, 240)), (x, y + 20))
        draw.text((x + 4, y + 4), f'Step {frame + 1}', fill='black')
    capture.release()
    contact_sheet.save(directory.parent / 'all_steps_preview.jpg')
    report = dict(ok=all(checks.values()), checks=checks, commands=metadata['frames_inferred'], physical_steps=len(sync),
        physical_duration_s=duration, task_success=result['result']['task_success'], camera_alignment=alignment,
        camera_alignment_source_sha256=hashlib.sha256(alignment_source.read_bytes()).hexdigest(),
        baseline_alignment=alignment_measurement(baseline / 'camera_right_mask/leg_mask/0.png'),
        alignment_by_frame=measurements, videos=video_reports,
        multiview_alignment='side frames 10,20,...,300 paired with post-command inference views')
    (directory.parent / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(ok=report['ok'], checks=checks, task_success=report['task_success']), indent=2))
    if not report['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
