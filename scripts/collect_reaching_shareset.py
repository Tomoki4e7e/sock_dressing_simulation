#!/usr/bin/env python3
"""Collect only the adopted 12 routes, retaining native geometric validation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from sock_dressing_simulation.reaching_collection import sha256
from sock_dressing_simulation.reaching_dataset import verify_dataset_inputs
from replay_diverse_left_reaching import compress_intermediates
from report_photo_grasp_adjustment import actual_recheck


def split_for(number):
    return 'validation' if number == 20 else 'test' if number == 50 else 'train'


def check_raw(episode):
    import numpy as np
    metadata = json.loads((episode / 'metadata.json').read_text())
    frames = list(csv.DictReader((episode / 'frames.csv').open()))
    count = len(frames)
    checks = dict(completed=metadata['completed'], replay=metadata.get('replay_passed', False),
                  count=count == metadata['frames'] and count > 29,
                  physics_steps_unchanged=metadata.get('physics_steps_unchanged', False),
                  stages={'reach', 'settle', 'closure', 'closed_settle'} <= {r['stage'] for r in frames},
                  frame_numbers=[int(r['frame']) for r in frames] == list(range(count)),
                  steps=[int(r['step']) for r in frames] == list(range(5, 5 * count + 1, 5)),
                  image_ids={p.stem for p in (episode / 'camera_right').glob('*.png')} == {str(i) for i in range(count)})
    for signal in ('angle', 'torque', 'external_torque'):
        data = np.loadtxt(episode / (signal + '.csv'), delimiter=',', ndmin=2)
        checks[signal] = data.shape == (count, 18) and bool(np.isfinite(data).all())
        if signal == 'torque':checks['nonzero_effort'] = bool(np.any(data))
    rows = [json.loads(s) for s in (episode / 'telemetry_50hz.jsonl').read_text().splitlines()]
    checks['telemetry_count'] = len(rows) == metadata['physics_steps']
    checks['telemetry_steps'] = [r['step'] for r in rows] == list(range(1, len(rows) + 1))
    checks['physics_clock'] = all(r['cloth_fixed_steps'] - rows[0]['cloth_fixed_steps'] == i for i, r in enumerate(rows))
    for name in ('angle', 'torque', 'external_torque'):
        checks[name + '_alignment'] = bool(np.array_equal(
            np.loadtxt(episode / (name + '.csv'), delimiter=',', ndmin=2),
            np.array([r[name] for r in rows[4::5]])))
    checks['effort_clock'] = all(r['inverse_dynamics_step'] - rows[0]['inverse_dynamics_step'] == i for i, r in enumerate(rows))
    if metadata.get('camera_profile') == 'head-stereo-zero-offset-v2':
        for folder in ('camera_left', 'camera_overview'):
            checks[folder + '_ids'] = {p.stem for p in (episode / folder).glob('*.png')} == {str(i) for i in range(count)}
        poses = [json.loads(s) for s in (episode / 'camera_poses.jsonl').read_text().splitlines()]
        checks['stereo_pose_count'] = len(poses) == count
        checks['stereo_synchronized'] = all(p['step'] == int(f['step']) and p['frame'] == int(f['frame']) for p, f in zip(poses, frames))
        checks['stereo_baseline'] = all(abs(p['baseline_m'] - .084) < 1e-4 for p in poses)
    report = dict(passed=all(checks.values()), checks=checks, frames=count,
                  duration_s=len(rows) * .02)
    (episode / 'raw-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/reaching-shareset-v1')
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--python', default='/tmp/sock-diversity-venv/bin/python')
    ap.add_argument('--head-cameras', action='store_true')
    args = ap.parse_args()
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    adopted_path = ROOT / 'artifacts/diverse-left-reaching/corrected-collection/manifest.json'
    adopted = json.loads(adopted_path.read_text())
    if adopted['count'] != 12 or adopted['stage_counts'] != {'A': 12, 'B': 0}:
        raise ValueError('The adopted route set changed')
    if sha256(ROOT / adopted['shared_initial_configuration']) != adopted['shared_initial_sha256']:
        raise ValueError('Shared initial configuration changed')
    dataset = 'data_sock_reaching_v1'
    work = [(i, row) for i, row in enumerate(adopted['rows']) if not args.ids or row['id'] in args.ids]
    if args.ids and {r['id'] for _, r in work} != set(args.ids):raise ValueError('Unknown route ID')

    def collect(item):
        i, row = item
        source = ROOT / row['trajectory'];plan = json.loads(source.read_text());verify_dataset_inputs(plan, ROOT)
        if sha256(ROOT / row['angle_file']) != row['angle_sha256']:raise ValueError('Adopted feedback hash changed')
        episode = out / 'data' / dataset / split_for(row['number']) / row['id']
        pilot = out / 'pilot' / row['id'].replace('-trajectory', '')
        destination = pilot if pilot.exists() else out / 'native' / row['id']
        destination.mkdir(parents=True, exist_ok=True)
        if not (episode / 'metadata.json').exists():
            if episode.exists():raise RuntimeError(f'Incomplete episode requires inspection: {episode}')
            if shutil.disk_usage(out).free < 1_000_000_000:raise RuntimeError('Less than 1 GB free; collection stopped')
            env = dict(os.environ)
            env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       MPLCONFIGDIR='/tmp/sock-collection-mpl', PYTHONUNBUFFERED='1',
                       __GLX_VENDOR_LIBRARY_NAME='mesa', LIBGL_ALWAYS_SOFTWARE='1',
                       SOCK_REGISTERED_BATCHMODE='1', SOCK_REGISTERED_PLAYER_LOG=str(destination / 'player.log'),
                       PYTHONPATH=str(ROOT / '.deps/RCareWorld/pyrcareworld') + ':/opt/ros/noetic/lib/python3/dist-packages')
            cmd = [args.python, str(ROOT / 'scripts/capture_fixed_torso_initial.py'),
                   '--trajectory', str(source), '--output', str(destination), '--port', str(17500 + i * 10),
                   '--replay', '--collect-episode', str(episode)]
            if args.head_cameras:cmd.append('--head-cameras')
            print('COLLECT', row['id'], flush=True)
            with (destination / 'replay.log').open('w') as log:
                subprocess.run(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        native = json.loads((destination / 'native-route-initial/replay-result.json').read_text())
        if not native['passed'] or native['source_sha256'] != sha256(source):raise ValueError('Native replay gate failed or stale')
        initial = json.loads((destination / 'native-route-initial/result.json').read_text())
        if not initial['passed'] or not initial['camera_transform_unchanged_during_motion']:
            raise ValueError('Initial-state or fixed-camera gate failed')
        closed_path = destination / 'actual-closed-grasp-recheck.json'
        if not closed_path.exists():actual_recheck(plan, native, destination)
        if not json.loads(closed_path.read_text())['passed']:raise ValueError('Actual closed grasp geometry failed')
        metadata = json.loads((episode / 'metadata.json').read_text())
        if args.head_cameras and metadata.get('camera_profile') != 'head-stereo-zero-offset-v2':
            raise ValueError('Existing episode belongs to the overview-camera profile')
        if metadata['trajectory_sha256'] != sha256(source):raise ValueError('Episode trajectory hash changed')
        report = check_raw(episode)
        compress_intermediates(destination)
        if not report['passed']:raise ValueError(f'Raw QA failed: {row["id"]}: {report}')
        metadata.update(raw_validation_passed=True, native_replay=str(destination.relative_to(ROOT)),
                        actual_closed_capture_passed=True, split=split_for(row['number']))
        (episode / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('RAW PASSED', row['id'], report['frames'], flush=True)
        return dict(id=row['id'], number=row['number'], split=split_for(row['number']),
                    path=str(episode), frames=report['frames'], trajectory_sha256=sha256(source))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(collect, work))
    (out / ('pilot-collection.json' if args.ids else 'collection.json')).write_text(json.dumps(
        dict(source_manifest=str(adopted_path), source_sha256=sha256(adopted_path), rows=rows), indent=2) + '\n')
    if not args.ids:
        import yaml
        manifest = {dataset: {split: {r['id']: dict(start=0, end=r['frames']) for r in rows if r['split'] == split}
                              for split in ('train', 'validation', 'test')}}
        (out / 'data/dataset_reaching.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
    print('COLLECTION COMPLETE', len(rows), flush=True)


if __name__ == '__main__':main()
