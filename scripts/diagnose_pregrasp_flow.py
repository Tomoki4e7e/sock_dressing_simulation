#!/usr/bin/env python3
"""Paired teacher-input sampling and native closed-loop Flow ablations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_IDS = ['A-02', 'A-13', 'B-08', 'B-22', 'C-17', 'C-28', 'D-02', 'D-34', 'E-07', 'E-29']
LIVE_IDS = ['D-34', 'A-02', 'C-28']
ARMS16 = list(range(7)) + list(range(8, 15))
ARMS18 = list(range(7)) + list(range(9, 16))


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def generation_metrics(samples, truth):
    """Bias/variance use population variance, so total MSE=bias²+variance."""
    values = samples[:, ARMS16].astype('float64')
    reference = truth[ARMS16]
    mean = values.mean(0)
    bias = float(np.mean((mean - reference) ** 2))
    variance = float(np.mean(np.var(values, axis=0)))
    total = float(np.mean((values - reference) ** 2))
    assert np.isclose(total, bias + variance)
    errors = np.sqrt(np.mean((values - reference) ** 2, axis=1))
    return dict(arm_rmse_rad=float(np.sqrt(total)), bias_rmse_rad=float(np.sqrt(bias)),
                sampling_std_rms_rad=float(np.sqrt(variance)), bias_mse_fraction=bias / max(total, 1e-30),
                median_sample_arm_rmse_rad=float(np.median(errors)),
                p95_sample_arm_rmse_rad=float(np.quantile(errors, .95)),
                finger_rmse_m=float(np.sqrt(np.mean((samples[:, [7, 15]] - truth[[7, 15]]) ** 2))))


def aggregate_generation(records):
    result = {}
    for steps in (8, 32):
        selected = [r for r in records if r['ode_steps'] == steps]
        result[str(steps)] = dict(observations=len(selected),
            arm_rmse_rad=float(np.sqrt(np.mean([r['arm_rmse_rad'] ** 2 for r in selected]))),
            bias_rmse_rad=float(np.sqrt(np.mean([r['bias_rmse_rad'] ** 2 for r in selected]))),
            sampling_std_rms_rad=float(np.sqrt(np.mean([r['sampling_std_rms_rad'] ** 2 for r in selected]))))
    return result


def offline(args):
    import torch
    sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT), str(ROOT.parent / 'ShareSet/share_folder/src')]
    import train_contact_pregrasp as training
    from models.SAMDAMSARNN_Flow import SAMDAMSARNNFlow
    torch.set_num_threads(2)
    training.cv2.setNumThreads(1)
    out = args.output / 'offline'
    out.mkdir(exist_ok=True)
    if not (out / 'cache').exists():
        # The evaluation SSD does not support symbolic links.
        shutil.copytree(args.run / 'cache', out / 'cache')
    manifest = json.loads((args.run / 'manifest.json').read_text())
    rows = [r for r in manifest['rows'] if r['id'] in OFFLINE_IDS]
    assert len(rows) == len(OFFLINE_IDS) and all(r['split'] == 'train' for r in rows)
    # Fit ranges from the complete original train split, never from this subset.
    datasets, low, high = training.load_arrays(manifest['rows'], out)
    assert json.loads((out / 'normalization.json').read_text()) == json.loads((args.run / 'normalization.json').read_text())
    config = json.loads((args.run / 'configuration.json').read_text())
    saved = torch.load(args.output / 'checkpoint.pth', map_location='cpu', weights_only=False)
    model = SAMDAMSARNNFlow(**config['flow_model']).cuda().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    index = model.action_indices.cpu().numpy()
    records, convergence, sensor_ablation = [], [], []
    started = time.time()
    with torch.inference_mode():
        for row in datasets['train']:
            if row['id'] not in OFFLINE_IDS:
                continue
            frames = len(row['joint'])
            selected = sorted({0, 1, 4, 9, 25, (frames - 2) // 2, max(0, frames - 9), frames - 2})
            state = None
            for t in range(max(selected) + 1):
                tensors = [row[key][None, t].cuda() for key in ('rgb', 'joint', 'sock', 'foot')]
                observation = model(*tensors, state)
                state = observation.state
                if t not in selected:
                    continue
                state_before = [v.clone() for branch in state for v in branch]
                noises = torch.cat([torch.randn((1, 8, 16), device='cuda',
                    generator=torch.Generator(device='cuda').manual_seed(seed + 137 * t))
                    for seed in range(args.samples)])
                condition = observation.condition.expand(args.samples, -1).contiguous()
                anchor = tensors[1][:, :18].expand(args.samples, -1)
                predictions = {}
                truth = row['reference'][t + 1, index]
                for steps in (8, 32):
                    offsets = model.sample_offsets(condition, steps=steps, initial_noise=noises)
                    norm = model.offsets_to_angles(offsets, anchor)
                    physical = model.denormalize_actions(norm, low[:18], high[:18]).cpu().numpy()
                    assert np.isfinite(physical).all()
                    predictions[steps] = physical
                    records.append(dict(case=row['id'], frame=t, time_s=.2 * t, ode_steps=steps,
                        samples=args.samples, teacher_next_arm_motion_rms_rad=float(np.sqrt(np.mean(
                        (row['reference'][t + 1, index][ARMS16] - row['reference'][t, index][ARMS16]) ** 2))),
                        **generation_metrics(physical[:, 0], truth)))
                np.savez_compressed(out / '{}-t{}.npz'.format(row['id'], t),
                    ode8=predictions[8], ode32=predictions[32], reference_next=truth)
                if t in (0, (frames - 2) // 2):
                    offsets = model.sample_offsets(condition, steps=64, initial_noise=noises)
                    p64 = model.denormalize_actions(model.offsets_to_angles(offsets, anchor), low[:18], high[:18]).cpu().numpy()
                    convergence.append(dict(case=row['id'], frame=t,
                        ode8_vs64_arm_rms_rad=float(np.sqrt(np.mean((predictions[8][:, 0, ARMS16] - p64[:, 0, ARMS16]) ** 2))),
                        ode32_vs64_arm_rms_rad=float(np.sqrt(np.mean((predictions[32][:, 0, ARMS16] - p64[:, 0, ARMS16]) ** 2))),
                        ode8_vs32_arm_rms_rad=float(np.sqrt(np.mean((predictions[8][:, 0, ARMS16] - predictions[32][:, 0, ARMS16]) ** 2))),
                        ode64_metrics=generation_metrics(p64[:, 0], truth)))
                if t == 0:
                    live = args.run / 'live-train10-best/SAMDAMSARNNFlow_best' / row['id'] / 'policy/motion-trace.jsonl'
                    if live.exists():
                        first = json.loads(live.open().readline())
                        stats = json.loads((args.run / 'normalization.json').read_text())
                        minimum = np.array(stats['joint_min']); span = np.array(stats['joint_max']) - minimum
                        measured = np.r_[first['observed_angle'], first['filtered_observed_torque']]
                        normalized = ((measured - minimum) / span).astype('float32')
                        altered = model(tensors[0], torch.from_numpy(normalized[None]).cuda(), tensors[2], tensors[3], None)
                        offsets = model.sample_offsets(altered.condition, steps=8, initial_noise=noises[:1])
                        physical = model.denormalize_actions(model.offsets_to_angles(offsets,
                            torch.from_numpy(normalized[None, :18]).cuda()), low[:18], high[:18]).cpu().numpy()[0, 0]
                        original_live = np.array(first['prediction']['action'])[index]
                        sensor_ablation.append(dict(case=row['id'],
                            cached_seed0_arm_error_rad=float(np.sqrt(np.mean((predictions[8][0, 0, ARMS16] - truth[ARMS16]) ** 2))),
                            live_state_cached_vision_arm_error_rad=float(np.sqrt(np.mean((physical[ARMS16] - truth[ARMS16]) ** 2))),
                            saved_live_arm_error_rad=float(np.sqrt(np.mean((original_live[ARMS16] - truth[ARMS16]) ** 2))),
                            state_change_effect_arm_rms_rad=float(np.sqrt(np.mean((physical[ARMS16] - predictions[8][0, 0, ARMS16]) ** 2))),
                            remaining_live_vision_difference_arm_rms_rad=float(np.sqrt(np.mean((original_live[ARMS16] - physical[ARMS16]) ** 2)))))
                assert all(torch.equal(before, after) for before, after in zip(state_before, [v for branch in state for v in branch]))
                write(out / 'progress.json', dict(completed=False, case=row['id'], frame=t, observations=len(records) // 2))
            print('OFFLINE_CASE', row['id'], flush=True)
    report = dict(completed=True, checkpoint_epoch=saved['epoch'], samples=args.samples, precision='float32',
        cases=OFFLINE_IDS, observations=len(records) // 2, state_frozen_during_sampling=True,
        paired_noise='seed+137*teacher_frame; identical initial noise for 8/32/64',
        all_observations=aggregate_generation(records), initial_observations=aggregate_generation([r for r in records if r['frame'] == 0]),
        rows=records, convergence=convergence, initial_sensor_ablation=sensor_ablation,
        elapsed_s=time.time() - started, scope='Recorded teacher inputs; no simulator movement; 64 steps is a numerical reference, not ground truth')
    write(out / 'report.json', report)
    print('OFFLINE_COMPLETE', json.dumps(report['initial_observations']), flush=True)


def live(args):
    from run_contact_pregrasp_policy_tests import run_case
    out = args.output / 'live'
    out.mkdir(exist_ok=True)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    cases = {r['id']: r for r in manifest['rows'] if r['id'] in LIVE_IDS}
    assert len(cases) == len(LIVE_IDS) and all(r['split'] == 'train' for r in cases.values())
    checkpoint = args.output / 'checkpoint.pth'
    checksum = digest(checkpoint)
    jobs = [(cases[case], seed, ode, execute) for case in LIVE_IDS for seed in (0, 1)
            for ode in (8, 32) for execute in (1, 2, 4)]
    results = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for index, (case, seed, ode, execute) in enumerate(jobs):
            mode = 'ode{}_execute{}_seed{}'.format(ode, execute, seed)
            options = dict(ode_steps=ode, execute_steps=execute, flow_seed=seed)
            job = pool.submit(run_case, case, args.run, out / mode, index, checkpoint, checksum,
                              args.port_base, options)
            pending[job] = dict(case=case['id'], seed=seed, ode_steps=ode, execute_steps=execute,
                                directory=str(out / mode / case['id']))
        for job in as_completed(pending):
            entry = pending[job]
            try:
                report = job.result()
                entry.update(completed=report.get('completed', False), failure=report.get('failure'),
                             report=str(Path(entry['directory']) / 'policy/report.json'))
            except Exception as error:
                entry.update(completed=False, failure=repr(error))
            results.append(entry)
            write(out / 'progress.json', dict(completed=False, expected=len(jobs), finished=len(results),
                                               elapsed_s=time.time() - started, rows=results))
            print('LIVE_FINISHED', len(results), '/', len(jobs), json.dumps(entry), flush=True)
    summary = dict(completed=all(r['completed'] for r in results), expected=len(jobs), finished=len(results),
                   elapsed_s=time.time() - started, rows=results)
    write(out / 'report.json', summary)
    write(out / 'progress.json', summary)
    assert digest(checkpoint) == checksum
    if not summary['completed']:
        raise RuntimeError('Some live diagnostic trials failed; inspect live/report.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--port-base', type=int, default=54100)
    parser.add_argument('--phase', choices=['all', 'offline', 'live'], default='all')
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / 'checkpoint.pth'
    if not checkpoint.exists():
        shutil.copy2(args.run / 'model/SAMDAMSARNNFlow_best.pth', checkpoint)
    assert digest(checkpoint) == digest(args.run / 'model/SAMDAMSARNNFlow_best.pth')
    if args.phase == 'all':
        write(args.output / 'settings.json', dict(run=str(args.run), checkpoint_sha256=digest(checkpoint),
            offline_ids=OFFLINE_IDS, live_ids=LIVE_IDS, samples=args.samples, workers=args.workers,
            ode_steps=[8, 32], execute_steps=[1, 2, 4], live_seeds=[0, 1],
            noise_control='Advance one Gaussian draw per observation, including held-plan steps',
            lstm_state='Update on every live observation; frozen during repeated offline sampling'))
        for phase in ('offline', 'live'):
            command = [sys.executable, '-u', str(Path(__file__)), '--run', str(args.run), '--output', str(args.output),
                       '--samples', str(args.samples), '--workers', str(args.workers), '--port-base', str(args.port_base), '--phase', phase]
            with (args.output / (phase + '.log')).open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        write(args.output / 'execution-status.json', dict(completed=True))
    elif args.phase == 'offline':
        offline(args)
    else:
        live(args)


if __name__ == '__main__':
    main()
