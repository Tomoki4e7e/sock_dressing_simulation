#!/usr/bin/env python3
"""Hold observations to recorded trajectories while comparing chunk execution.

This additional control removes feedback-induced differences between policies.
It does not drive a robot, predict physical tracking, or reuse predicted states.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT), str(ROOT.parent / 'ShareSet/share_folder/src')]
from diagnose_pregrasp_flow import OFFLINE_IDS, ARMS18, digest, write


def main(args):
    import train_contact_pregrasp as training
    from models.SAMDAMSARNN_Flow import SAMDAMSARNNFlow
    from sock_dressing_simulation.config import load_config
    from sock_dressing_simulation.joints import JointMap
    started = time.time()
    torch.set_num_threads(2); training.cv2.setNumThreads(1)
    manifest = json.loads((args.run / 'manifest.json').read_text())
    data, low, high = training.load_arrays(manifest['rows'], args.output / 'offline')
    config = json.loads((args.run / 'configuration.json').read_text())
    checkpoint = args.output / 'checkpoint.pth'
    assert digest(checkpoint) == digest(args.run / 'model/SAMDAMSARNNFlow_best.pth')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SAMDAMSARNNFlow(**config['flow_model']).cuda().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    mapping = JointMap.from_config(load_config(ROOT / 'config/arm_reaching_safe_start.yaml'))
    rules = [mapping.mimic[side + '_gripper/mimic_joint'] for side in ('left', 'right')]
    multipliers = [float(rule['multiplier']) for rule in rules]
    offsets = [float(rule['offset']) for rule in rules]
    dest = args.output / 'teacher-forced'; dest.mkdir(exist_ok=True)
    records = []
    with torch.inference_mode():
        for row in data['train']:
            if row['id'] not in OFFLINE_IDS:
                continue
            n = len(row['joint']) - 1
            conditions = []; state = None
            for t in range(n):
                inputs = [row[key][None, t].cuda() for key in ('rgb', 'joint', 'sock', 'foot')]
                observation = model(*inputs, state); state = observation.state
                conditions.append(observation.condition)
            conditions = torch.cat(conditions)
            anchor = row['joint'][:-1, :18].cuda()
            truth = row['reference'][1:, :18]
            for seed in (0, 1):
                generator = torch.Generator(device='cuda').manual_seed(seed)
                # Shape and ordering match one draw per live observation.
                noises = torch.cat([torch.randn((1, 8, 16), device='cuda', generator=generator) for _ in range(n)])
                first_hash = hashlib.sha256(noises[0:1].cpu().numpy().tobytes()).hexdigest()
                for ode in (8, 32):
                    chunks = []
                    for start in range(0, n, 128):
                        end = min(n, start + 128)
                        delta = model.sample_offsets(conditions[start:end], steps=ode, initial_noise=noises[start:end])
                        physical = model.denormalize_actions(model.offsets_to_angles(delta, anchor[start:end]), low[:18], high[:18])
                        chunks.append(model.expand_mimic_joints(physical, multipliers, offsets).cpu().numpy())
                    chunks = np.concatenate(chunks)
                    assert np.isfinite(chunks).all()
                    for execute in (1, 2, 4):
                        t = np.arange(n); starts = t - t % execute
                        commands = chunks[starts, t % execute]
                        # The bound uses the observed teacher q in this control;
                        # this is NOT a claim about physical closed-loop motion.
                        applied = np.array([mapping.bound(command, prior[:18])
                                            for command, prior in zip(commands, row['reference'][:-1])])
                        error = commands[:, ARMS18] - truth[:, ARMS18]
                        replan_times = np.arange(execute, n, execute)
                        switches = chunks[replan_times, 0][:, ARMS18] - chunks[replan_times - execute, execute][:, ARMS18]
                        rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
                        record = dict(case=row['id'], seed=seed, ode_steps=ode, execute_steps=execute, steps=n,
                            arm_command_rmse_rad=rms(error), first10_arm_command_rmse_rad=rms(error[:10]),
                            command_acceleration_rms_rad_s2=rms(np.diff(commands[:, ARMS18], n=2, axis=0) / .2 ** 2),
                            bounded_command_acceleration_rms_rad_s2=rms(np.diff(applied[:, ARMS18], n=2, axis=0) / .2 ** 2),
                            plan_boundary_change_rms_rad=rms(switches), first_noise_sha256=first_hash)
                        records.append(record)
                        np.savez_compressed(dest / '{}-seed{}-ode{}-k{}.npz'.format(row['id'], seed, ode, execute),
                                            command=commands, teacher=truth)
            print('TEACHER_FORCED', row['id'], flush=True)
    keys = ['arm_command_rmse_rad', 'first10_arm_command_rmse_rad', 'command_acceleration_rms_rad_s2',
            'bounded_command_acceleration_rms_rad_s2', 'plan_boundary_change_rms_rad']
    modes = []
    for ode in (8, 32):
        for execute in (1, 2, 4):
            group = [r for r in records if r['ode_steps'] == ode and r['execute_steps'] == execute]
            modes.append(dict(ode_steps=ode, execute_steps=execute, trials=len(group),
                              **{key: float(np.mean([r[key] for r in group])) for key in keys}))
    write(dest / 'report.json', dict(completed=True, cases=OFFLINE_IDS, rows=records, modes=modes,
        checkpoint_sha256=digest(checkpoint), elapsed_s=time.time() - started,
        scope='Recorded teacher observations and recurrent histories are identical for all policies. No physics. '
              'Generate all hypothetical plans batched at 128, then select slots from plan-start anchors. '
              'Each case/seed uses the exact live per-observation latent draw schedule. '
              'Metrics are equal-trial means; command error differs from physical feedback error.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.run = args.run.resolve(); args.output = args.output.resolve()
    main(args)
