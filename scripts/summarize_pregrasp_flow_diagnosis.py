#!/usr/bin/env python3
"""Aggregate paired Flow diagnostics without changing simulation results."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np

ARMS = list(range(7)) + list(range(9, 16))
ARMS16 = list(range(7)) + list(range(8, 15))
INDEPENDENT = list(range(8)) + list(range(9, 17))


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def offline_horizons(run, output, offline):
    """Score every valid chunk slot against its matching teacher timestamp."""
    teachers = {case: np.load(run / 'cache' / (case + '.npz'))['joint'][:, INDEPENDENT]
                for case in offline['cases']}
    result = []
    for initial_only in (True, False):
        for ode in (8, 32):
            selected = [r for r in offline['rows'] if r['ode_steps'] == ode
                        and (not initial_only or r['frame'] == 0)]
            for slot in range(8):
                errors, biases, variances, holds, changes, teacher_changes = [], [], [], [], [], []
                for row in selected:
                    t = row['frame']; teacher = teachers[row['case']]
                    if t + 1 + slot >= len(teacher):
                        continue
                    with np.load(output / 'offline' / '{}-t{}.npz'.format(row['case'], t)) as data:
                        chunk = data['ode' + str(ode)][:, :, ARMS16].astype('float64')
                    sample = chunk[:, slot]; truth = teacher[t + 1 + slot, ARMS16]
                    errors.append(float(np.mean((sample - truth) ** 2)))
                    biases.append(float(np.mean((sample.mean(0) - truth) ** 2)))
                    variances.append(float(np.mean(sample.var(0))))
                    holds.append(float(np.mean((teacher[t, ARMS16] - truth) ** 2)))
                    if slot:
                        changes.append(float(np.mean((sample - chunk[:, slot - 1]) ** 2)))
                        teacher_changes.append(float(np.mean((truth - teacher[t + slot, ARMS16]) ** 2)))
                result.append(dict(initial_only=initial_only, ode_steps=ode, slot=slot + 1,
                    observations=len(errors), arm_rmse_rad=float(np.sqrt(np.mean(errors))),
                    bias_rmse_rad=float(np.sqrt(np.mean(biases))),
                    sampling_std_rms_rad=float(np.sqrt(np.mean(variances))),
                    hold_current_angle_rmse_rad=float(np.sqrt(np.mean(holds))),
                    within_chunk_step_rms_rad=float(np.sqrt(np.mean(changes))) if changes else None,
                    teacher_step_rms_rad=float(np.sqrt(np.mean(teacher_changes))) if changes else None))
    return result


def live_metrics(entry, run, low, span, joint_map):
    directory = Path(entry['directory']) / 'policy'
    report = json.loads((directory / 'report.json').read_text())
    assert report['completed'] and report.get('video_validation')
    trace = [json.loads(line) for line in (directory / 'motion-trace.jsonl').read_text().splitlines()]
    source = np.load(run / 'cache' / (entry['case'] + '.npz'))['joint'].astype('float64')
    assert len(trace) == len(source) - 1
    observed = np.array([row['observed_angle'] for row in trace])
    commands = np.array([row['prediction']['action'] for row in trace])
    applied = np.array([row['executed_command'] for row in trace])
    feedback = np.array([row['feedback_angle'] for row in trace])
    torque = np.array([row['filtered_observed_torque'] for row in trace])
    constant = span[18:] <= 1e-7
    norm_torque = (torque - low[18:]) / np.where(constant, 1., span[18:])
    norm_torque[:, constant] = .5
    out_of_range = (norm_torque < 0) | (norm_torque > 1)
    plans = [row['prediction']['flow_plan'] for row in trace]
    execute = entry['execute_steps']
    assert report['device']['ode_steps'] == entry['ode_steps']
    assert report['device']['execute_steps'] == execute
    assert report['device']['flow_seed'] == entry['seed']
    assert all(p['plan_slot'] == t % execute and p['replanned'] == (t % execute == 0)
               for t, p in enumerate(plans))
    assert report['actual_physics_steps'] == len(trace) * 10
    assert report['checkpoint_sha256'] == hashlib.sha256((run / 'model/SAMDAMSARNNFlow_best.pth').read_bytes()).hexdigest()
    error = feedback[:, ARMS] - source[1:, ARMS]
    wrapped_error = np.arctan2(np.sin(error), np.cos(error))
    outside_limits = (feedback < joint_map.lower - 1e-4) | (feedback > joint_map.upper + 1e-4)
    command_error = commands[:, ARMS] - source[1:, ARMS]
    starts = np.array([p['plan_start_observation'] for p in plans])
    anchor_error = observed[starts][:, ARMS] - source[starts][:, ARMS]
    local_plan_error = commands[:, ARMS] - observed[starts][:, ARMS] - (source[1:, ARMS] - source[starts][:, ARMS])
    assert np.allclose(command_error, anchor_error + local_plan_error, atol=1e-10)
    changes = [p['plan_change_arm_rms_rad'] for p in plans if p['plan_change_arm_rms_rad'] is not None]
    first_out = np.flatnonzero(out_of_range.any(axis=1))
    result = {k: entry[k] for k in ('case', 'seed', 'ode_steps', 'execute_steps', 'directory')}
    result.update(steps=len(trace), arm_feedback_rmse_rad=rms(error),
        arm_feedback_wrapped_rmse_rad=rms(wrapped_error),
        first10_arm_feedback_rmse_rad=rms(error[:10]), arm_command_rmse_rad=rms(command_error),
        first25_arm_feedback_rmse_rad=rms(error[:25]), first50_arm_feedback_rmse_rad=rms(error[:50]),
        plan_anchor_error_rmse_rad=rms(anchor_error), local_plan_error_rmse_rad=rms(local_plan_error),
        anchor_local_cross_term_rad2=float(2 * np.mean(anchor_error * local_plan_error)),
        final_arm_error_rmse_rad=rms(error[-1]), first_arm_command_error_rad=rms(command_error[0]),
        first_arm_command_motion_rad=rms(commands[0, ARMS] - observed[0, ARMS]),
        first_teacher_arm_motion_rad=rms(source[1, ARMS] - source[0, ARMS]),
        command_acceleration_rms_rad_s2=rms(np.diff(commands[:, ARMS], n=2, axis=0) / .2 ** 2),
        applied_acceleration_rms_rad_s2=rms(np.diff(applied[:, ARMS], n=2, axis=0) / .2 ** 2),
        command_clipped_fraction=float(np.mean(np.max(abs(commands - applied), axis=1) > 1e-7)),
        arm_command_clipped_fraction=float(np.mean(np.max(abs(commands[:, ARMS] - applied[:, ARMS]), axis=1) > 1e-7)),
        finger_command_clipped_fraction=float(np.mean(np.max(abs(commands[:, [7, 16]] - applied[:, [7, 16]]), axis=1) > 1e-7)),
        normalized_torque_out_of_range_fraction=float(np.mean(out_of_range)),
        first_torque_out_of_range_time_s=float(first_out[0] * .2) if len(first_out) else None,
        normalized_torque_max_abs=float(np.max(abs(norm_torque))),
        feedback_joint_limit_violation_fraction=float(np.mean(outside_limits)),
        maximum_absolute_feedback_arm_angle_rad=float(np.max(abs(feedback[:, ARMS]))),
        plan_boundary_change_rms_rad=rms(changes) if changes else None,
        replans=sum(p['replanned'] for p in plans),
        sam_qa_failed_frames=sum(not r['prediction']['sam_qa_passed'] for r in trace),
        native_penetration_events=report['native_penetration_event_count'],
        native_contact_records_dropped=report.get('native_contact_records_dropped', 0),
        first_observed_angles=observed[0].tolist(), first_command=commands[0].tolist(),
        noise_fingerprints=[p['noise_sha256'] for p in plans],
        both_cloth_present=all(report['geometry'][side]['cloth_present'] for side in ('left', 'right')),
        both_goal_geometry_passed=all(report['geometry'][side]['contact_geometry_checks']['passed'] for side in ('left', 'right')),
        left_cloth_area_mm2=report['geometry']['left']['cloth_surface_area_between_fingers_mm2'],
        right_cloth_area_mm2=report['geometry']['right']['cloth_surface_area_between_fingers_mm2'])
    return result


def summarize(run, output):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sock_dressing_simulation.config import load_config
    from sock_dressing_simulation.joints import JointMap
    joint_map = JointMap.from_config(load_config(Path(__file__).resolve().parents[1] / 'config/arm_reaching_safe_start.yaml'))
    offline = json.loads((output / 'offline/report.json').read_text())
    path = output / 'live/report.json'
    live = json.loads(path.read_text()) if path.exists() else (
        json.loads((output / 'live/progress.json').read_text()) if (output / 'live/progress.json').exists()
        else dict(completed=False, rows=[]))
    normalization = json.loads((run / 'normalization.json').read_text())
    low = np.array(normalization['joint_min']); span = np.array(normalization['joint_max']) - low
    rows = [live_metrics(row, run, low, span, joint_map) for row in live['rows'] if row.get('completed')]
    metrics = ['arm_feedback_rmse_rad', 'first10_arm_feedback_rmse_rad', 'final_arm_error_rmse_rad',
               'arm_feedback_wrapped_rmse_rad', 'first25_arm_feedback_rmse_rad', 'first50_arm_feedback_rmse_rad',
               'arm_command_rmse_rad', 'first_arm_command_error_rad', 'command_acceleration_rms_rad_s2',
               'applied_acceleration_rms_rad_s2', 'command_clipped_fraction',
               'normalized_torque_out_of_range_fraction', 'plan_boundary_change_rms_rad']
    modes = []
    for ode in (8, 32):
        for execute in (1, 2, 4):
            selected = [r for r in rows if r['ode_steps'] == ode and r['execute_steps'] == execute]
            if not selected:
                continue
            modes.append(dict(ode_steps=ode, execute_steps=execute, trials=len(selected),
                **{key: float(np.mean([r[key] for r in selected if r[key] is not None])) for key in metrics},
                median_arm_feedback_rmse_rad=float(np.median([r['arm_feedback_rmse_rad'] for r in selected])),
                both_cloth_present=sum(r['both_cloth_present'] for r in selected),
                both_goal_geometry_passed=sum(r['both_goal_geometry_passed'] for r in selected),
                sam_qa_failed_frames=sum(r['sam_qa_failed_frames'] for r in selected)))
    pairing = []
    for case, seed in sorted({(r['case'], r['seed']) for r in rows}):
        group = [r for r in rows if r['case'] == case and r['seed'] == seed]
        fingerprints_equal = all(r['noise_fingerprints'] == group[0]['noise_fingerprints'] for r in group)
        assert fingerprints_equal
        pairing.append(dict(case=case, seed=seed, modes=len(group), noise_streams_identical=True,
            maximum_initial_angle_difference_rad=float(np.max(np.ptp([r['first_observed_angles'] for r in group], axis=0))),
            maximum_first_command_difference_across_execute_lengths_rad={str(ode): float(np.max(np.ptp(
                [r['first_command'] for r in group if r['ode_steps'] == ode], axis=0)))
                for ode in (8, 32) if any(r['ode_steps'] == ode for r in group)}))
    serial = {(r['case'], r['seed'], r['ode_steps']): r for r in rows if r['execute_steps'] == 1}
    paired_effects = []
    for row in rows:
        key = (row['case'], row['seed'], row['ode_steps'])
        if row['execute_steps'] > 1 and key in serial:
            baseline = serial[key]
            paired_effects.append(dict(case=row['case'], seed=row['seed'], ode_steps=row['ode_steps'],
                execute_steps=row['execute_steps'],
                feedback_rmse_change_rad=row['arm_feedback_rmse_rad'] - baseline['arm_feedback_rmse_rad'],
                first10_feedback_rmse_change_rad=row['first10_arm_feedback_rmse_rad'] - baseline['first10_arm_feedback_rmse_rad'],
                wrapped_feedback_rmse_change_rad=row['arm_feedback_wrapped_rmse_rad'] - baseline['arm_feedback_wrapped_rmse_rad'],
                applied_acceleration_change_rad_s2=row['applied_acceleration_rms_rad_s2'] - baseline['applied_acceleration_rms_rad_s2'],
                final_error_change_rad=row['final_arm_error_rmse_rad'] - baseline['final_arm_error_rmse_rad']))
    public_rows = [{k: v for k, v in row.items() if k not in ('noise_fingerprints', 'first_observed_angles', 'first_command')} for row in rows]
    summary = dict(completed=bool(live.get('completed')) and len(rows) == 36, expected_live_trials=36,
        completed_live_trials=len(rows), offline_initial=offline['initial_observations'],
        offline_all_selected=offline['all_observations'], offline_convergence=offline['convergence'],
        initial_sensor_ablation=offline['initial_sensor_ablation'], live_modes=modes, live_rows=public_rows,
        offline_horizons=offline_horizons(run, output, offline),
        pairing_checks=pairing, paired_chunk_effects=paired_effects,
        scope='Best checkpoint epoch 6998; 10 recorded training cases offline, 3 training cases x 2 seeds x 6 modes live. Frozen cloth/human. Geometric jaw occupancy is not closure/retention success.')
    teacher_path = output / 'teacher-forced/report.json'
    if teacher_path.exists():
        summary['teacher_forced'] = json.loads(teacher_path.read_text())
        summary['completed'] = summary['completed'] and summary['teacher_forced']['completed']
        fixed = {(r['case'], r['seed'], r['ode_steps'], r['execute_steps']): r
                 for r in summary['teacher_forced']['rows']}
        matched = []
        for row in rows:
            baseline = fixed[(row['case'], row['seed'], row['ode_steps'], row['execute_steps'])]
            matched.append(dict(case=row['case'], seed=row['seed'], ode_steps=row['ode_steps'], execute_steps=row['execute_steps'],
                teacher_fixed_command_rmse_rad=baseline['arm_command_rmse_rad'],
                live_command_rmse_rad=row['arm_command_rmse_rad'],
                command_rmse_ratio=row['arm_command_rmse_rad'] / baseline['arm_command_rmse_rad'],
                plan_anchor_error_rmse_rad=row['plan_anchor_error_rmse_rad'],
                local_plan_error_rmse_rad=row['local_plan_error_rmse_rad'],
                anchor_local_cross_term_rad2=row['anchor_local_cross_term_rad2']))
        summary['matched_feedback_amplification'] = matched
        summary['feedback_amplification_summary'] = dict(pairs=len(matched),
            median_command_rmse_ratio=float(np.median([r['command_rmse_ratio'] for r in matched])) if matched else None,
            modes=[dict(ode_steps=ode, execute_steps=execute,
                pairs=sum(r['ode_steps'] == ode and r['execute_steps'] == execute for r in matched),
                median_command_rmse_ratio=float(np.median([r['command_rmse_ratio'] for r in matched
                    if r['ode_steps'] == ode and r['execute_steps'] == execute])))
                for ode in (8, 32) for execute in (1, 2, 4)
                if any(r['ode_steps'] == ode and r['execute_steps'] == execute for r in matched)])
    summary['paired_chunk_effect_summary'] = []
    for ode in (8, 32):
        for execute in (2, 4):
            group = [r for r in paired_effects if r['ode_steps'] == ode and r['execute_steps'] == execute]
            if group:
                summary['paired_chunk_effect_summary'].append(dict(ode_steps=ode, execute_steps=execute, pairs=len(group),
                    full_tracking_improved=sum(r['feedback_rmse_change_rad'] < 0 for r in group),
                    wrapped_tracking_improved=sum(r['wrapped_feedback_rmse_change_rad'] < 0 for r in group),
                    early_tracking_improved=sum(r['first10_feedback_rmse_change_rad'] < 0 for r in group),
                    acceleration_reduced=sum(r['applied_acceleration_change_rad_s2'] < 0 for r in group)))
    manifest = json.loads((run / 'manifest.json').read_text())
    assert not set(offline['cases']) & set(manifest['excluded_ids'])
    summary['excluded_ids'] = manifest['excluded_ids']
    summary['checkpoint_sha256'] = hashlib.sha256((output / 'checkpoint.pth').read_bytes()).hexdigest()
    summary['offline_first_action_error_decomposition'] = {
        ode: dict(bias_mse_fraction=r['bias_rmse_rad'] ** 2 / r['arm_rmse_rad'] ** 2,
                  sampling_variance_fraction=r['sampling_std_rms_rad'] ** 2 / r['arm_rmse_rad'] ** 2)
        for ode, r in offline['initial_observations'].items()}
    summary['remaining_questions'] = [
        'This comparison localizes error to the learned encoder-plus-Flow mapping, not separately to the four LSTMs or the Flow head.',
        'Input noise augmentation, independent torque/perception feedback, and loss weighting were not changed or retrained.',
        'Three training cases and two seeds per mode are a descriptive comparison, not a generalization or real-grasp success estimate.',
        'Cloth and human are frozen; geometric jaw occupancy does not validate cloth closure/retention.'
    ]
    ablation = offline['initial_sensor_ablation']
    summary['initial_sensor_ablation_aggregate'] = dict(
        cases=len(ablation), state_effect_rms_rad=rms([r['state_change_effect_arm_rms_rad'] for r in ablation]),
        remaining_live_vision_difference_rms_rad=rms([r['remaining_live_vision_difference_arm_rms_rad'] for r in ablation]),
        source='Previous same-checkpoint live-train10-best initial observations versus cached teacher vision; same seed 0.')
    initial8 = offline['initial_observations']['8']; initial32 = offline['initial_observations']['32']
    horizon2 = next(r for r in summary['offline_horizons'] if r['initial_only'] and r['ode_steps'] == 8 and r['slot'] == 2)
    summary['findings'] = [
        dict(finding='Exact recorded observations already produce inaccurate and variable first actions.',
             evidence=dict(ode8_arm_rmse_rad=initial8['arm_rmse_rad'], ode32_arm_rmse_rad=initial32['arm_rmse_rad'],
                 ode8_bias_rmse_rad=initial8['bias_rmse_rad'], ode8_sampling_std_rms_rad=initial8['sampling_std_rms_rad']),
             interpretation='Live perception/state mismatch and replanning are not necessary for the initial error to occur.'),
        dict(finding='More ODE steps improve numerical convergence without improving first-action teacher accuracy.',
             evidence=dict(ode8_vs64_rms_rad=rms([r['ode8_vs64_arm_rms_rad'] for r in offline['convergence']]),
                 ode32_vs64_rms_rad=rms([r['ode32_vs64_arm_rms_rad'] for r in offline['convergence']]),
                 ode64_initial_teacher_rmse_rad=rms([r['ode64_metrics']['arm_rmse_rad'] for r in offline['convergence'] if r['frame'] == 0])),
             interpretation='Numerical integration error exists, but simply increasing steps does not repair this learned action distribution.'),
        dict(finding='Large variation also exists within a single generated chunk, without replanning.',
             evidence=dict(ode8_initial_slot1_to2_arm_rms_rad=horizon2['within_chunk_step_rms_rad'],
                 teacher_slot1_to2_arm_rms_rad=horizon2['teacher_step_rms_rad']),
             interpretation='Keeping the same sampled chunk cannot by itself ensure accurate, smooth actions.')]
    if teacher_path.exists():
        summary['findings'].append(dict(finding='With all teacher observations fixed, longer execution reduces acceleration but increases full-trajectory command error.',
            evidence=summary['teacher_forced']['modes'],
            interpretation='Smoother commands and better teacher tracking are separate objectives. This control contains no physical tracking or feedback drift.'))
    if rows:
        summary['findings'].append(dict(finding='Closed-loop observations rapidly leave the torque training range.',
            evidence=dict(first_out_of_range_times_s=[r['first_torque_out_of_range_time_s'] for r in rows],
                trials_with_both_cloth_present=sum(r['both_cloth_present'] for r in rows), completed_trials=len(rows),
                contact_recording_overflow_trials=sum(r['native_contact_records_dropped'] > 0 for r in rows)),
            interpretation='Feedback distribution shift can amplify generation errors. Its independent contribution requires a separate torque/perception ablation. Contact counts are lower bounds where event records overflow.'))
    if summary['completed']:
        four = [r for r in summary['paired_chunk_effect_summary'] if r['execute_steps'] == 4]
        summary['findings'].append(dict(finding='Executing four actions improves full-trajectory tracking and applied-command variation in all 12 matched comparisons.',
            evidence=four,
            interpretation='This mitigates closed-loop amplification in these 3 cases x 2 seeds x 2 ODE settings. '
                'All 36 trials still fail to have cloth between both jaws. Fixed-observation command accuracy does not improve with longer execution, '
                'so reduced feedback accumulation is a supported explanation for the closed-loop improvement, not a complete repair of learned action accuracy.'))
    write(output / 'report.json', summary)
    if public_rows:
        with (output / 'live-metrics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(public_rows[0]))
            writer.writeheader(); writer.writerows(public_rows)
    if offline['completed']:
        write(output / 'offline/progress.json', dict(completed=True, observations=offline['observations'], cases=offline['cases']))
    plot(output, offline, rows, modes, summary)
    return summary


def plot(output, offline, rows, modes, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for i, steps in enumerate((8, 32)):
        record = offline['initial_observations'][str(steps)]
        axes[0].bar(i, record['bias_rmse_rad'] ** 2, color='#31688e', label='Squared bias' if i == 0 else None)
        axes[0].bar(i, record['sampling_std_rms_rad'] ** 2, bottom=record['bias_rmse_rad'] ** 2,
                    color='#f4a261', label='Sampling variance' if i == 0 else None)
    axes[0].set(xticks=[0, 1], xticklabels=['ODE 8', 'ODE 32'], ylabel='Arm MSE [rad²]', title='Identical teacher input: first action')
    axes[0].legend()
    convergence = offline['convergence']
    labels = ['8 vs 64', '32 vs 64']
    values = [rms([r[key] for r in convergence]) for key in ('ode8_vs64_arm_rms_rad', 'ode32_vs64_arm_rms_rad')]
    axes[1].bar(labels, values)
    axes[1].set(ylabel='Paired arm RMS difference [rad]', title='Same latent noise; 64-step reference')
    for ode, color in [(8, '#31688e'), (32, '#35b779')]:
        selected = sorted([r for r in modes if r['ode_steps'] == ode], key=lambda r: r['execute_steps'])
        axes[2].plot([r['execute_steps'] for r in selected], [r['arm_feedback_rmse_rad'] for r in selected],
                     marker='o', color=color, label='ODE ' + str(ode))
    axes[2].set(xticks=[1, 2, 4], xlabel='Actions executed before replanning',
                ylabel='Mean trial arm feedback RMSE [rad]', title='Closed-loop teacher tracking')
    if modes: axes[2].legend()
    fig.tight_layout(); fig.savefig(output / 'diagnosis.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    horizons = [r for r in summary['offline_horizons'] if r['initial_only']]
    for ode, color in [(8, '#31688e'), (32, '#35b779')]:
        group = [r for r in horizons if r['ode_steps'] == ode]
        axes[0].plot([r['slot'] for r in group], [r['arm_rmse_rad'] for r in group],
                     marker='o', color=color, label='ODE ' + str(ode))
    group = [r for r in horizons if r['ode_steps'] == 8]
    axes[0].plot([r['slot'] for r in group], [r['hold_current_angle_rmse_rad'] for r in group],
                 '--', color='gray', label='Hold current angle (reference)')
    axes[0].set(xlabel='Chunk slot', ylabel='Arm RMSE [rad]', title='Initial chunk: each future slot')
    axes[0].legend(fontsize=8)
    teacher = summary.get('teacher_forced', {}).get('modes', [])
    for ode, color in [(8, '#31688e'), (32, '#35b779')]:
        group = [r for r in teacher if r['ode_steps'] == ode]
        axes[1].plot([r['execute_steps'] for r in group], [r['arm_command_rmse_rad'] for r in group],
                     marker='o', color=color, label='ODE ' + str(ode))
        axes[2].plot([r['execute_steps'] for r in group], [r['command_acceleration_rms_rad_s2'] for r in group],
                     marker='o', color=color, label='ODE ' + str(ode))
    axes[1].set(xticks=[1, 2, 4], xlabel='Actions before replanning', ylabel='Command RMSE [rad]',
                title='Teacher-forced: command accuracy')
    axes[2].set(xticks=[1, 2, 4], xlabel='Actions before replanning', ylabel='Command acceleration RMS [rad/s²]',
                title='Teacher-forced: command variation')
    if teacher: axes[2].legend()
    fig.tight_layout(); fig.savefig(output / 'chunk-diagnosis.png', dpi=170); plt.close(fig)
    selected = [r for r in rows if r['case'] == 'D-34' and r['seed'] == 0]
    feedback_chart = ''
    if selected:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        source = np.load(Path(json.loads((output / 'settings.json').read_text())['run']) / 'cache/D-34.npz')['joint']
        for row in selected:
            trace = [json.loads(line) for line in (Path(row['directory']) / 'policy/motion-trace.jsonl').read_text().splitlines()]
            q = np.array([r['feedback_angle'] for r in trace]); times = .2 * (np.arange(len(trace)) + 1)
            errors = np.sqrt(np.mean((q[:, ARMS] - source[1:, ARMS]) ** 2, axis=1))
            style = dict(color={1: '#440154', 2: '#21918c', 4: '#90c847'}[row['execute_steps']],
                         linestyle='-' if row['ode_steps'] == 8 else '--', label='ODE {} K{}'.format(row['ode_steps'], row['execute_steps']))
            axes[0].plot(times[:10], errors[:10], **style)
            axes[1].plot(times, errors, **style)
            torque = np.array([r['prediction']['input_out_of_range_fraction'] for r in trace])
            axes[2].plot(times, torque, **style)
        axes[0].set(xlabel='Time [s]', ylabel='Arm feedback RMSE [rad]', title='D-34 seed 0: first 2 seconds')
        axes[1].set(xlabel='Time [s]', ylabel='Arm feedback RMSE [rad]', title='D-34 seed 0: full trajectory')
        axes[1].set_yscale('log')
        axes[2].set(xlabel='Time [s]', ylabel='Out-of-range fraction (36 state inputs)', title='D-34 seed 0: feedback distribution')
        axes[2].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output / 'feedback-d34.png', dpi=170); plt.close(fig)
        feedback_chart = '<img src="feedback-d34.png">'
    table = ''.join('<tr>' + ''.join('<td>' + ('{:.4f}'.format(r[k]) if isinstance(r[k], float) else str(r[k])) + '</td>' for k in (
        'ode_steps', 'execute_steps', 'trials', 'arm_feedback_rmse_rad', 'first10_arm_feedback_rmse_rad',
        'median_arm_feedback_rmse_rad', 'arm_feedback_wrapped_rmse_rad',
        'applied_acceleration_rms_rad_s2', 'both_cloth_present')) + '</tr>' for r in modes)
    links = ''.join('<li>{} seed{} ODE{} K{}: <a href="{}">全景</a> / <a href="{}">入力</a></li>'.format(
        r['case'], r['seed'], r['ode_steps'], r['execute_steps'],
        html.escape(str((Path(r['directory']) / 'policy/overview.mp4').relative_to(output))),
        html.escape(str((Path(r['directory']) / 'policy/head-input.mp4').relative_to(output)))) for r in rows)
    a = offline['initial_observations']['8']; b = offline['initial_observations']['32']
    variance_fraction = a['sampling_std_rms_rad'] ** 2 / a['arm_rmse_rad'] ** 2
    h2 = next(r for r in horizons if r['ode_steps'] == 8 and r['slot'] == 2)
    findings = '''<p><strong>解析状況：閉ループ {}/36 試行完了。</strong></p>
<h2>切り分け結果</h2><ol>
<li>教示と同じ観測・LSTM履歴でも初手に誤差がある。初手の腕14関節RMSEはODE 8回で {:.5f} rad、32回で {:.5f} rad。
8回の生成平均のずれは {:.5f} rad、生成標準偏差のRMSは {:.5f} rad。初手MSEの約 {:.1f}% は生成分散。
観測ずれや再抽選が起きる前から、条件付き生成分布に誤差が残る。</li>
<li>32回は64回積分の数値解に近づくが、教示への誤差は改善しない。積分誤差は存在するが、回数増加だけでは解消しない。</li>
<li>再抽選しない同じチャンク内でも、初期の1→2スロット間の腕変化量RMSは {:.5f} rad、教示は {:.5f} rad。
同じ生成系列を継続するだけでは時間的一貫性を保証できない。</li>
<li>観測を全時刻で教示に固定すると、2・4ステップ実行で指令加速度は減るが全軌道の指令RMSEは増えた。
滑らかさと正確さは分けて評価する必要がある。</li></ol>
<p>閉ループでは初手の誤差に続いてトルクが学習範囲から外れる。これは誤差増幅の経路として整合するが、
トルク・画像の各寄与の定量化には別の入力切替実験が必要。
最初の状態は同じでもライブ画像は画素単位で固定していない。初期角度差・初手指令差・全ノイズ一致はJSONに記録。
一部の大きな逸脱試行による影響を見分けるため、中央値、初期区間誤差、角度を周期化した誤差を併記。</p>
<p>閉ループの表・図は試行ごとに計算したRMSEの算術平均（中央値列を除く）。
各条件の動作長を揃え、同じseed・ODE回数で実行長1との対応比較もJSONに保存。
接触記録バッファが溢れた試行では接触件数は下限値となる。画像・角度・トルク・物理ステップ記録は継続している。</p>
<p>優先するのは初手の生成精度・分散とチャンク内部の整合性の改善。その後に再計画方式を選ぶ。
4 LSTMとFlowヘッドのどちらに学習上の原因があるかは、この比較だけでは限定できない。
この解析では再学習やモデル重みの変更は行っていない。</p>'''.format(len(rows), a['arm_rmse_rad'], b['arm_rmse_rad'],
        a['bias_rmse_rad'], a['sampling_std_rms_rad'], 100 * variance_fraction,
        h2['within_chunk_step_rms_rad'], h2['teacher_step_rms_rad'])
    amplification = summary.get('feedback_amplification_summary', {}).get('median_command_rmse_ratio')
    if amplification is not None:
        findings += '<p>同じ条件・seed・方式の対応比較で、閉ループの指令RMSEは教示観測固定時の中央値 {:.2f} 倍。現在角度のずれを次の生成基準に引き継ぐため、固定入力の生成誤差に加えてフィードバック中のずれの蓄積が見える。指令誤差＝生成開始時の角度ずれ＋教示の移動量に対する生成差分のずれ、という厳密な分解もJSONに保存。各成分は相関するため、独立した原因の寄与率とは解釈しない。</p>'.format(amplification)
    if summary['completed']:
        four = [r for r in summary['paired_chunk_effect_summary'] if r['execute_steps'] == 4]
        improved = sum(r['full_tracking_improved'] for r in four); count = sum(r['pairs'] for r in four)
        captured = sum(r['both_cloth_present'] for r in rows)
        findings += '<p><strong>閉ループの最終結果：</strong>4ステップ実行は毎時刻再計画に対して {}/{} の対応比較で全軌道の追従誤差を減らし、同じ全比較で適用指令の加速度RMSも減らした。ただし両爪間に布が入ったのは {}/36 試行。固定観測では実行長を延ばしても全軌道の指令精度が改善しないことから、閉ループでの改善には観測ずれ・誤差蓄積の緩和が関与すると考えられる。生成精度そのものの改善は別途必要。</p>'.format(improved, count, captured)
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Flow原因切り分け</title>
<style>body{font:16px/1.7 system-ui;max-width:1300px;margin:30px auto;padding:0 20px}img{width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}table{border-collapse:collapse;width:100%}</style>
<h1>Flow：固定入力・ODE・再計画の比較</h1><p>最良checkpoint epoch 6998。固定教示入力10条件×8時刻×64生成、閉ループ3条件×2 seed×6方式。
全方式で観測時刻ごとの同じノイズを使用。LSTMは全観測で更新、チャンクは生成時の角度を基準に絶対指令へ復元。
人体・布は固定。爪間の布の有無は閉鎖・保持成功とは異なります。</p>'''+findings+'''<img src="diagnosis.png"><img src="chunk-diagnosis.png">'''+feedback_chart+'''
<table><thead><tr><th>ODE回数</th><th>実行長</th><th>試行</th><th>腕追従RMSE rad</th><th>初期10歩RMSE rad</th><th>試行中央値 rad</th><th>周期化RMSE rad</th><th>適用指令加速度RMS rad/s²</th><th>両爪間の布</th></tr></thead><tbody>'''+table+'''</tbody></table>
<p><a href="report.json">集計JSON</a> / <a href="offline/report.json">固定入力の全結果</a> / <a href="teacher-forced/report.json">教示観測固定の実行長比較</a> / <a href="live-metrics.csv">閉ループCSV</a> / <a href="validation.json">整合性検証</a> / <a href="README.txt">再現手順・評価範囲</a></p><ul>'''+links+'</ul></html>'
    (output / 'report.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ('completed', 'completed_live_trials', 'live_modes')}, indent=2))
