#!/usr/bin/env python3
"""Run at most twelve fresh preparations for the fixed-camera foot-size fit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from adjust_real_foot_size import OUTPUT, ROOT, write_json


def residual(report):
    value = report['measured']
    return np.array([value['width_relative_error'], value['length_relative_error']])


def rank(trial):
    value = trial['report']['measured']
    error = residual(trial['report'])
    # Size is primary; a cropped ankle must not count as a matched foot.
    return (not trial.get('prepared_grasp_unchanged', True),
            value.get('visible_toe_fraction', 1.) < .5,
            not value['ankle_visible'] or value.get('visible_ankle_fraction', 1.) < .5,
            not value['width_length_complete'], float(np.max(np.abs(error))), float(np.linalg.norm(error)),
            float(np.linalg.norm(np.asarray(value['center_error_px']) / [1280., 960.])))


def coupled_translation_step(jacobian, error, origin):
    delta = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
    factor = min(1., .08 / max(np.max(np.abs(delta)), 1e-12))
    lower, upper = np.array([-.15, -.2]), np.array([.15, .2])
    for index, value in enumerate(delta):
        if value > 1e-12:
            factor = min(factor, max(0., (upper[index] - origin[index]) / value))
        elif value < -1e-12:
            factor = min(factor, max(0., (lower[index] - origin[index]) / value))
    # Preserve coupled direction at both the trust limit and pose boundaries.
    return np.round((origin + delta * factor) / .005) * .005


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-candidates', type=int, default=12, choices=range(3, 13))
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    history = []
    proposed = [np.array([0., 0.]), np.array([.02, 0.]), np.array([0., .02])]
    used = set()
    matched = False
    for index in range(args.max_candidates):
        if index < len(proposed):
            point = proposed[index]
        else:
            valid = [trial for trial in history if trial.get('report')]
            if len(valid) < 3:
                raise RuntimeError('Initial probes did not produce three prepared foot measurements')
            # Use measured finite differences, then a local linear fit as more
            # fresh preparations arrive. All proposed poses are quantized to 5mm.
            best = min(valid, key=rank)
            origin = np.array(best['delta_yz_m'])
            nearby = sorted(valid, key=lambda item: np.linalg.norm(np.array(item['delta_yz_m']) - origin))[:6]
            design = np.array([np.r_[np.array(t['delta_yz_m']) - origin, 1.] for t in nearby])
            values = np.array([residual(t['report']) for t in nearby])
            fit = np.linalg.lstsq(design, values, rcond=None)[0]
            point = coupled_translation_step(fit[:2].T, residual(best['report']), origin)
            # If the local proposal was already measured, compare nearby grid
            # candidates and take the lowest predicted size error, preserving
            # an independent direction in the next finite-difference fit.
            if tuple(point.round(6)) in used:
                choices = []
                for step in [.005, .01, .02, .04]:
                    for dy, dz in [(step, 0.), (-step, 0.), (0., step), (0., -step)]:
                        candidate = np.round(np.clip(origin + [dy, dz], [-.15, -.2], [.15, .2]) / .005) * .005
                        if tuple(candidate.round(6)) not in used:
                            predicted = np.r_[candidate - origin, 1.] @ fit
                            choices.append((float(np.max(np.abs(predicted))), candidate))
                if not choices:
                    break
                point = min(choices, key=lambda item: item[0])[1]
        used.add(tuple(point.round(6)))
        name = f'probe{index:02d}'
        directory = OUTPUT / name
        report_path = directory / 'before_policy.json'
        preview_path = directory / 'preview.json'
        log = OUTPUT / f'{name}_controller.log'
        if not (report_path.exists() and preview_path.exists()):
            if directory.exists():
                raise RuntimeError(f'Incomplete existing preparation: {directory}; inspect it before resuming')
            command = [sys.executable, '-u', str(ROOT / 'scripts/adjust_real_foot_size.py'), '--name', name,
                       '--dy', str(float(point[0])), '--dz', str(float(point[1])), '--preview-only']
            print('starting ' + json.dumps(dict(name=name, delta_yz_m=point.tolist(), log=str(log))), flush=True)
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                history.append(dict(name=name, delta_yz_m=point.tolist(), failed=True, log=str(log)))
                write_json(OUTPUT / 'search.json', dict(trials=history, max_candidates=args.max_candidates))
                continue
        if not (report_path.exists() and preview_path.exists()):
            history.append(dict(name=name, delta_yz_m=point.tolist(), failed=True, log=str(log)))
            write_json(OUTPUT / 'search.json', dict(trials=history, max_candidates=args.max_candidates))
            continue
        preview = json.loads(preview_path.read_text())
        if not np.allclose(preview['delta_world_m'][1:], point, atol=1e-8):
            # Preserve completed measurements when selection rules are refined;
            # this reuses measurements, never a simulator/preparation state.
            used.discard(tuple(point.round(6)))
            point = np.array(preview['delta_world_m'][1:])
            used.add(tuple(point.round(6)))
        report = json.loads(report_path.read_text())
        trial = dict(name=name, delta_yz_m=point.tolist(), report=report, log=str(log))
        base_geometry = history[0]['report']['scene_geometry'] if history else report['scene_geometry']
        grasp_drift = max(float(np.max(np.abs(np.array(report['scene_geometry'][key]) - base_geometry[key])))
                          for key in ('left_grasp_position', 'right_grasp_position'))
        trial.update(prepared_grasp_maximum_coordinate_drift_m=grasp_drift,
                     prepared_grasp_unchanged=grasp_drift <= .002)
        history.append(trial)
        write_json(OUTPUT / 'search.json', dict(trials=history, max_candidates=args.max_candidates))
        print('measured ' + json.dumps(dict(name=name, delta_yz_m=point.tolist(), foot=report['measured'])), flush=True)
        value = report['measured']
        if (index >= 2 and trial['prepared_grasp_unchanged'] and value['width_length_complete'] and value.get('visible_toe_fraction', 1.) >= .5
                and value.get('visible_ankle_fraction', 1.) >= .5 and np.all(np.abs(residual(report)) <= .1)):
            matched = True
            break
    valid = [trial for trial in history if trial.get('report')]
    if not valid:
        raise RuntimeError('No successful fresh preparation')
    best = min(valid, key=rank)
    write_json(OUTPUT / 'search.json', dict(trials=history, max_candidates=args.max_candidates, selected=best['name'],
        selected_delta_yz_m=best['delta_yz_m'], size_matched=matched, final_fresh_validation_required=True))
    print('selected ' + json.dumps(dict(name=best['name'], delta_yz_m=best['delta_yz_m'], foot=best['report']['measured'])), flush=True)


if __name__ == '__main__':
    main()
