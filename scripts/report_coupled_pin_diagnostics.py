#!/usr/bin/env python3
"""Summarize immutable/partial pin diagnostics without qualifying a trial."""
import argparse
import csv
import json
from pathlib import Path


def rows(path):
    result = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return result


def summarize(row):
    safety = row['dressing_qa']['foot_collision_safety']
    classes = row['stretch']['edge_classes']
    normalized = [v['maximum_stretch'] / (2 if k == 'opening_rim' else
        1.1 if k == 'opening_body' else 1.5)
        for k, v in classes.items() if v.get('maximum_stretch') is not None]
    states = row['grasp_state']
    return {'command': row['frame'] + 1,
        # Runtime anchor-error diagnostics cover attached pins only. An empty
        # attachment must not appear as a successful zero-error measurement.
        'anchor_error_mm': safety['maximum_original_grasp_anchor_error_m'] * 1000 if any(s['attached'] for s in states) else None,
        'peak_slip_error_mm': max((s['peak_constraint_error'] * 1000 for s in states), default=0),
        'attached_sides': sum(s['attached'] for s in states),
        'body_stretch': classes['body_body']['maximum_stretch'],
        'opening_body_stretch': classes.get('opening_body', {}).get('maximum_stretch'),
        'opening_rim_stretch': classes.get('opening_rim', {}).get('maximum_stretch'),
        'maximum_normalized_stretch': max(normalized),
        'geometric_penetration_mm': row['dressing_qa']['geometric_maximum_cloth_foot_penetration_m'] * 1000,
        'material_path_clips': safety['material_path_clip_count'],
        'stage_diagnostics': safety.get('grasp_stage_diagnostics', {}),
        'pin_correction_diagnostics': safety.get('pin_correction_diagnostics', {}),
        'final_contact_projected_pin_residual_mm': (
            safety['final_contact_projected_pin_residual_m'] * 1000
            if 'final_contact_projected_pin_residual_m' in safety else None),
        'material_constraint_residual_fraction': safety.get('material_constraint_residual_fraction'),
        'release_reasons': [s['release_reason'] for s in states if s['release_reason']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-label', default='V31')
    parser.add_argument('--repaired', type=Path, required=True)
    parser.add_argument('--repaired-label', default='V32')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    groups = {name: [summarize(row) for row in rows(path)]
        for name, path in [(args.baseline_label, args.baseline), (args.repaired_label, args.repaired)]}
    (args.output / 'command-comparison.json').write_text(json.dumps({
        'scope': 'recorded frames only; stage samples retain greatest error increase, not full histories',
        'rows': groups}, indent=2) + '\n')
    fields = ['version', 'command', 'anchor_error_mm', 'peak_slip_error_mm',
        'attached_sides', 'body_stretch', 'opening_body_stretch', 'opening_rim_stretch',
        'maximum_normalized_stretch', 'final_contact_projected_pin_residual_mm',
        'material_constraint_residual_fraction',
        'geometric_penetration_mm', 'material_path_clips']
    with (args.output / 'command-comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader()
        for version, values in groups.items():
            writer.writerows(dict(version=version, **value) for value in values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for version, values in groups.items():
        x = [r['command'] for r in values]
        axes[0].plot(x, [r['anchor_error_mm'] for r in values], label=version)
        axes[1].plot(x, [r['maximum_normalized_stretch'] for r in values], label=version)
        released=next((r['command'] for r in values if r['attached_sides']<2),None)
        if released is not None:
            axes[0].axvline(released, linestyle=':', alpha=.6, label=version+' grasp release')
    axes[0].axhline(200, color='black', linestyle='--', label='Original slip threshold')
    axes[0].set_ylabel('Attached pin error (mm)'); axes[0].legend()
    axes[1].axhline(1.01, color='black', linestyle='--', label='Material limit + 1% tolerance')
    axes[1].set_yscale('log'); axes[1].set_ylabel('Stretch / original limit')
    axes[1].set_xlabel('Recorded command number'); axes[1].legend()
    for axis in axes: axis.grid(alpha=.25)
    figure.tight_layout(); figure.savefig(args.output / 'command-comparison.png', dpi=160)
    print(json.dumps({k: len(v) for k, v in groups.items()}))


if __name__ == '__main__':
    main()
