#!/usr/bin/env python3
"""Publish a verified fixed-motion conformance result, retaining failed trials."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tune_foot_conforming import OUTPUT, assess, score
from scripts.summarize_foot_contact_replay import export_diagnostics


def write_comparison(output: Path, basename: str = 'comparison'):
    rows = []
    paths=[];visited=set()
    # Archived recordings retain their original paths through directory
    # symlinks. Include those records while preventing recursive link loops.
    for directory,names,files in os.walk(output,followlinks=True):
        real=Path(directory).resolve()
        if real in visited:
            names[:]=[]
            continue
        visited.add(real)
        if 'result.json' in files:paths.append(Path(directory)/'result.json')
    for path in sorted(paths):
        record = json.loads(path.read_text())
        metadata_path = Path(record['metadata'])
        metadata = json.loads(metadata_path.read_text())
        steps = int(metadata['frames'])
        row = assess(metadata_path, steps)
        config = yaml.safe_load((path.parent / 'config.yaml').read_text())
        rollout = config.get('foot_conforming_rollout', {})
        parameters = rollout.get('parameters', config['obi']['expected'])
        row.update(condition=str(path.parent.relative_to(output)),
                   parameters={key: parameters.get(key) for key in
                       ('friction', 'damping', 'stretch_compliance', 'bend_compliance',
                        'particle_radius_m', 'collision_margin_m')},
                   substeps=parameters['substeps'], iterations=parameters['solver_iterations'],
                   mode=rollout.get('surface_mode', 'legacy'),
                   transport=rollout.get('transport', 'bulk'),
                   fixed=rollout.get('fixed_grasp_offsets', False),
                   replay_engine_sha256=metadata.get('replay_engine_sha256'))
        rows.append(row)
    columns = ['condition', 'steps', 'safe', 'conforming_passed', 'mean_tail_toes_containment',
               'mean_tail_forefoot_containment', 'mean_tail_surface_containment',
               'containment_metric', 'legacy_mean_tail_toes_containment',
               'legacy_mean_tail_forefoot_containment', 'legacy_mean_tail_surface_containment',
               'tail_stretch_violation_frames', 'tail_maximum_stretch',
               'maximum_post_penetration_m', 'maximum_anchor_error_m', 'maximum_offset_drift_m',
               'maximum_contact_correction_m', 'maximum_command_error', 'original_task_success',
               'substeps', 'iterations', 'mode', 'transport', 'fixed', 'replay_engine_sha256', 'parameters']
    with (output / (basename + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows({**row, 'parameters': json.dumps(row['parameters'], sort_keys=True)} for row in rows)
    (output / (basename + '.json')).write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def publish(output: Path, visual_reviewed: bool):
    comparison = json.loads((output / 'final_comparison.json').read_text())
    rows = []
    for condition in comparison['conditions']:
        current = assess(Path(condition['metadata']), 100)
        current.update({key: condition[key] for key in
                        ('condition', 'parameters', 'substeps', 'iterations', 'mode', 'transport', 'fixed')})
        rows.append(current)
    passing = [row for row in rows if row['conforming_passed']]
    passing.sort(key=lambda row: (*score(row), row['substeps'] * row['iterations']))
    if not passing:
        raise RuntimeError('No complete, conforming recording is available')
    if not visual_reviewed:
        raise RuntimeError('Review overview and side movies before publishing')
    selected = passing[0]
    metadata_path = Path(selected['metadata'])
    episode = metadata_path.parent
    condition = metadata_path.parents[3]
    metadata = json.loads(metadata_path.read_text())
    best = output / 'best'
    best.mkdir(exist_ok=True)
    for name in ('demo.mp4', 'inference_camera.mp4', 'applied_action.csv', 'metadata.json'):
        shutil.copy2(episode / name, best / name)
    for name in ('side_demo.mp4', 'config.yaml', 'material.yaml', 'applied_material.json'):
        shutil.copy2(condition / name, best / name)
    for name in ('live_frame_diagnostics.jsonl', 'grasp_feasibility.json'):
        if (condition / name).exists():
            shutil.copy2(condition / name, best / name)
    published_config=yaml.safe_load((best/'config.yaml').read_text())
    if metadata.get('native_player_executable'):
        published_config['rcareworld']['executable']=metadata['native_player_executable']
    (best/'config.yaml').write_text(yaml.safe_dump(published_config,sort_keys=False))
    export_diagnostics(metadata_path, best / 'frame_diagnostics.csv')
    (best / 'selection.json').write_text(json.dumps(selected, indent=2) + '\n')
    (best / 'frame_diagnostics.json').write_text(json.dumps({
        'dressing_quality_by_frame': metadata['dressing_quality_by_frame'],
        'cloth_quality_by_frame': metadata['cloth_quality_by_frame'],
        'grasp_quality_by_frame': metadata['grasp_quality_by_frame']}, indent=2) + '\n')
    comparison.update(conditions=rows, best=selected['condition'], visual_review_required=False,
                      overview_and_side_reviewed=True)
    (output / 'final_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    write_comparison(output)
    return selected


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=OUTPUT)
    parser.add_argument('--visual-reviewed', action='store_true')
    parser.add_argument('--comparison-only', action='store_true')
    args = parser.parse_args()
    if args.comparison_only:
        rows = write_comparison(args.output_root, 'comparison-current')
        print(json.dumps({'recorded_trials': len(rows),
                          'completed_trials': sum(row['checks']['completed'] and row['steps']>0 for row in rows),
                          'conforming_trials': sum(row['conforming_passed'] for row in rows)}))
    else:
        print(json.dumps(publish(args.output_root, args.visual_reviewed), indent=2))
