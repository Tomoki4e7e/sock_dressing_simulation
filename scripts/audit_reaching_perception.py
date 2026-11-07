#!/usr/bin/env python3
"""Audit existing SAM outputs without rerunning or altering model predictions."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.perception_audit import projection_precision


def audit(episode):
    episode = Path(episode).resolve();folder = episode / 'perception'
    meta = json.loads((episode / 'metadata.json').read_text())
    if meta.get('camera_profile') == 'head-stereo-zero-offset-v2':
        raise ValueError('This audit uses the overview camera profile. Head data are checked by process_head_reaching_shareset.py.')
    path = folder / 'validation.json'
    if not path.exists():raise ValueError(f'Full model processing is incomplete: {episode}')
    original_path = folder / 'validation-strict-projection.json'
    previous = json.loads((original_path if original_path.exists() else path).read_text())
    if not original_path.exists():original_path.write_text(json.dumps(previous, indent=2) + '\n')
    if previous['sample_only']:raise ValueError('A sample report cannot qualify a complete episode')
    if not (meta.get('raw_validation_passed') and meta.get('replay_passed')
            and meta.get('actual_closed_capture_passed')):
        raise ValueError('Native replay, closure and raw synchronization must pass before qualification')
    count = int(meta['frames'])
    if previous['frames'] != count or len(previous['rows']) != count:raise ValueError('Perception report frame count mismatch')
    expected = {str(i) for i in range(count)}
    for name in ('camera_right', 'camera_depth', 'camera_right_mask/sock_mask', 'camera_right_mask/foot_mask',
                 'depth_mask/sock_depth', 'depth_mask/foot_depth'):
        if {p.stem for p in (episode / name).glob('*.png')} != expected:
            raise ValueError(f'Frame ID mismatch: {episode / name}')
    digest = hashlib.sha256((episode / 'angle.csv').read_bytes()).hexdigest()
    if previous['angle_sha256'] != digest:raise ValueError('Joint feedback changed after processing')
    with Image.open(folder / 'reference-cloth-amodal.png') as image:
        reference = np.asarray(image) > 0
    report = copy.deepcopy(previous)
    repair_path = folder / 'repairs.json'
    repairs = json.loads(repair_path.read_text())['rows'] if repair_path.exists() else {}
    selected = set(np.linspace(0, count - 1, 8, dtype=int)) | {int(key) for key in repairs}
    thumbs = []
    for index, row in enumerate(report['rows']):
        if str(index) in repairs:
            repair = repairs[str(index)]
            for change in repair['changes']:
                for key, source in (('corrected_sha256', change['path']), ('original_sha256', change['original'])):
                    if hashlib.sha256((episode / source).read_bytes()).hexdigest() != change[key]:
                        raise ValueError('Repaired/original prediction hash changed')
            row = copy.deepcopy(repair['audited_row']);report['rows'][index] = row
        elif str(index - 1) in repairs:
            row['checks'].pop('area_continuity', None)
            row['annotation_transition'] = 'Boundary after an independent SAM2 prompt reset'
        if row['frame'] != index:raise ValueError('Perception frame order changed')
        with Image.open(episode / 'camera_right_mask/sock_mask' / f'{index}.png') as image:
            if image.mode != 'L' or image.size != (1280, 960):raise ValueError('Invalid stored sock mask')
            values = np.asarray(image)
        if not np.isin(values, [0, 255]).all():raise ValueError('Nonbinary sock mask')
        precision = projection_precision(values > 0, reference, 2)
        if abs(precision['strict'] - row['cloth_amodal_precision']) > 1e-12:
            raise ValueError('Mask changed after its model processing report')
        row['cloth_amodal_precision_tolerant'] = precision['tolerant']
        row['checks'].pop('cloth_reference_precision', None)
        row['checks']['cloth_reference_precision_with_2px_raster_tolerance'] = precision['tolerant'] >= .8
        row['passed'] = all(row['checks'].values())
        if index in selected:
            with Image.open(episode / 'camera_right' / f'{index}.png') as image:
                overlay = np.array(image.convert('RGB'))
            for kind, color in (('sock', [255, 40, 40]), ('foot', [40, 255, 40])):
                with Image.open(episode / 'camera_right_mask' / (kind + '_mask') / f'{index}.png') as image:
                    mask = np.asarray(image) > 0
                overlay[mask] = overlay[mask] * .5 + np.array(color) * .5
            thumb = Image.fromarray(overlay).crop((570, 370, 880, 590)).resize((480, 340))
            ImageDraw.Draw(thumb).text((6, 6), f"frame {index} / QA={row['passed']} / 2px / prompt reset={str(index) in repairs}", fill='red')
            thumbs.append(thumb)
    sheet = Image.new('RGB', (1440, 340 * ((len(thumbs) + 2) // 3)), 'white')
    for index, image in enumerate(thumbs):sheet.paste(image, ((index % 3) * 480, (index // 3) * 340))
    sheet.save(folder / 'review-audited.jpg', quality=90)
    report.update(passed=all(row['passed'] for row in report['rows']),
        projection_raster_tolerance_px=2, projection_raster_tolerance_metric='Chebyshev image pixels',
        projection_threshold=.8,
        projection_audit_reason='Binary mesh rasterization and resized SAM logits differ at silhouette edges; allow 2 image pixels without modifying masks or native collision/grasp checks',
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        strict_projection_report='validation-strict-projection.json', stored_model_predictions_unchanged=not bool(repairs),
        repaired_frames=sorted(int(key) for key in repairs), repair_provenance='repairs.json' if repairs else None)
    temp = path.with_suffix('.tmp');temp.write_text(json.dumps(report, indent=2) + '\n');temp.replace(path)
    meta.update(perception_passed=report['passed'], reaching_learning_ready=report['passed'],
                projection_raster_tolerance_px=2, torque_recorded=True, images_synchronized=True)
    meta['perception_repaired_frames'] = sorted(int(key) for key in repairs)
    target = episode / 'metadata.json';temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps(meta, indent=2) + '\n');temp.replace(target)
    progress = dict(completed=True, processed=count, total=count, passed=report['passed'],
                    elapsed_s=report['elapsed_s'], projection_audit=True,
                    last=report['rows'][-1])
    (folder / 'progress.json').write_text(json.dumps(progress, indent=2) + '\n')
    summary = dict(episode=episode.name, passed=report['passed'], frames=count,
                   failed_frames=sum(not row['passed'] for row in report['rows']),
                   minimum_strict_precision=min(row['cloth_amodal_precision'] for row in report['rows']),
                   minimum_tolerant_precision=min(row['cloth_amodal_precision_tolerant'] for row in report['rows']))
    print(json.dumps(summary), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('episodes', nargs='+', type=Path)
    args = ap.parse_args();results = [audit(path) for path in args.episodes]
    if not all(row['passed'] for row in results):raise RuntimeError('Perception audit failed')


if __name__ == '__main__':main()
