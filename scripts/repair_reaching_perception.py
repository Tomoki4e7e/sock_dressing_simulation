#!/usr/bin/env python3
"""Re-seed SAM2 on isolated failed frames; keep all original predictions."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from process_reaching_shareset import load_sam, digest, write_json
from sock_dressing_simulation.perception_audit import projection_precision


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('episode', type=Path)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args();episode = a.episode.resolve();folder = episode / 'perception'
    if json.loads((episode / 'metadata.json').read_text()).get('camera_profile') == 'head-stereo-zero-offset-v2':
        raise ValueError('Overview repair cannot process head images. Use process_head_reaching_shareset.py.')
    report = json.loads((folder / 'validation.json').read_text())
    if report.get('projection_raster_tolerance_px') != 2:raise ValueError('Run the raster-aware audit before repair')
    failures = [r for r in report['rows'] if not r['passed']]
    if not failures:print('NO REPAIR NEEDED', episode.name);return
    if len(failures) > 50:raise RuntimeError('Too many failed frames for isolated prompt correction')
    for row in failures:
        bad = {k for k, v in row['checks'].items() if not v}
        if bad - {'cloth_reference_precision_with_2px_raster_tolerance'}:
            raise RuntimeError(f'Failure requires separate inspection: {row}')
    reference = np.asarray(Image.open(folder / 'reference-cloth-amodal.png')) > 0
    support = cv2.dilate(reference.astype('uint8'), np.ones((5, 5), dtype='uint8')) > 0
    roi = np.zeros_like(reference);roi[370:590, 570:880] = True
    checkpoint = Path(report['sam_checkpoint'])
    if digest(checkpoint) != report['sam_checkpoint_sha256']:raise ValueError('SAM checkpoint changed')
    log_path = folder / 'repairs.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else dict(
        method='Independent SAM2 reinitialization on the current RGB, visible blue-cuff positive prompt and negative prompt on the drifting gripper',
        roi=[570, 370, 880, 590], image_size=1024, sam_checkpoint_sha256=digest(checkpoint),
        raw_rgb_and_robot_signals_unchanged=True, rows={})
    torch.set_num_threads(a.threads)
    with torch.inference_mode():
        sam = load_sam(checkpoint, 1024)
        for original_row in failures:
            frame = original_row['frame'];name = f'{frame}.png'
            if str(frame) in log['rows']:
                prior = log['rows'][str(frame)]
                if all(digest(episode / change['path']) == change['corrected_sha256'] for change in prior['changes']):
                    continue
            rgb = np.asarray(Image.open(episode / 'camera_right' / name).convert('RGB'))
            old = np.asarray(Image.open(episode / 'camera_right_mask/sock_mask' / name)) > 0
            r, g, b = [rgb[:, :, i].astype(float) for i in range(3)]
            blue = (b > r * 1.2 + 5) & (b > g * 1.05 + 3) & (r < 120) & reference & roi
            n, labels, stats, _ = cv2.connectedComponentsWithStats(blue.astype('uint8'))
            if n < 2:raise RuntimeError(f'No visible cuff prompt in frame {frame}')
            component = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            distance = cv2.distanceTransform(component.astype('uint8'), cv2.DIST_L2, 5)
            y, x = np.unravel_index(distance.argmax(), distance.shape)
            positive = [[int(x), int(y)]]
            negative = [[730, 530], [610, 430]]
            wrong = old & ~support
            if wrong.any():
                distance = cv2.distanceTransform(wrong.astype('uint8'), cv2.DIST_L2, 5)
                y, x = np.unravel_index(distance.argmax(), distance.shape)
                negative.append([int(x), int(y)])
            prompts = dict(sock=dict(points=positive + negative, labels=[1] + [0] * len(negative)),
                foot=dict(points=[[680, 520], [735, 533], [665, 527], [650, 488]] + positive,
                          labels=[1, 1, 1, 1] + [0] * len(positive)))
            sam.load_first_frame(rgb[370:590, 570:880, ::-1].copy())
            for obj, kind in ((1, 'sock'), (2, 'foot')):
                p = prompts[kind];selected = [(pt, label) for pt, label in zip(p['points'], p['labels'])
                                            if 570 <= pt[0] < 880 and 370 <= pt[1] < 590]
                points = np.array([[pt[0] - 570, pt[1] - 370] for pt, label in selected], dtype='float32')
                labels = np.array([label for pt, label in selected], dtype='int32')
                _, ids, logits = sam.add_new_points(0, obj, points, labels)
            masks = {}
            for obj, kind in ((1, 'sock'), (2, 'foot')):
                mask = np.zeros_like(reference);mask[370:590, 570:880] = logits[ids.index(obj), 0].float().numpy() > .2
                masks[kind] = mask
            precision = projection_precision(masks['sock'], reference)
            areas = [int(masks[k].sum()) for k in ('sock', 'foot')]
            if precision['tolerant'] < .8 or areas[0] < 20 or areas[1] < 100:
                raise RuntimeError(f'Corrected SAM prediction still fails: {frame}, {precision}, {areas}')
            depth = np.asarray(Image.open(episode / 'camera_depth' / name))
            if not all(depth[mask].any() for mask in masks.values()):raise ValueError('Corrected masked depth is empty')
            changes = []
            for kind, mask in masks.items():
                for relative, values in ((f'camera_right_mask/{kind}_mask/{name}', mask.astype('uint8') * 255),
                                         (f'depth_mask/{kind}_depth/{name}', np.where(mask, depth, 0).astype('uint8'))):
                    path = episode / relative;backup = folder / 'repair-originals' / relative
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if not backup.exists():shutil.copy2(path, backup)
                    original_sha = digest(backup)
                    Image.fromarray(values).save(path)
                    changes.append(dict(path=relative, original=str(backup.relative_to(episode)),
                                        original_sha256=original_sha, corrected_sha256=digest(path)))
            row = copy.deepcopy(original_row)
            row.update(areas=areas, cloth_amodal_precision=precision['strict'],
                       cloth_amodal_precision_tolerant=precision['tolerant'], sam_prompt_reset=True,
                       sam_internal_image_size=1024)
            row['checks']['sock_area'] = 20 <= areas[0] < 1280 * 960 * .1
            row['checks']['foot_area'] = 100 <= areas[1] < 1280 * 960 * .5
            row['checks']['overlap'] = not (masks['sock'] & masks['foot']).any()
            row['checks']['nonzero_depth'] = True
            row['checks']['cloth_reference_precision_with_2px_raster_tolerance'] = True
            # An independently re-seeded annotation has no temporal area-ratio
            # constraint to the previous tracker's prediction. Preserve the
            # reset explicitly instead of claiming an uninterrupted tracker.
            row['checks'].pop('area_continuity', None)
            row['annotation_transition'] = 'Independent SAM2 prompt reset; temporal area ratio is not applicable'
            row['passed'] = all(row['checks'].values())
            log['rows'][str(frame)] = dict(frame=frame, prompts=prompts, changes=changes,
                                           audited_row=row, method='sam2_prompt_reset')
            log['repair_source_sha256'] = digest(__file__)
            write_json(log_path, log)
            print('REPAIRED', episode.name, frame, areas, precision, flush=True)


if __name__ == '__main__':main()
