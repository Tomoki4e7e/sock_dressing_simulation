#!/usr/bin/env python3
"""Independently check saved head stereo, masks, masked depth and original-image copies."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.head_camera_profile import head_references
from sock_dressing_simulation.perception_audit import projection_precision
from sock_dressing_simulation.reaching_collection import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/reaching-shareset-head-v2')
    a = ap.parse_args();out = a.output.resolve()
    cv2.setNumThreads(1)
    manifest = out / 'report.json'
    if not manifest.exists():manifest = out / 'collection.json'
    report = json.loads(manifest.read_text());results = []
    calibration = json.loads((out / 'data/data_sock_reaching_v1/train/depth.json').read_text())
    quantum = (calibration['depth_max'] - calibration['depth_min']) / 255.
    # DAv2 ends with ReLU: zero is a valid prediction. The fitted lower
    # cutoff is only 0.014 quantization bins above that native floor.
    # Counting encoded zeros as clipped/missing depth is therefore invalid.
    if not 0 <= calibration['depth_min'] < quantum:raise ValueError('Lower depth cutoff discards resolvable model depths')
    def audit_episode(row):
        ep = out / 'data/data_sock_reaching_v1' / row['split'] / row['id']
        meta = json.loads((ep / 'metadata.json').read_text())
        perception = json.loads((ep / 'perception/validation.json').read_text())
        repairs_path = ep / 'perception/head-repairs.json'
        if repairs_path.exists():
            for repair in json.loads(repairs_path.read_text())['rows'].values():
                for change in repair['changes']:
                    if sha256(ep / change['path']) != change['corrected_sha256'] or sha256(ep / change['original']) != change['original_sha256']:
                        raise ValueError('Head repair/original mask provenance changed')
                index = repair['frame']
                if sha256(ep / f'camera_right/{index}.png') != repair['rgb_sha256'] or sha256(ep / f'camera_depth/{index}.png') != repair['depth_sha256']:
                    raise ValueError('Raw RGB/depth changed after head repair')
        if not (meta['camera_profile'] == 'head-stereo-zero-offset-v2' and perception['passed']
                and meta['raw_validation_passed'] and meta['reaching_learning_ready']):raise ValueError('Unqualified episode')
        refs = head_references(meta, ROOT);minimum = dict(sock=1., foot=1.)
        max_clip = max(r['depth_clip_fraction'] for r in perception['rows'])
        saturation = dict(sock=0., foot=0.)
        zero_fraction = dict(sock=0., foot=0.)
        poses = [json.loads(line) for line in (ep / 'camera_poses.jsonl').read_text().splitlines()]
        if len(poses) != meta['frames']:raise ValueError('Camera pose count differs')
        for index, pose in enumerate(poses):
            if pose['frame'] != index or pose['step'] != 5*(index+1) or abs(pose['baseline_m'] - .084) > 1e-4:
                raise ValueError('Stereo pose/step mismatch')
            for side in ('right', 'left'):
                camera = pose['cameras'][side]
                if camera['parent_link'] != f'head/see3cam_{side}/camera_color_frame':raise ValueError('Incorrect camera mount')
                if np.linalg.norm(camera['local_position']) > 1e-6:raise ValueError('Translated camera')
                if np.max(abs((np.asarray(camera['local_rotation']) + 180) % 360 - 180)) > 1e-4:
                    raise ValueError('Rotated camera')
                if np.linalg.norm(np.asarray(camera['position']) - poses[0]['cameras'][side]['position']) > 1e-5:
                    raise ValueError('Head camera moved during fixed-torso replay')
        def read(folder, index, mode):
            with Image.open(ep / folder / f'{index}.png') as im:
                if im.mode != mode or im.size != (1280, 960):raise ValueError(f'Invalid image format: {ep}/{folder}/{index}')
                return np.array(im)
        for index in range(meta['frames']):
            # Read all streams: decoding catches truncated PNGs as well as incorrect headers.
            views = [read(folder, index, 'RGB') for folder in ('camera_right', 'camera_left', 'camera_overview')]
            if any(cv2.meanStdDev(view)[1].max() < 1. for view in views):raise ValueError('Blank camera image')
            if np.array_equal(views[0], views[1]) or np.array_equal(views[0], views[2]):
                raise ValueError('Stereo/overview images are duplicated')
            depth = read('camera_depth', index, 'L');masks = {}
            for kind in ('sock', 'foot'):
                values = read(f'camera_right_mask/{kind}_mask', index, 'L')
                if ((values != 0) & (values != 255)).any():raise ValueError('Nonbinary mask')
                mask = values > 0;masks[kind] = mask
                if int(mask.sum()) != perception['rows'][index]['areas'][kind]:raise ValueError('Stored mask area changed')
                fraction = float((depth[mask] == 255).mean())
                zero_fraction[kind] = max(zero_fraction[kind], float((depth[mask] == 0).mean()))
                saturation[kind] = max(saturation[kind], fraction)
                if fraction > .02:raise ValueError(f'Upper depth saturation exceeds 2%: {ep.name}, frame {index}, {kind}: {fraction}')
                masked_depth = read(f'depth_mask/{kind}_depth', index, 'L')
                if not np.array_equal(masked_depth, np.where(mask, depth, 0)):raise ValueError('Masked depth differs from right depth/mask')
                precision = projection_precision(mask, refs[kind], 2)['tolerant']
                if precision < .8 or not masked_depth.any():raise ValueError('Stored mask/depth fails QA')
                if abs(precision - perception['rows'][index]['projection_precision_2px'][kind]) > 1e-12:
                    raise ValueError('Stored mask changed after processing')
                minimum[kind] = min(minimum[kind], precision)
            if (masks['sock'] & masks['foot']).any():raise ValueError('Class masks overlap')
        results.append(dict(id=row['id'], frames=meta['frames'], minimum_projection_precision_2px=minimum,
            maximum_depth_clip_fraction=max_clip,
            maximum_target_depth_upper_saturation_fraction=saturation,
            maximum_target_zero_fraction=zero_fraction,
            prompt_resets=[r['frame'] for r in perception['resets']]))
        print('STORED HEAD AUDIT PASSED', row['id'], flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(audit_episode, report['rows']))
    order = {row['id']: index for index, row in enumerate(report['rows'])}
    results.sort(key=lambda row: order[row['id']])
    if any(s['split'] != 'train' or s['number'] not in (10, 13, 25, 38) for s in calibration['samples']):
        raise ValueError('Depth calibration uses held-out data')
    if calibration.get('calibration_version')!='full-training-frames-v3':raise ValueError('Full-training calibration required')
    migration_path = out / 'camera-migration.json'
    preserved = 0
    if migration_path.exists():
        migration = json.loads(migration_path.read_text())
        for item in migration['files']:
            if sha256(out / item['preserved']) != item['sha256']:raise ValueError('Preserved overview changed')
            if sha256(Path(migration['overview_archive']) / item['original']) != item['sha256']:raise ValueError('Archived overview changed')
            preserved += 1
    elif (out / 'original-overview-before-migration.json').exists():
        baseline = json.loads((out / 'original-overview-before-migration.json').read_text())
        for item in baseline['files']:
            rel = Path(item['path'])
            target = out / rel.parent.parent / 'confirmation/camera_overview_original' / rel.name
            if sha256(target) != item['sha256'] or sha256(Path(baseline['source']) / rel) != item['sha256']:
                raise ValueError('Original overview copies differ before activation')
            preserved += 1
    result = dict(passed=True, episodes=len(results), frames=sum(r['frames'] for r in results),
        decoded_images=sum(r['frames'] for r in results)*8, original_overview_copies_verified=preserved,
        rows=results, audit_source_sha256=sha256(__file__), model_accuracy_validated=False,
        depth_quality_scope='Zero is valid ReLU/uint8 relative depth; lower cutoff is <1 quantization bin above zero. Target upper saturation <=2%; global clipping including native floor/robot/background is recorded.',
        lower_cutoff_quantization_bins=calibration['depth_min']/quantum,
        scope='Stored data consistency, independent of live tracker')
    (out / 'stored-data-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('ALL STORED HEAD DATA PASSED', result['frames'], 'original overview', preserved, flush=True)


if __name__ == '__main__':main()
