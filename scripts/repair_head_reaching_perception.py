#!/usr/bin/env python3
"""Repair isolated head-view SAM failures with 1024px current-image prompts."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1];sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from process_reaching_shareset import load_sam, digest, write_json
from process_head_reaching_shareset import initialize, masks_from, inspect_masks, PROFILE
from sock_dressing_simulation.head_camera_profile import head_references, head_prompts, interior_point


def repair_prediction(sam, rgb, refs, old):
    prompts = head_prompts(rgb, refs)
    support = {k: cv2.dilate(m.astype('uint8'), np.ones((5,5), dtype='uint8')) > 0 for k,m in refs.items()}
    masks = old
    attempts = []
    for attempt in range(3):
        for kind in ('sock', 'foot'):
            wrong = masks[kind] & ~support[kind]
            for _ in range(6):
                if not wrong.any():break
                point = interior_point(wrong)
                prompts[kind]['points'].append(point);prompts[kind]['labels'].append(0)
                cv2.circle(wrong.view('uint8'), tuple(point), 18, 0, -1)
        ids, logits = initialize(sam, rgb, prompts)
        masks = masks_from(ids, logits);checks, areas, precision = inspect_masks(masks, refs)
        attempts.append(dict(attempt=attempt, checks=checks, areas=areas, precision=precision, prompts=copy.deepcopy(prompts)))
        if all(checks.values()):return masks, attempts
    raise RuntimeError(f'Head repair still fails semantic QA: {attempts[-1]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('episode', type=Path)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--probe-frames', nargs='*', type=int)
    a = ap.parse_args();ep = a.episode.resolve();folder = ep/'perception'
    meta = json.loads((ep/'metadata.json').read_text())
    if not meta.get('raw_validation_passed') or meta.get('camera_profile') != PROFILE:raise ValueError('Qualified head stereo raw data required')
    refs = head_references(meta, ROOT)
    path = folder/'validation.json'
    report = json.loads(path.read_text()) if path.exists() else None
    if a.probe_frames is None:
        if report is None:raise ValueError('Wait until the full head processor finishes')
        frames = [row['frame'] for row in report['rows'] if not row['passed']]
        if not frames:print('NO HEAD REPAIR NEEDED');return
        if len(frames) > 50:raise ValueError('Too many failures for isolated repair')
        for index in frames:
            bad = {key for key,value in report['rows'][index]['checks'].items() if not value}
            if bad - {'sock_projection', 'foot_projection'}:raise ValueError('Failure requires separate inspection')
    else:frames = a.probe_frames
    checkpoint = Path('/home/cm-pc/catkin_ws/src/SegmentAnythingModel2/checkpoints/sam2_hiera_small.pt')
    log_path = folder/'head-repairs.json'
    log = json.loads(log_path.read_text()) if log_path.exists() else dict(method='Independent current-image SAM2 1024; negative prompts on drift outside projected support', rows={})
    if a.probe_frames is None:
        original = folder/'validation-before-isolated-repair.json'
        if not original.exists():shutil.copy2(path, original)
        previous = folder/'review-before-isolated-repair.jpg'
        if not previous.exists():shutil.copy2(folder/'review.jpg', previous)
    torch.set_num_threads(a.threads);torch.set_grad_enabled(False)
    with torch.inference_mode():
        sam = load_sam(checkpoint, 1024)
        for index in frames:
            rgb = np.array(Image.open(ep/'camera_right'/f'{index}.png').convert('RGB'))
            old = {k: np.array(Image.open(ep/'camera_right_mask'/f'{k}_mask'/f'{index}.png')) > 0 for k in ('sock','foot')}
            masks, attempts = repair_prediction(sam, rgb, refs, old)
            print('HEAD REPAIR PREDICTION PASSED', ep.name, index, attempts[-1]['precision'], flush=True)
            if a.probe_frames is not None:continue
            depth = np.array(Image.open(ep/'camera_depth'/f'{index}.png'))
            checks, areas, precision = inspect_masks(masks, refs)
            checks.update(depth_finite=report['rows'][index]['checks']['depth_finite'],
                          depth_nonzero=all(bool(depth[mask].any()) for mask in masks.values()))
            if not all(checks.values()):raise ValueError('Repaired depth/masks fail QA')
            changes = []
            for kind,mask in masks.items():
                for name,data in [(f'camera_right_mask/{kind}_mask',mask.astype('uint8')*255),
                                  (f'depth_mask/{kind}_depth',np.where(mask,depth,0).astype('uint8'))]:
                    target = ep/name/f'{index}.png';original = folder/'isolated-repair-originals'/str(index)/name/'original.png'
                    original.parent.mkdir(parents=True, exist_ok=True)
                    if not original.exists():shutil.copy2(target, original)
                    Image.fromarray(data).save(target)
                    changes.append(dict(path=str(target.relative_to(ep)), original=str(original.relative_to(ep)),
                        original_sha256=digest(original), corrected_sha256=digest(target)))
            row = copy.deepcopy(report['rows'][index]);row.update(passed=True,checks=checks,areas=areas,projection_precision_2px=precision,
                annotation_transition='Independent current-frame SAM2 1024 prompt correction')
            log['rows'][str(index)] = dict(frame=index, attempts=attempts, changes=changes,
                original_row=report['rows'][index], corrected_row=row, rgb_sha256=digest(ep/'camera_right'/f'{index}.png'),
                depth_sha256=digest(ep/'camera_depth'/f'{index}.png'), sam_checkpoint_sha256=digest(checkpoint))
            report['rows'][index] = row;write_json(log_path, log)
    if a.probe_frames is not None:return
    report.update(passed=all(row['passed'] for row in report['rows']), isolated_repaired_frames=sorted(int(key) for key in log['rows']),
                  isolated_repair_source_sha256=digest(__file__), isolated_repair_provenance='head-repairs.json')
    write_json(path, report)
    meta.update(perception_passed=report['passed'], reaching_learning_ready=report['passed'],
                head_isolated_repaired_frames=report['isolated_repaired_frames'])
    write_json(ep/'metadata.json',meta)
    thumbs=[]
    for index in sorted(set(np.linspace(0,meta['frames']-1,8,dtype=int)) | set(report['isolated_repaired_frames'])):
        rgb=np.array(Image.open(ep/'camera_right'/f'{index}.png'))
        for kind,color in [('sock',[255,40,40]),('foot',[40,255,40])]:
            mask=np.array(Image.open(ep/'camera_right_mask'/f'{kind}_mask'/f'{index}.png'))>0
            rgb[mask]=rgb[mask]*.5+np.array(color)*.5
        im=Image.fromarray(rgb).resize((480,360));ImageDraw.Draw(im).text((5,5),f'HEAD frame {index} repair={str(index) in log["rows"]}',fill='red');thumbs.append(im)
    sheet=Image.new('RGB',(1440,360*((len(thumbs)+2)//3)),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*480,(i//3)*360))
    sheet.save(folder/'review.jpg',quality=90)
    write_json(folder/'progress.json',dict(completed=True,passed=report['passed'],processed=meta['frames'],total=meta['frames'],isolated_repairs=len(log['rows']),last=report['rows'][-1]))
    if not report['passed']:raise ValueError('Head episode still fails after isolated repair')
    print('HEAD ISOLATED REPAIRS PASSED',ep.name,report['isolated_repaired_frames'],flush=True)


if __name__=='__main__':main()
