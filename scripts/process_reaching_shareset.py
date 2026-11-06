#!/usr/bin/env python3
"""Run ShareSet SAM2/Depth Anything on synchronized reaching episodes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
SHARE = ROOT.parent / 'ShareSet'
sys.path[:0] = [str(SHARE / 'sam2_realtime'), str(SHARE / 'sam2_realtime/sam2'),
                str(SHARE / 'Depth-Anything-V2')]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path);tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n');tmp.replace(path)


def reference_cloth_mask(meta):
    from scipy.spatial.transform import Rotation
    state = json.loads((ROOT / meta['geometry_record']['state_file']).read_text())
    plan = json.loads((ROOT / meta['trajectory']).read_text())
    vertices = np.asarray(state['particles_world']) + plan['translation_m']
    camera = meta['camera_transform']
    xyz = (vertices - np.asarray(camera['position'])) @ Rotation.from_quat(camera['quaternion']).as_matrix()
    focal = 960 / (2 * np.tan(np.deg2rad(meta['camera_fov']) / 2))
    xy = np.column_stack([640 + xyz[:, 0] / xyz[:, 2] * focal, 480 - xyz[:, 1] / xyz[:, 2] * focal])
    mask = np.zeros((960, 1280), dtype='uint8')
    cv2.fillPoly(mask, [np.rint(xy[face]).astype('int32') for face in np.asarray(state['triangles']).reshape(-1, 3)], 255)
    return mask > 0


def prompts(rgb, reference=None):
    # The adopted reference sock is blue. Use its visible cuff only to seed
    # SAM; every saved foreground mask is produced by SAM2, not color masking.
    r, g, b = [rgb[..., i].astype(float) for i in range(3)]
    seed = (b > r * 1.2 + 5) & (b > g * 1.05 + 3) & (r < 120)
    roi = np.zeros(seed.shape, bool);roi[350:600, 550:800] = True;seed &= roi
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed.astype('uint8'))
    if n < 2:raise RuntimeError('No visible blue cuff to initialize SAM2')
    component = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    distance = cv2.distanceTransform(component.astype('uint8'), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    ys, xs = np.where(component)
    positive = [[int(x), int(y)]]
    for fraction in (.2, .8):
        desired = np.array([np.quantile(ys, fraction), np.quantile(xs, fraction)])
        points = np.column_stack([ys, xs]);index = np.argmin(np.linalg.norm(points - desired, axis=1))
        positive.append([int(xs[index]), int(ys[index])])
    if reference is not None:
        # The reference cloth includes a dark, folded surface above the blue
        # cuff. Seed both parts; color alone only identifies the cuff edge.
        rows, columns = np.indices(reference.shape)
        region = reference & (rows < y - 4) & (rows > y - 30) & (columns > x - 25) & (columns < x + 10)
        if not region.any():raise RuntimeError('Reference folded-cloth prompt unavailable')
        distance = cv2.distanceTransform(region.astype('uint8'), cv2.DIST_L2, 5)
        sy, sx = np.unravel_index(np.argmax(distance), distance.shape)
        positive.append([int(sx), int(sy)])
    return dict(sock=dict(points=positive + [[730, 530], [610, 430]], labels=[1] * len(positive) + [0, 0]),
                foot=dict(points=[[680, 520], [735, 533], [665, 527], [650, 488],
                                  [int(x), int(y)], [1040, 460], [690, 565]], labels=[1, 1, 1, 1, 0, 0, 0]))


def load_depth(checkpoint):
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'))
    return model.eval()


def load_sam(checkpoint, image_size=512):
    from sam2.build_sam import build_sam2_camera_predictor
    return build_sam2_camera_predictor(str(SHARE / 'sam2_realtime/sam2_configs/sam2_hiera_s.yaml'),
                                      str(checkpoint), device='cpu', apply_postprocessing=False,
                                      hydra_overrides_extra=[f'++model.image_size={image_size}',
                                                            '++model.non_overlap_masks=true'])


def calibration(args, depth):
    out = args.output
    path = out / 'data/data_sock_reaching_v1/train/depth.json'
    if path.exists():
        record = json.loads(path.read_text())
        if record['checkpoint_sha256'] != digest(args.depth_checkpoint):raise ValueError('Depth calibration checkpoint changed')
        return record
    samples = []
    for number in (10, 13, 25, 38):
        folder = ROOT / f'artifacts/diverse-left-reaching/final/replays/pose-{number:02d}-held-A-v00-trajectory/native-route-initial'
        for name in ('fixed-camera-after.png', 'fixed-camera-endpoint.png'):
            image = folder / name
            raw = depth.infer_image(cv2.imread(str(image)), input_size=518)
            samples.append(dict(path=str(image), sha256=digest(image), minimum=float(raw.min()), maximum=float(raw.max()),
                                number=number, split='train'))
    record = dict(depth_min=min(r['minimum'] for r in samples), depth_max=max(r['maximum'] for r in samples),
                  checkpoint=str(args.depth_checkpoint), checkpoint_sha256=digest(args.depth_checkpoint),
                  source='Frozen calibration on initial/final snapshots from training conditions only; no validation/test images', samples=samples)
    if record['depth_max'] <= record['depth_min']:raise ValueError('Degenerate depth range')
    path.parent.mkdir(parents=True, exist_ok=True);write_json(path, record)
    return record


def process(args):
    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    depth = load_depth(args.depth_checkpoint)
    cal = calibration(args, depth)
    if args.calibrate_only:
        print('CALIBRATED', cal['depth_min'], cal['depth_max'], flush=True);return
    episode = args.episode.resolve()
    meta = json.loads((episode / 'metadata.json').read_text())
    if not meta.get('raw_validation_passed'):raise ValueError('Raw episode validation must pass first')
    raw_hash = digest(episode / 'angle.csv')
    sam = load_sam(args.sam_checkpoint, args.sam_image_size)
    files = sorted((episode / 'camera_right').glob('*.png'), key=lambda p: int(p.stem))
    if len(files) != meta['frames']:raise ValueError('RGB count changed')
    if args.sample_only:
        rows = list(csv.DictReader((episode / 'frames.csv').open()))
        chosen = {0, len(files) - 1}
        for stage in sorted({r['stage'] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r['stage'] == stage]
            chosen.update((ids[0], ids[len(ids) // 2], ids[-1]))
        files = [files[i] for i in sorted(chosen)]
    target = episode / (f'pilot-perception-{args.sam_image_size}' if args.sample_only else 'perception')
    target.mkdir(exist_ok=True)
    dirs = ['camera_depth', 'camera_right_mask/sock_mask', 'camera_right_mask/foot_mask',
            'depth_mask/sock_depth', 'depth_mask/foot_depth']
    dest = target if args.sample_only else episode
    for name in dirs:(dest / name).mkdir(parents=True, exist_ok=True)
    first = np.array(Image.open(files[0]).convert('RGB'))
    reference = reference_cloth_mask(meta)
    Image.fromarray(reference.astype('uint8') * 255).save(target / 'reference-cloth-amodal.png')
    points = prompts(first, reference);write_json(target / 'prompts.json', points)
    x0, y0, x1, y1 = args.sam_roi
    if not 0 <= x0 < x1 <= 1280 or not 0 <= y0 < y1 <= 960:raise ValueError('Invalid SAM ROI')
    sam.load_first_frame(first[y0:y1, x0:x1, ::-1].copy())
    logits = None
    for obj_id, key in ((1, 'sock'), (2, 'foot')):
        p = points[key]
        selected = [(pt, label) for pt, label in zip(p['points'], p['labels']) if x0 <= pt[0] < x1 and y0 <= pt[1] < y1]
        coordinates = np.array([[pt[0] - x0, pt[1] - y0] for pt, label in selected], dtype=np.float32)
        labels = np.array([label for pt, label in selected], dtype=np.int32)
        _, ids, logits = sam.add_new_points(0, obj_id, coordinates, labels)
    report = [];previous = None;thumbs = []
    start = time.monotonic()
    for i, file in enumerate(files):
        frame = np.array(Image.open(file).convert('RGB'))
        if i:ids, logits = sam.track(frame[y0:y1, x0:x1, ::-1].copy())
        masks = {}
        for obj_id, key in ((1, 'sock'), (2, 'foot')):
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[y0:y1, x0:x1] = logits[ids.index(obj_id), 0].float().cpu().numpy() > .2
            masks[key] = mask
        raw = depth.infer_image(frame[..., ::-1].copy(), input_size=518)
        normalized = (raw - cal['depth_min']) / (cal['depth_max'] - cal['depth_min'])
        encoded = np.clip(normalized * 255, 0, 255).astype('uint8')
        areas = [int(masks[k].sum()) for k in ('sock', 'foot')]
        checks = dict(finite=bool(np.isfinite(raw).all()), sock_area=20 <= areas[0] < frame.shape[0] * frame.shape[1] * .1,
                      foot_area=100 <= areas[1] < frame.shape[0] * frame.shape[1] * .5,
                      distinct=not np.array_equal(masks['sock'], masks['foot']),
                      overlap=float((masks['sock'] & masks['foot']).sum() / max(1, min(areas))) <= .25)
        checks['cloth_reference_precision'] = float((masks['sock'] & reference).sum() / max(1, areas[0])) >= .8
        checks['nonzero_depth'] = all(bool(encoded[mask].any()) for mask in masks.values())
        checks['depth_clip_fraction'] = float(((normalized < 0) | (normalized > 1)).mean()) <= .02
        if previous is not None:
            checks['area_continuity'] = max(max(a, b) / max(1, min(a, b)) for a, b in zip(areas, previous)) <= 4
        row = dict(frame=int(file.stem), passed=all(checks.values()), checks=checks, areas=areas,
                   cloth_amodal_precision=float((masks['sock'] & reference).sum() / max(1, areas[0])),
                   depth_clip_fraction=float(((normalized < 0) | (normalized > 1)).mean()))
        report.append(row);previous = areas
        Image.fromarray(encoded).save(dest / 'camera_depth' / file.name)
        for key in ('sock', 'foot'):
            Image.fromarray(masks[key].astype('uint8') * 255).save(dest / 'camera_right_mask' / (key + '_mask') / file.name)
            Image.fromarray(np.where(masks[key], encoded, 0).astype('uint8')).save(dest / 'depth_mask' / (key + '_depth') / file.name)
        if args.sample_only or i in set(np.linspace(0, len(files) - 1, 8, dtype=int)):
            overlay = frame.copy()
            for key, color in (('sock', [255, 40, 40]), ('foot', [40, 255, 40])):
                mask = masks[key];overlay[mask] = overlay[mask] * .5 + np.array(color) * .5
            thumb = Image.fromarray(overlay).resize((480, 360));ImageDraw.Draw(thumb).text((4, 4), f'{file.stem} {areas} QA={row["passed"]}', fill='red')
            thumbs.append(thumb)
        if i % 25 == 0:
            write_json(target / 'progress.json', dict(processed=i + 1, total=len(files), last=row, elapsed_s=time.monotonic() - start))
            print('PERCEPTION', episode.name, i + 1, '/', len(files), 'areas', areas, 'seconds', round(time.monotonic() - start), flush=True)
    sheet = Image.new('RGB', (480 * 3, 360 * ((len(thumbs) + 2) // 3)), 'white')
    for i, thumb in enumerate(thumbs):sheet.paste(thumb, ((i % 3) * 480, (i // 3) * 360))
    sheet.save(target / 'review.jpg', quality=90)
    outcome = dict(passed=all(r['passed'] for r in report), frames=len(report), sample_only=args.sample_only,
                   rows=report, depth_calibration=cal, sam_checkpoint=str(args.sam_checkpoint),
                   sam_checkpoint_sha256=digest(args.sam_checkpoint), sam_config='sam2_hiera_s.yaml',
                   sam_source=str(SHARE / 'sam2_realtime'), shared_image_encoder=True, objects=['sock', 'foot'],
                   sam_threshold=.2, sam_hole_filling=False, sam_non_overlap_masks=True,
                   depth_input_size=518, rgb_order='RGB; BGR passed to ShareSet models',
                   sam_image_size=args.sam_image_size,
                   sam_roi=args.sam_roi, processing_profile='roi-v2',
                   processing_source_sha256=digest(__file__),
                   elapsed_s=time.monotonic() - start, angle_sha256=raw_hash)
    write_json(target / 'validation.json', outcome)
    if not args.sample_only:
        meta.update(perception_passed=outcome['passed'], reaching_learning_ready=outcome['passed'],
                    depth_encoding='uint8 clipped Depth Anything V2 relative depth with frozen training calibration',
                    depth_calibration=str(args.output / 'data/data_sock_reaching_v1/train/depth.json'))
        write_json(episode / 'metadata.json', meta)
    if not outcome['passed']:raise RuntimeError(f'Perception QA failed; inspect {target / "review.jpg"}')
    print('PERCEPTION PASSED', episode.name, len(report), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/reaching-shareset-v1')
    ap.add_argument('--episode', type=Path)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--sam-image-size', type=int, choices=[512, 1024], default=512)
    ap.add_argument('--sam-roi', type=int, nargs=4, default=[570, 370, 880, 590], metavar=('X0', 'Y0', 'X1', 'Y1'))
    ap.add_argument('--sample-only', action='store_true')
    ap.add_argument('--calibrate-only', action='store_true')
    ap.add_argument('--sam-checkpoint', type=Path, default=Path('/home/cm-pc/catkin_ws/src/SegmentAnythingModel2/checkpoints/sam2_hiera_small.pt'))
    ap.add_argument('--depth-checkpoint', type=Path, default=Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth'))
    args = ap.parse_args();args.output = args.output.resolve()
    if not args.calibrate_only and args.episode is None:ap.error('--episode is required')
    with torch.inference_mode():process(args)


if __name__ == '__main__':main()
