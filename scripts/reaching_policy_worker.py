#!/usr/bin/env python3
"""Persistent ML worker for the separate Python 3.11 Unity client."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT.parent / 'ShareSet/share_folder/src')]
from video_reaching_inference import load_model
from data.reaching import ARM_COLUMNS, normalized_signal
from sock_dressing_simulation.gravity_reference import StaticGravityReference
from sock_dressing_simulation.head_camera_profile import head_references
from sock_dressing_simulation.lit_perception import initialize_lit
from process_reaching_shareset import load_sam, load_depth
from process_head_reaching_shareset import masks_from, inspect_masks


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True);ap.add_argument('--episode', type=Path, required=True)
    ap.add_argument('--device', default='cpu');a = ap.parse_args()
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    with contextlib.redirect_stdout(sys.stderr):
        config = json.loads((a.model_dir / 'configuration.json').read_text())
        stats = json.loads((a.model_dir / 'data.json').read_text());lower = np.array(stats['joint_min']);upper = np.array(stats['joint_max'])
        model, epoch = load_model(config, a.checkpoint, torch.device(a.device))
        # Use the same SAM2 and depth model/configuration as the corrected data.
        sam = load_sam(Path('/home/cm-pc/catkin_ws/src/SegmentAnythingModel2/checkpoints/sam2_hiera_small.pt'), 512)
        depth = load_depth(Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth'))
        calibration = json.loads((Path(config['dataset']['root']) / config['dataset']['name'] / 'train/depth.json').read_text())
        gravity = StaticGravityReference(Path(stats['torque_processing']['gravity_root']).parent / 'gravity-model.json')
        meta = json.loads((a.episode / 'metadata.json').read_text());refs = head_references(meta, ROOT)
    state = None;initialized = False
    print(json.dumps(dict(ready=True, epoch=epoch)), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('quit'):break
        with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
            rgb = np.array(Image.open(request['rgb']).convert('RGB'))
            if not initialized:
                masks, _ = initialize_lit(sam, rgb, refs);initialized = True
            else:
                ids, logits = sam.track(rgb[:, :, ::-1].copy());masks = masks_from(ids, logits)
                checks, _, _ = inspect_masks(masks, refs)
                if not all(checks.values()):masks, _ = initialize_lit(sam, rgb, refs)
            raw_depth = depth.infer_image(rgb[:, :, ::-1].copy(), input_size=518)
            encoded = np.clip((raw_depth - calibration['depth_min']) /
                              (calibration['depth_max'] - calibration['depth_min']) * 255, 0, 255).astype('uint8')
            q = np.array(request['angle'], dtype='float32');effort = np.array(request['torque'], dtype='float32')
            effort[ARM_COLUMNS] += gravity.predict(q[None])[0]
            measured = np.r_[q, effort]
            normalized = normalized_signal(measured[None], lower, upper)[0].astype('float32')
            arrays = [np.array(Image.fromarray(rgb).resize((64, 64)), dtype='float32').transpose(2, 0, 1) / 255., normalized]
            for kind in ('sock', 'foot'):
                image = Image.fromarray(np.where(masks[kind], encoded, 0).astype('uint8'))
                arrays.append(np.array(image.filter(ImageFilter.GaussianBlur(15)).resize((64, 64)), dtype='float32')[None] / 255.)
            inputs = [torch.from_numpy(v[None]).to(a.device) for v in arrays]
            image, prediction, _, _, state, _ = model(*inputs, state)
            prediction = prediction[0].cpu().numpy()
            index = request['index'];blend = 0.
            if 2 < index < 22:
                blend = .8 - .8 / 19 * (index - 3)
                prediction = blend * normalized + (1 - blend) * prediction
            decoded = prediction * (upper - lower) + lower
            if not np.isfinite(decoded).all():raise RuntimeError('Non-finite model command')
            predicted = np.clip(image[0].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype('uint8')
            Image.fromarray(predicted).save(request['predicted_image'])
        print(json.dumps(dict(action=decoded[:18].tolist(), predicted_torque=decoded[18:].tolist(),
                              observation_torque=effort.tolist(), measured_blend=blend,
                              input_out_of_range_fraction=float(((normalized < 0) | (normalized > 1)).mean()))), flush=True)


if __name__ == '__main__':main()
