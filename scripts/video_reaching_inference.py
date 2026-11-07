#!/usr/bin/env python3
"""Save held-out, observed-input SAMDAMSARNN predictions as MP4 and CSV."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parent / 'ShareSet/share_folder/src'), str(ROOT)]
from data.reaching import ReachingEpisodes, read_signals
from models.SAMDAMSARNN import SAMDAMSARNN


def load_model(config, checkpoint, device):
    m = config['model']
    model = SAMDAMSARNN(rec_dim=m['rec_dim'], k_dim=m['k_dim'], joint_dim=36,
                       temperature=m['temperature'], heatmap_size=m['heatmap_size'], im_size=m['im_size'])
    saved = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(saved['model_state_dict'])
    return model.to(device).eval(), saved['epoch']


class MP4:
    def __init__(self, path, size, fps=5):
        self.path = Path(path);self.size = size
        self.process = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size[0]}x{size[1]}', '-r', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def append(self, rgb):
        assert rgb.shape == (self.size[1], self.size[0], 3)
        self.process.stdin.write(np.asarray(rgb, dtype='uint8').tobytes());self.frames += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:raise RuntimeError(f'Video encoding failed: {self.path}')


COLORS = [(53, 180, 255), (255, 120, 80), (100, 230, 150), (210, 130, 255),
          (250, 210, 70), (90, 220, 220), (240, 140, 190)]


def text(image, label, xy, scale=.65, color=(230, 235, 245)):
    cv2.putText(image, label, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def chart(canvas, rectangle, truth, prediction, index, label):
    x, y, w, h = rectangle;text(canvas, label, (x + 10, y + 24))
    left, top, width, height = x + 55, y + 45, w - 75, h - 80
    a = np.r_[truth, prediction];low, high = float(a.min()), float(a.max())
    margin = max((high - low) * .1, .001);low -= margin;high += margin
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (70, 80, 95), 1)
    text(canvas, f'{high:.2f}', (x + 3, top + 6), .4);text(canvas, f'{low:.2f}', (x + 3, top + height), .4)
    for values, until, is_truth in ((truth, len(truth), True), (prediction, index + 1, False)):
        for j in range(values.shape[1]):
            points = np.column_stack((left + np.arange(until) / max(1, len(truth) - 1) * width,
                                      top + height * (1 - (values[:until, j] - low) / (high - low)))).astype('int32')
            if is_truth:
                for k in range(0, len(points) - 1, 4):cv2.polylines(canvas, [points[k:k + 2]], False, COLORS[j % 7], 1, cv2.LINE_AA)
            else:cv2.polylines(canvas, [points], False, COLORS[j % 7], 2, cv2.LINE_AA)
    px = left + int(index / max(1, len(truth) - 1) * width)
    cv2.line(canvas, (px, top), (px, top + height), (190, 195, 200), 1)
    text(canvas, 'dashed: reference   solid: prediction', (x + 10, y + h - 12), .45)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args();torch.set_num_threads(2);a.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.model_dir / 'configuration.json').read_text());h = config['hyperparameters'];ds = config['dataset']
    stats = json.loads((a.model_dir / 'data.json').read_text())
    lower, upper = np.array(stats['joint_min']), np.array(stats['joint_max'])
    checkpoint = a.checkpoint or a.model_dir / 'SARNN_latest.pth'
    model, epoch = load_model(config, checkpoint, torch.device(a.device))
    manifest = yaml.safe_load(Path(ds['manifest']).read_text())[ds['name']]
    settings = dict(img_size=64, smooth_torque=h['smooth_torque'], skip_num=h['skip_num'], vmin=h['vmin'], vmax=h['vmax'],
                    torque=stats['torque_processing'], expected_camera_profile=ds.get('expected_camera_profile'),
                    expected_lighting_profile=ds.get('expected_lighting_profile'))
    summaries = []
    for name, window in manifest['test'].items():
        path = Path(ds['root']) / ds['name'] / 'test' / name
        spec = dict(path=str(path), **window)
        row = ReachingEpisodes([spec], lower, upper, settings)[0]
        length = len(row['joint']);state = None;predictions = [];images = []
        with torch.inference_mode():
            for t in range(length - 1):
                inputs = [row[k][None, t].to(a.device) for k in ('rgb', 'joint', 'sock', 'foot')]
                image, pred, _, _, state, _ = model(*inputs, state)
                predictions.append(pred[0].cpu().numpy())
                images.append(np.clip(image[0].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype('uint8'))
        prediction = np.array(predictions) * (upper - lower) + lower
        truth = read_signals(path, window['start'], window['end'], h['smooth_torque'], settings['torque'])[::h['skip_num']][1:]
        destination = a.output / name;destination.mkdir(exist_ok=True)
        np.savetxt(destination / 'predicted_state.csv', prediction, delimiter=',')
        np.savetxt(destination / 'reference_next_state.csv', truth, delimiter=',')
        writer = MP4(destination / 'offline-prediction.mp4', (1600, 900), 10 / h['skip_num'])
        for t, predicted in enumerate(images):
            canvas = np.full((900, 1600, 3), (19, 26, 38), dtype='uint8')
            text(canvas, f'Observed-input offline inference | epoch {epoch} | {name}', (20, 30), .7)
            text(canvas, f'target t={(t + 1) * .1 * h["skip_num"]:.1f}s | inputs observed at every step; no autonomous control', (20, 60), .55)
            frame = window['start'] + t * h['skip_num']
            for x, source, label in ((0, path / 'camera_right' / f'{frame}.png', 'Observed input t'),
                                     (533, path / 'camera_right' / f'{frame + h["skip_num"]}.png', 'Reference next frame'),
                                     (1066, predicted, 'Predicted next frame (64x64)')):
                image = np.array(Image.open(source).convert('RGB')) if isinstance(source, Path) else source
                canvas[105:505, x:x + 533] = cv2.resize(image, (533, 400))
                text(canvas, label, (x + 10, 94), .6)
            chart(canvas, (0, 520, 533, 370), truth[:, :7], prediction[:, :7], t, 'Left arm joint angle [rad]')
            chart(canvas, (533, 520, 533, 370), truth[:, 18:25], prediction[:, 18:25], t, 'Left arm effort incl. gravity [Nm]')
            chart(canvas, (1066, 520, 533, 370), truth[:, [7, 16]] * 1000, prediction[:, [7, 16]] * 1000, t, 'Gripper coordinates [mm]')
            writer.append(canvas)
            if t == len(images) // 2:Image.fromarray(canvas).save(destination / 'preview.jpg')
        writer.close()
        metrics = dict(episode=name, epoch=epoch, frames=writer.frames, video=str(writer.path),
                       inference_mode='observed_input_one_step', angle_rmse_rad=float(np.sqrt(np.mean((prediction[:, :7] - truth[:, :7]) ** 2))),
                       torque_rmse_Nm=float(np.sqrt(np.mean((prediction[:, 18:25] - truth[:, 18:25]) ** 2))))
        summaries.append(metrics);print(json.dumps(metrics), flush=True)
    report = dict(completed=True, checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  epoch=epoch, rows=summaries, scope='Held-out one-step predictions with observed state/images at each step; not closed-loop robot control')
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':main()
