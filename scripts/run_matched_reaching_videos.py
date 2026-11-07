#!/usr/bin/env python3
"""Generate intermediate/final offline and live-robot videos for a training run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    temp = path.with_suffix('.tmp');temp.write_text(json.dumps(value, indent=2) + '\n');temp.replace(path)


def generate(run, epoch, live_steps=260, device='auto', output=None):
    model = run / 'model';checkpoint = model / f'SARNN_{epoch}.pth'
    out = Path(output) if output else run / f'inference-epoch-{epoch}';out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MPLCONFIGDIR='/tmp/samdam-gravity-mpl', PYTHONUNBUFFERED='1')
    offline = out / 'offline'
    if not (offline / 'report.json').exists():
        with (out / 'offline.log').open('w') as log:
            subprocess.run(['/usr/bin/python3', str(ROOT / 'scripts/video_reaching_inference.py'), '--model-dir', str(model),
                '--checkpoint', str(checkpoint), '--output', str(offline), '--device', device], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    config = json.loads((model / 'configuration.json').read_text())
    data = config['dataset'];episodes = sorted((Path(data['root']) / data['name'] / 'test').glob('*/metadata.json'))
    records = []
    for i, metadata in enumerate(episodes):
        episode = metadata.parent;meta = json.loads(metadata.read_text())
        requested_steps = min(live_steps, (meta['frames'] - 1) // config['hyperparameters']['skip_num'])
        destination = out / 'simulator' / episode.name;destination.mkdir(parents=True, exist_ok=True)
        report = destination / 'policy-video/report.json'
        previous = json.loads(report.read_text()) if report.exists() else {}
        if not previous.get('completed') or previous.get('frames') != requested_steps:
            settings = dict(env, __GLX_VENDOR_LIBRARY_NAME='mesa', LIBGL_ALWAYS_SOFTWARE='1', LP_NUM_THREADS='2',
                SOCK_REGISTERED_BATCHMODE='1', SOCK_REGISTERED_PLAYER=str(ROOT / 'Build/SockDressingPlayer/RegisteredLightingV1/Player.x86_64'),
                SOCK_REGISTERED_PLAYER_LOG=str(destination / 'player.log'),
                PYTHONPATH=str(ROOT / '.deps/RCareWorld/pyrcareworld') + ':/opt/ros/noetic/lib/python3/dist-packages')
            with (destination / 'inference.log').open('w') as log:
                subprocess.run(['/tmp/sock-diversity-venv/bin/python', str(ROOT / 'scripts/capture_fixed_torso_initial.py'),
                    '--trajectory', str(ROOT / meta['trajectory']), '--output', str(destination), '--head-cameras',
                    '--lighting-profile', 'balanced', '--policy-model-dir', str(model), '--policy-checkpoint', str(checkpoint),
                    '--policy-episode', str(episode), '--policy-steps', str(requested_steps), '--policy-device', device,
                    '--port', str(20300 + i * 10)],
                    cwd=ROOT, env=settings, stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append(json.loads(report.read_text()))
    result = dict(completed=True, epoch=epoch, requested_device=device,
                  offline_device=json.loads((offline / 'report.json').read_text()).get('device'),
                  offline_report=str(offline / 'report.json'), simulator=records,
                  intermediate=epoch != config['hyperparameters']['num_epochs'])
    videos = []
    offline_rows = json.loads((offline / 'report.json').read_text())['rows']
    for record in offline_rows + records:
        video = Path(record['video'])
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,nb_frames,r_frame_rate', '-of', 'json', str(video)], text=True))['streams'][0]
        if probe['codec_name'] != 'h264' or int(probe['nb_frames']) != record['frames']:
            raise ValueError(f'Video format/frame count mismatch: {video}')
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
        videos.append(dict(path=str(video), frames=record['frames'], decoder_check_passed=True, probe=probe))
    result['video_validation'] = videos
    write(out / 'report.json', result)
    print('VIDEOS COMPLETE', epoch, str(out), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--epoch', type=int);ap.add_argument('--live-steps', type=int, default=260)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--output',type=Path,help='Separate video output directory; requires --epoch')
    a = ap.parse_args();run = a.run_dir.resolve()
    if a.output and not a.epoch:ap.error('--output requires --epoch')
    if a.epoch:generate(run, a.epoch, a.live_steps, a.device, a.output.resolve() if a.output else None);return
    status = run / 'video-pipeline.json';completed = [];started = time.time()
    while True:
        progress_path = run / 'model/progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        target = progress.get('target_epochs', 10000)
        if (run / 'model/parameter.json').exists() and progress.get('completed'):
            candidates = [target]
        else:candidates = []
        for epoch in candidates:
            if epoch in completed:continue
            write(status, dict(completed=False, active_epoch=epoch, training_epoch=progress.get('epoch'), videos_completed=completed))
            try:
                generate(run, epoch, a.live_steps, a.device);completed.append(epoch)
            except Exception as error:
                write(status, dict(completed=False, failed=True, error=repr(error), active_epoch=epoch, videos_completed=completed))
                raise
        if target in completed:
            write(status, dict(completed=True, videos_completed=completed, elapsed_s=time.time() - started));break
        write(status, dict(completed=False, waiting_for_training=True, training_epoch=progress.get('epoch'),
                           target_epochs=target, videos_completed=completed))
        time.sleep(30)


if __name__ == '__main__':main()
