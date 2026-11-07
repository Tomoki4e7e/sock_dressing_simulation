"""Live observation -> model command -> physics feedback, recorded to MP4."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from .head_cameras import capture_head_cameras


def run_policy_video(env, mapping, episode, model_dir, checkpoint, output, steps=260):
    root = Path(__file__).resolve().parents[1];output = Path(output);output.mkdir(parents=True, exist_ok=True)
    episode = Path(episode);model_dir = Path(model_dir);checkpoint = Path(checkpoint)
    worker_log = (output / 'worker.log').open('w')
    worker = subprocess.Popen(['/usr/bin/python3', str(root / 'scripts/reaching_policy_worker.py'),
        '--model-dir', str(model_dir), '--checkpoint', str(checkpoint), '--episode', str(episode)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log, text=True, bufsize=1)
    video_path = output / 'closed-loop-robot.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1600x900', '-r', '5', '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path)], stdin=subprocess.PIPE)
    frames = 0;trace = [];started = time.monotonic();failure = None
    angular = list(mapping.revolute_action_indices)
    def label(image, text, point, size=.65):
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, (230, 235, 245), 1, cv2.LINE_AA)
    try:
        ready = worker.stdout.readline()
        if not ready:raise RuntimeError('ML worker failed to initialize; see worker.log')
        epoch = json.loads(ready)['epoch']
        full = np.array(env.robot.data['joint_positions'], dtype=float)
        fixed = full[mapping.fixed_simulator_indices].copy()
        # This player records drive efforts without gravity, as in the source
        # corpus. The ML worker adds the same calibrated static gravity term.
        env.robot.SetJointUseGravity(False);env._env.step(simulate=False)
        for step in range(steps):
            env.robot.GetJointInverseDynamicsForce();env._env.step(simulate=False)
            signals = env.robot_signals();q = np.array(signals['angle'])
            if not signals['torque_available']:raise RuntimeError('Live drive effort is unavailable')
            env.recording_camera.GetRGB(1280, 960, 92.)
            views, poses = capture_head_cameras(env, 92.)
            overview = env._decode(env.recording_camera.data['rgb'], 'RGB')
            rgb_path = output / 'current-rgb.png';predicted_path = output / 'current-predicted.png'
            Image.fromarray(views['right']).save(rgb_path)
            request = dict(index=step, rgb=str(rgb_path), predicted_image=str(predicted_path), angle=q.tolist(), torque=signals['torque'].tolist())
            worker.stdin.write(json.dumps(request) + '\n');worker.stdin.flush()
            response = worker.stdout.readline()
            if not response:raise RuntimeError('ML worker stopped; see worker.log')
            prediction = json.loads(response);proposed = np.array(prediction['action'])
            command = mapping.bound(proposed, q)
            full = np.array(env.robot.data['joint_positions'], dtype=float)
            target = mapping.merge_for_simulator(command, full)
            target[mapping.fixed_simulator_indices] = fixed
            # Motors execute the prediction for ten 20-ms physics steps.
            # No trajectory replay or teacher joint angles are used here.
            for _ in range(10):
                env.robot.SetJointPosition(target.tolist())
                env.sock_cloth._send_data('SetSockReplayDriveVelocities', env.robot.id,
                                         json.dumps(dict(values=np.zeros(len(target)).tolist())))
                env._env.Simulate(.02, 1);env._env.step(simulate=False)
            actual = mapping.from_simulator(env.robot.data['joint_positions'])
            if not np.isfinite(actual).all():raise RuntimeError('Non-finite simulator feedback')
            canvas = np.full((900, 1600, 3), (19, 26, 38), dtype='uint8')
            label(canvas, f'Closed-loop simulated robot | epoch {epoch} | {episode.name}', (20, 30), .7)
            label(canvas, f't={step * .2:.1f}s | live RGB + SAM2/depth + measured joints -> model -> motors', (20, 60), .6)
            canvas[110:710, :800] = cv2.resize(views['right'], (800, 600))
            canvas[110:710, 800:] = cv2.resize(overview, (800, 600))
            label(canvas, 'Live head camera (input before command)', (20, 95))
            label(canvas, 'Live overview (input before command)', (820, 95))
            label(canvas, 'Frozen-cloth reaching scene; joint limits and command-rate limits applied', (20, 750), .65)
            limited = float(np.max(abs(command - proposed)))
            label(canvas, f'Max commanded/measured arm error: {np.max(abs(command[angular] - actual[angular])):.4f} rad', (20, 790), .6)
            label(canvas, f'Model output limited by {limited:.4f} (mixed joint units); input outside training range: {prediction["input_out_of_range_fraction"]:.0%}', (20, 825), .6)
            label(canvas, 'Rendering is simulation feedback. No recorded motion is replayed.', (20, 865), .6)
            encoder.stdin.write(canvas.tobytes());frames += 1
            if step == 0 or step == steps // 2:Image.fromarray(canvas).save(output / f'preview-{step}.jpg')
            trace.append(dict(step=step, time_s=step * .2, observed_angle=q.tolist(), prediction=prediction,
                              executed_command=command.tolist(), feedback_angle=actual.tolist(), command_limited_max=limited))
            (output / 'progress.json').write_text(json.dumps(dict(step=step + 1, steps=steps, elapsed_s=time.monotonic() - started)) + '\n')
            print('LIVE POLICY', step + 1, '/', steps, flush=True)
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        if worker.poll() is None:
            try:
                worker.stdin.write(json.dumps(dict(quit=True)) + '\n');worker.stdin.flush();worker.stdin.close()
            except BrokenPipeError:pass
            try:worker.wait(timeout=15)
            except subprocess.TimeoutExpired:worker.terminate();worker.wait(timeout=10)
        worker_log.close();encoder.stdin.close();returncode = encoder.wait()
        (output / 'motion-trace.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in trace))
        report = dict(completed=failure is None and returncode == 0, failure=failure, frames=frames, fps=5,
                      video=str(video_path), checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      live_observations=True, teacher_trajectory_used=False, motor_execution=True,
                      scene_scope='Frozen cloth; no cloth retention or pull-up success validation',
                      torque_scope='Instantaneous measured gravity-disabled drive effort plus calibrated static gravity; as real online control, no future samples used',
                      elapsed_s=time.monotonic() - started)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if returncode:raise RuntimeError('Closed-loop video encoding failed')
    return report
