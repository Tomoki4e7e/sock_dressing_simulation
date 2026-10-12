"""Record actual fixed steps without adding simulated time."""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path


class PhysicsStepRecorder:
    def __init__(self, environment, directory, command_index):
        self.environment = environment
        self.directory = Path(directory)
        self.command_index = command_index
        self.original_step = environment._env.step
        self.video = None
        self.frames = 0
        self.first_clock = None
        self.last_clock = None
        self.non_destructive_clock = bool(environment.config.get('physics_video_non_destructive_clock', False))
        source = Path(__file__)
        shutil.copyfile(source, self.directory / 'physics-video-source.py')
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        environment._env.step = self.step

    def clock(self):
        if self.non_destructive_clock:
            self.environment.sock_cloth.request_physics_clock()
        else:
            self.environment.sock_cloth.request_dressing_qa()
        self.original_step(simulate=False)
        if self.non_destructive_clock:
            return dict(self.environment.sock_cloth.data['cloth_physics_clock'])
        return dict(self.environment.sock_cloth.data['dressing_qa']['foot_collision_safety']['cloth_physics_clock'])

    def step(self, count=1, simulate=True, collect=True):
        if not simulate:
            return self.original_step(count=count, simulate=False, collect=collect)
        result = None
        for _ in range(count):
            result = self.original_step(count=1, simulate=True, collect=collect)
            # Collect replaces every Attr.data, including a camera response
            # that the caller has just requested. Preserve that response while
            # the recorder performs its own read-only communication rounds.
            native = self.environment._env
            saved = [(attr, attr.data) for attr in getattr(native, 'attrs', {}).values()]
            saved_environment = getattr(native, 'data', None)
            try:
                self.capture()
            finally:
                for attr, data in saved:
                    attr.data = data
                if saved_environment is not None:
                    native.data = saved_environment
        return result

    def capture(self):
        before = self.clock()
        camera = self.environment.side_recording_camera
        if camera is None:
            raise RuntimeError('Physics recording requires the side camera')
        camera.GetRGB(960, 720, 60)
        self.original_step(simulate=False)
        rgb = self.environment._decode(camera.data['rgb'], 'RGB')
        after = self.clock()
        if before != after:
            raise RuntimeError('Physics video capture advanced simulated time')
        dt = float(before['last_step_time_s'])
        if abs(dt - .02) > 1e-7:
            raise RuntimeError('50 fps physics video requires 0.02 s fixed steps')
        if self.last_clock is not None and before['fixed_steps'] != self.last_clock['fixed_steps'] + 1:
            raise RuntimeError('Physics video missed a fixed step')
        from .demo import _open_video, _write_video_frame
        if self.video is None:
            self.video = _open_video(self.directory / 'side_physics_50fps.mp4', rgb.shape, 50.)
            self.first_clock = before
        _write_video_frame(self.video, rgb, self.frames)
        with (self.directory / 'physics-video-sync.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(frame=self.frames,
                command=self.command_index(), clock_before=before, clock_after=after)) + '\n')
        self.frames += 1
        self.last_clock = before

    def close(self):
        self.environment._env.step = self.original_step
        if self.video is not None:
            self.video.release()
        (self.directory / 'physics-video.json').write_text(json.dumps(dict(
            frames=self.frames, fps=50, first_clock=self.first_clock,
            last_clock=self.last_clock, additional_simulated_steps=0,
            non_destructive_clock=self.non_destructive_clock,
            source_sha256=self.source_sha256), indent=2) + '\n')
