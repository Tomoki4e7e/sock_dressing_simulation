import json
from types import SimpleNamespace

import numpy as np
import pytest

from sock_dressing_simulation.physics_video import PhysicsStepRecorder


def fixture(tmp_path, monkeypatch):
    import sock_dressing_simulation.demo as demo
    written = []
    video = SimpleNamespace(release=lambda: None)
    monkeypatch.setattr(demo, '_open_video', lambda *args: video)
    monkeypatch.setattr(demo, '_write_video_frame', lambda v, rgb, frame: written.append(frame))
    clock = {'fixed_steps': 0, 'last_step_time_s': .02, 'simulated_time_s': 0.}
    cloth = SimpleNamespace(data={}, request_physics_clock=lambda: None)
    camera = SimpleNamespace(data={'rgb': np.zeros((2, 2, 3), dtype=np.uint8)}, GetRGB=lambda *args: None)
    def step(count=1, simulate=True, collect=True):
        if simulate:
            clock['fixed_steps'] += count
            clock['simulated_time_s'] += count * .02
        cloth.data = {'cloth_physics_clock': dict(clock)}
    env = SimpleNamespace(_env=SimpleNamespace(step=step), sock_cloth=cloth,
                          side_recording_camera=camera, _decode=lambda rgb, _: rgb,
                          config={'physics_video_non_destructive_clock': True})
    return env, written, PhysicsStepRecorder(env, tmp_path, lambda: 1)


def test_records_each_existing_step_without_simulation_during_capture(tmp_path, monkeypatch):
    env, written, recorder = fixture(tmp_path, monkeypatch)
    env._env.step(count=3)
    env._env.step(simulate=False)
    assert written == [0, 1, 2]
    assert recorder.clock()['fixed_steps'] == 3
    recorder.close()
    rows = [json.loads(s) for s in (tmp_path / 'physics-video-sync.jsonl').read_text().splitlines()]
    assert all(r['clock_before'] == r['clock_after'] for r in rows)
    assert env._env.step == recorder.original_step


def test_rejects_wrong_physics_timebase(tmp_path, monkeypatch):
    env, _, recorder = fixture(tmp_path, monkeypatch)
    original_clock = recorder.clock
    recorder.clock = lambda: dict(original_clock(), last_step_time_s=.01)
    with pytest.raises(RuntimeError, match='0.02'):
        env._env.step()
    recorder.close()


def test_recording_preserves_camera_response_for_caller(tmp_path, monkeypatch):
    env, _, recorder = fixture(tmp_path, monkeypatch)
    camera = SimpleNamespace(data={})
    env._env.attrs = {1: camera}
    env._env.data = {}
    original = recorder.original_step
    def collect(count=1, simulate=True, collect=True):
        original(count=count, simulate=simulate, collect=collect)
        camera.data = {'rgb': 'requested-image'} if simulate else {}
        env._env.data = {'simulated': simulate}
    recorder.original_step = collect
    env._env.step()
    assert camera.data['rgb'] == 'requested-image'
    assert env._env.data == {'simulated': True}
    recorder.close()
