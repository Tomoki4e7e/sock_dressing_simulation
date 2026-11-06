import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sock_dressing_simulation.reaching_collection import ReachingCollector


def test_camera_cannot_advance_physics(tmp_path):
    source = tmp_path / 'route.json';source.write_text('{}')
    shared = tmp_path / 'shared.yaml';shared.write_text('{}')
    mapping = SimpleNamespace(names=[str(i) for i in range(18)], revolute_action_indices=range(7))
    plan = dict(shared_initial_configuration='shared.yaml', dataset_inputs=[], geometry_record={},
                workbook_pattern={}, applied_opening_stage='A')
    data = dict(inverse_dynamics_available=True, inverse_dynamics_step=100)
    cloth_data = dict(sock_compound_replay=dict(cloth_fixed_steps=0))
    def step(**kwargs):cloth_data['sock_compound_replay']['cloth_fixed_steps'] += 1
    env = SimpleNamespace(robot=SimpleNamespace(data=data, id=1),
        robot_signals=lambda: dict(angle=np.zeros(18), torque=np.ones(18), external_torque=np.zeros(18),
            torque_available=True, torque_source='drive_forces', external_torque_source='measured contact'),
        recording_camera=SimpleNamespace(GetRGB=lambda *args:None),
        sock_cloth=SimpleNamespace(data=cloth_data, _send_data=lambda *args:None),
        _env=SimpleNamespace(step=step))
    with pytest.raises(RuntimeError, match='Camera capture advanced physics'):
        with ReachingCollector(tmp_path / 'episode', mapping, plan, source, tmp_path, {'camera': {'fov': 60}}) as c:
            for i in range(5):
                c.append(env, mapping, dict(step=i, cloth_fixed_steps=10+i, cloth_time_s=(10+i)*.02,
                    stage='reach', stage_time_s=i*.02))
    metadata = json.loads((tmp_path / 'episode/metadata.json').read_text())
    assert not metadata['completed'] and not metadata['reaching_learning_ready']
