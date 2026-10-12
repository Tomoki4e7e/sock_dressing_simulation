import numpy as np
import pytest
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.joints import JointMap
from sock_dressing_simulation.real_speed import LimitedTrajectory, SpeedMatchedRollout, urdf_limits


def fixture(tmp_path):
    config = load_config('config/real_speed_autonomous40.yaml')
    mapping = JointMap.from_config(config)
    root = ET.Element('robot')
    for i, name in enumerate(mapping.names):
        joint = ET.SubElement(root, 'joint', name=name)
        ET.SubElement(joint, 'limit', velocity=str(np.deg2rad(150) if i in mapping.revolute_action_indices else .3),
                      acceleration=str(20 if i in mapping.revolute_action_indices else 2))
    urdf = tmp_path/'robot.urdf'
    ET.ElementTree(root).write(urdf)
    vmax, amax = urdf_limits(mapping, urdf, config['real_speed'])
    position = mapping.bound((mapping.lower + mapping.upper) / 2)
    return mapping, vmax, amax, LimitedTrajectory(mapping, position, vmax, amax, .02)


def test_override_units_and_mimic_limits(tmp_path):
    mapping, vmax, amax, _ = fixture(tmp_path)
    assert np.isclose(vmax[0], np.deg2rad(150) * .25)
    for name, rule in mapping.mimic.items():
        target, source = mapping.names.index(name), mapping.names.index(rule['source'])
        assert np.isclose(vmax[target], vmax[source] * abs(rule['multiplier']))
        assert np.isclose(amax[target], amax[source] * abs(rule['multiplier']))


def test_target_reversals_and_joint_bound_braking(tmp_path):
    mapping, vmax, amax, trajectory = fixture(tmp_path)
    prior_velocity = np.zeros(18)
    for step in range(800):
        target = mapping.upper if (step // 13) % 2 else mapping.lower
        q, velocity, acceleration = trajectory.step(target)
        assert np.all(q >= mapping.lower - 1e-10)
        assert np.all(q <= mapping.upper + 1e-10)
        assert np.all(np.abs(velocity) <= vmax + 1e-9)
        assert np.all(np.abs((velocity - prior_velocity) / .02) <= amax + 1e-7)
        for name, rule in mapping.mimic.items():
            assert np.isclose(q[mapping.names.index(name)],
                              q[mapping.names.index(rule['source'])] * rule['multiplier'] + rule['offset'])
        prior_velocity = velocity


def test_reaches_stationary_target_without_max_delta_clipping(tmp_path):
    mapping, vmax, amax, trajectory = fixture(tmp_path)
    target = mapping.bound(trajectory.position + .2)
    for _ in range(500):
        trajectory.step(target)
    assert np.allclose(trajectory.position, target, atol=1e-8)
    assert np.allclose(trajectory.velocity, 0, atol=1e-8)


@pytest.mark.parametrize('velocity_multiplier', [1., 2.])
def test_ten_substeps_use_drive_targets_and_reject_overspeed(tmp_path, velocity_multiplier):
    mapping, _, _, trajectory = fixture(tmp_path)
    config = load_config('config/real_speed_autonomous40.yaml')
    state = mapping.merge_for_simulator(trajectory.position, np.zeros(29))
    robot = SimpleNamespace(data={'joint_positions': state.tolist(), 'joint_velocities': [0.]*29,
                                 'joint_stiffness': [100000.]*29, 'joint_damping': [9000.]*29})
    target = [state.copy()]
    sent = []
    clock = {'fixed_steps': 0, 'simulated_time_s': 0., 'last_step_time_s': .02}
    cloth = SimpleNamespace(data={}, request_physics_clock=lambda: None)
    def send(command, values):
        sent.append(command)
        if command == 'ConfigurePrismaticStopVelocityGuard':
            robot.data['prismatic_stop_velocity_guard'] = {'indices': values, 'corrections': []}
            return
        field = {'SetJointMaximumVelocity': 'joint_max_velocities', 'SetJointStiffness': 'joint_stiffness',
                 'SetJointDamping': 'joint_damping'}[command]
        robot.data[field] = values
    def step(simulate=True):
        if simulate:
            old = np.asarray(robot.data['joint_positions'])
            robot.data['joint_positions'] = target[0].tolist()
            robot.data['joint_velocities'] = ((target[0]-old)/.02*velocity_multiplier).tolist()
            clock['fixed_steps'] += 1
            clock['simulated_time_s'] += .02
        cloth.data = {'cloth_physics_clock': dict(clock)}
    robot._send_data = send
    robot.SetJointPosition = lambda values: target.__setitem__(0, np.asarray(values))
    env = SimpleNamespace(config=config, robot=robot, sock_cloth=cloth,
        _env=SimpleNamespace(step=step),
        robot_signals=lambda: {'angle': mapping.from_simulator(robot.data['joint_positions'])})
    rollout = SpeedMatchedRollout(env, tmp_path, lambda: 1, tmp_path/'robot.urdf')
    assert clock['fixed_steps'] == 0
    if velocity_multiplier > 1:
        with pytest.raises(RuntimeError, match='Measured joint speed exceeded'):
            rollout.command(mapping.upper)
        rollout.close()
        assert len((tmp_path/'joint-motion.jsonl').read_text().splitlines()) == 1
        return
    rollout.command(mapping.upper)
    for _ in range(9):
        rollout.step()
    rollout.close()
    assert sent == ['SetJointMaximumVelocity', 'SetJointStiffness', 'SetJointDamping', 'ConfigurePrismaticStopVelocityGuard']
    assert clock['fixed_steps'] == 10
    assert np.isclose(clock['simulated_time_s'], .2)
    import json
    rows = [json.loads(line) for line in (tmp_path/'joint-motion.jsonl').read_text().splitlines()]
    assert len(rows) == 10
    assert np.allclose(rows[-1]['actual_velocity'], rows[-1]['target_velocity'])
