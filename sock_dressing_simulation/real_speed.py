"""Configuration-based approximation of real robot online motion limits."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .joints import JointMap


class LimitedTrajectory:
    """Discrete velocity/acceleration limiter with braking before hard bounds.

    Absolute policy targets may change while moving. Braking can cross a new
    policy target, but never a joint bound. No joint position is teleported.
    """

    def __init__(self, mapping, position, velocity_limit, acceleration_limit, dt):
        self.mapping = mapping
        self.position = mapping.bound(position)
        self.velocity = np.zeros(18)
        self.vmax = np.asarray(velocity_limit, dtype=float)
        self.amax = np.asarray(acceleration_limit, dtype=float)
        self.dt = float(dt)
        if self.dt <= 0 or any(v.shape != (18,) or not np.all(np.isfinite(v)) or np.any(v <= 0)
                               for v in (self.vmax, self.amax)):
            raise ValueError('Invalid trajectory limits')

    def stopping_velocity(self, distance):
        # For the proposed next velocity, include this step and every subsequent
        # maximally decelerated step. Bisection inverts the discrete stop distance.
        lo = np.zeros(18)
        hi = self.vmax.copy()
        for _ in range(45):
            v = (lo + hi) / 2
            n = np.ceil(v / (self.amax * self.dt))
            stopping = self.dt * (n * v - self.amax * self.dt * n * (n - 1) / 2)
            valid = stopping <= np.maximum(0, distance)
            lo = np.where(valid, v, lo)
            hi = np.where(valid, hi, v)
        return lo

    def step(self, target):
        target = self.mapping.bound(target)
        error = target - self.position
        desired = np.sign(error) * self.stopping_velocity(np.abs(error))
        old_velocity = self.velocity.copy()
        velocity = np.clip(desired, old_velocity - self.amax * self.dt,
                           old_velocity + self.amax * self.dt)
        velocity = np.clip(velocity,
                           -self.stopping_velocity(self.position - self.mapping.lower),
                           self.stopping_velocity(self.mapping.upper - self.position))
        position = self.mapping.bound(self.position + velocity * self.dt)
        velocity = (position - self.position) / self.dt
        acceleration = (velocity - old_velocity) / self.dt
        if np.any(np.abs(velocity) > self.vmax + 1e-9) or np.any(np.abs(acceleration) > self.amax + 1e-7):
            raise RuntimeError('Trajectory velocity/acceleration bound failed')
        self.position, self.velocity = position, velocity
        return position.copy(), velocity.copy(), acceleration


def urdf_limits(mapping, urdf, settings):
    elements = {j.attrib['name']: j for j in ET.parse(urdf).getroot().findall('joint')}
    vmax, amax = np.zeros(18), np.zeros(18)
    for i, name in enumerate(mapping.names):
        if name in mapping.mimic:
            continue
        limit = elements[name].find('limit')
        override = float(settings['arm_speed_override'] if i in mapping.revolute_action_indices
                         else settings['gripper_speed_override'])
        if not 0 < override <= 1:
            raise ValueError('Speed override must be in (0, 1]')
        vmax[i] = float(limit.attrib['velocity']) * override
        amax[i] = float(limit.attrib['acceleration'])
    for name, rule in mapping.mimic.items():
        i, source = mapping.names.index(name), mapping.names.index(rule['source'])
        scale = abs(float(rule['multiplier']))
        vmax[i], amax[i] = vmax[source] * scale, amax[source] * scale
    return vmax, amax


class SpeedMatchedRollout:
    def __init__(self, environment, directory, command_index, urdf):
        self.env = environment
        self.directory = Path(directory)
        self.command_index = command_index
        self.settings = environment.config['real_speed']
        self.mapping = JointMap.from_config(environment.config)
        self.dt = float(self.settings['fixed_step_s'])
        interval = float(self.settings['command_interval_s'])
        if not np.isclose(interval, environment.config['inference']['physics_steps_per_action'] * self.dt):
            raise ValueError('Command interval does not match physical steps')
        self.vmax, self.amax = urdf_limits(self.mapping, urdf, self.settings)
        self.trajectory = LimitedTrajectory(self.mapping, environment.robot_signals()['angle'],
                                            self.vmax, self.amax, self.dt)
        self.target = self.trajectory.position.copy()
        self.steps = 0
        self.previous_actual = self.trajectory.position.copy()
        robot = environment.robot
        full = np.full(len(robot.data['joint_positions']), 100.)
        stiffness = np.asarray(robot.data['joint_stiffness'], dtype=float).copy()
        damping = np.asarray(robot.data['joint_damping'], dtype=float).copy()
        margin = float(self.settings['drive_velocity_margin'])
        if not 0 < margin <= 1:
            raise ValueError('Invalid drive velocity margin')
        for i, name in enumerate(self.mapping.names):
            if name not in self.mapping.mimic:
                full[self.mapping.simulator_indices[i]] = self.vmax[i] * margin
                stiffness[self.mapping.simulator_indices[i]] = float(self.settings['drive_stiffness'])
                damping[self.mapping.simulator_indices[i]] = float(self.settings['drive_damping'])
        robot._send_data('SetJointMaximumVelocity', full.tolist())
        robot._send_data('SetJointStiffness', stiffness.tolist())
        robot._send_data('SetJointDamping', damping.tolist())
        stop_indices = [int(self.mapping.simulator_indices[i]) for i, name in enumerate(self.mapping.names)
                        if i not in self.mapping.revolute_action_indices and name not in self.mapping.mimic]
        if self.settings.get('reconcile_prismatic_stop_velocity', False):
            robot._send_data('ConfigurePrismaticStopVelocityGuard', stop_indices)
        environment._env.step(simulate=False)
        if not np.allclose(robot.data.get('joint_max_velocities', []), full, rtol=1e-5):
            raise RuntimeError('Native joint velocity limit readback failed')
        if not np.allclose(robot.data['joint_stiffness'], stiffness) or not np.allclose(robot.data['joint_damping'], damping):
            raise RuntimeError('Native drive gain readback failed')
        if self.settings.get('reconcile_prismatic_stop_velocity', False) and robot.data.get(
                'prismatic_stop_velocity_guard', {}).get('indices') != stop_indices:
            raise RuntimeError('Native hard-stop velocity guard readback failed')
        self.drive_limits = full
        self.stream = (self.directory / 'joint-motion.jsonl').open('w')
        (self.directory / 'speed-limits.json').write_text(json.dumps(dict(
            basis=self.settings['basis'], names=self.mapping.names, settings=self.settings,
            urdf=str(urdf), velocity_limit=self.vmax.tolist(), acceleration_limit=self.amax.tolist(),
            native_drive_limits=full.tolist(), native_drive_stiffness=stiffness.tolist(),
            native_drive_damping=damping.tolist(), units='rad/s and rad/s^2; prismatic m/s and m/s^2'), indent=2)+'\n')

    def command(self, target):
        self.target = self.mapping.bound(target)
        return self.step()

    def step(self):
        position, velocity, acceleration = self.trajectory.step(self.target)
        robot = self.env.robot
        full = self.mapping.merge_for_simulator(position, robot.data['joint_positions'])
        robot.SetJointPosition(full.tolist())
        self.env._env.step()
        actual = self.env.robot_signals()['angle']
        measured_velocity = np.asarray(robot.data['joint_velocities'], dtype=float)
        actual_velocity = measured_velocity[self.mapping.simulator_indices].copy()
        actual_velocity[list(self.mapping.revolute_action_indices)] = np.deg2rad(
            actual_velocity[list(self.mapping.revolute_action_indices)])
        for name, rule in self.mapping.mimic.items():
            actual_velocity[self.mapping.names.index(name)] = actual_velocity[self.mapping.names.index(rule['source'])] * rule['multiplier']
        displacement_velocity = (actual - self.previous_actual) / self.dt
        self.previous_actual = actual.copy()
        self.steps += 1
        clock = self.env.sock_cloth.data.get('cloth_physics_clock')
        if clock is None:
            self.env.sock_cloth.request_physics_clock()
            self.env._env.step(simulate=False)
            clock = self.env.sock_cloth.data['cloth_physics_clock']
        self.stream.write(json.dumps(dict(step=self.steps, command=self.command_index(),
            clock=clock, policy_target=self.target.tolist(), drive_target=position.tolist(),
            target_velocity=velocity.tolist(), target_acceleration=acceleration.tolist(),
            actual_position=actual.tolist(), actual_velocity=actual_velocity.tolist(),
            prismatic_stop_velocity_guard=robot.data.get('prismatic_stop_velocity_guard'),
            displacement_velocity=displacement_velocity.tolist()))+'\n')
        self.stream.flush()
        if np.any(np.abs(actual_velocity) > self.vmax + 1e-5) or np.any(np.abs(displacement_velocity) > self.vmax + 1e-5):
            raise RuntimeError('Measured joint speed exceeded URDF*override limit; see joint-motion.jsonl')
        return position

    def close(self):
        self.stream.close()
