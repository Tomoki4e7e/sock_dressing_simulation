import json
from types import SimpleNamespace

import numpy as np
import pytest

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.episode import EpisodeWriter


def environment():
    env = SockDressingEnv(load_config())
    env.robot = SimpleNamespace(data={
        "joint_positions": np.zeros(29), "joint_force": np.arange(29, dtype=float),
        "drive_forces": np.ones(29), "gravity_forces": np.ones(29) * 2,
        "coriolis_centrifugal_forces": np.ones(29) * 3,
        "rigid_contact_projection_available": True,
        "rigid_contact_joint_forces": -np.arange(29, dtype=float),
        "rigid_contacts": [{"collider_id": 2105, "force_world_n": [-5, 0, 0]}],
        "rigid_contact_step": 7, "rigid_contact_step_seconds": .01,
    })
    return env


def test_contact_reaches_external_torque_without_double_counting_residual():
    env = environment()
    result = env.robot_signals()
    indices = env.config["joints"]["simulator_indices"]
    np.testing.assert_array_equal(result["external_torque"], -np.arange(29)[indices])
    np.testing.assert_array_equal(result["external_torque_residual"], np.arange(29)[indices] - 6)
    np.testing.assert_array_equal(result["torque"], np.ones(18))
    assert result["rigid_contact"][0]["collider_id"] == 2105
    assert "impulse projection" in result["external_torque_source"]
    # A supported zero-contact sample is a measured zero, not a fallback.
    env.robot.data["rigid_contact_joint_forces"] = np.zeros(29)
    env.robot.data["rigid_contacts"] = []
    np.testing.assert_array_equal(env.robot_signals()["external_torque"], np.zeros(18))


def test_older_player_retains_explicit_residual_fallback():
    env = environment()
    env.robot.data.pop("rigid_contact_projection_available")
    signals = env.robot_signals()
    np.testing.assert_array_equal(signals["external_torque"], signals["external_torque_residual"])
    assert not signals["rigid_contact_force_available"]


def test_required_feedback_rejects_older_player():
    env = environment()
    env.config["scene"]["require_rigid_contact_telemetry"] = True
    env.robot.data["rigid_contact_projection_available"] = False
    with pytest.raises(RuntimeError, match="rebuild"):
        env.robot_signals()


def test_measured_zero_motor_effort_is_available():
    env = environment()
    env.robot.data["drive_forces"] = np.zeros(29)
    env.robot.data["inverse_dynamics_available"] = True
    signals = env.robot_signals()
    assert signals["torque_available"] and signals["torque_source"] == "drive_forces"
    np.testing.assert_array_equal(signals["torque"], np.zeros(18))


def test_feedback_profile_enables_physical_contacts_and_drive_rollout():
    config = load_config("config/rigid_contact_feedback.yaml")
    assert config["scene"]["require_rigid_contact_telemetry"]
    assert not config["scene"]["ignore_robot_human_rigid_collisions"]
    assert config["scene"]["ignore_non_gripper_robot_human_rigid_collisions"]
    assert not config["scene"]["robot_rollout_direct_joint_control"]


@pytest.mark.parametrize("bad", [[], np.full(29, np.nan)])
def test_invalid_advertised_contact_load_is_rejected(bad):
    env = environment()
    env.robot.data["rigid_contact_joint_forces"] = bad
    with pytest.raises(RuntimeError, match="invalid rigid-contact"):
        env.robot_signals()


def test_contact_and_residual_recording_is_frame_aligned(tmp_path):
    env = environment()
    signals = env.robot_signals()
    episode = tmp_path / "episode"
    with EpisodeWriter(episode, env.config["joints"]["names"]) as writer:
        for _ in range(2):
            writer.append(rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                          sock_mask=np.zeros((2, 2), dtype=bool), leg_mask=np.zeros((2, 2), dtype=bool),
                          camera_depth=np.zeros((2, 2), dtype=np.uint8),
                          angle=signals["angle"], torque=signals["torque"], external_torque=signals["external_torque"],
                          robot_telemetry=signals)
    rows = [json.loads(line) for line in (episode / "robot_contact.jsonl").read_text().splitlines()]
    assert [row["frame"] for row in rows] == [0, 1]
    assert rows[0]["rigid_contact_step"] == 7
    assert rows[0]["rigid_contact"][0]["collider_id"] == 2105
    assert rows[0]["external_torque_residual"] == signals["external_torque_residual"].tolist()
