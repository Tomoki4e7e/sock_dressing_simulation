import csv
import json
from pathlib import Path

import numpy as np
import pytest

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.demo import (
    _cloth_following_state,
    _cloth_frame_report,
    _grasp_frame_report,
    _reference_action_at_frame,
    _task_success,
    _write_video_frame,
    run_demo,
)
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.perception import PerceptionResult


class _Environment:
    def __init__(self, config):
        self.config = config
        self.angle = np.zeros(18)
        self.commands = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def load(self, *args, **kwargs):
        pass

    def apply_scenario(self, scenario):
        return {"ok": True}

    def observe(self):
        return {
            "camera": {"rgb": np.full((8, 8, 3), 10, np.uint8)},
            "recording_camera": {"rgb": np.full((8, 8, 3), 20, np.uint8)},
            "angle": self.angle.copy(),
            "torque": np.zeros(18),
            "external_torque": np.zeros(18),
            "diagnostics": {},
            "dressing_qa": _passing_dressing_qa(),
        }

    def command(self, command, previous):
        from sock_dressing_simulation.joints import JointMap

        bounded = JointMap.from_config(self.config).bound(command, previous)
        self.angle = bounded
        self.commands.append(bounded)
        return bounded

    def advance_physics(self, steps):
        self.physics_steps = getattr(self, "physics_steps", 0) + int(steps)


class _Perception:
    def __init__(self, config):
        mask = np.zeros((8, 8), bool)
        mask[1, 1] = True
        leg = np.zeros((8, 8), bool)
        leg[2, 3] = True
        depth = np.full((8, 8), 20, np.uint8)
        self.result = PerceptionResult(
            depth, mask, leg, np.where(mask, depth, 0).astype(np.uint8),
            np.where(leg, depth, 0).astype(np.uint8), {"ok": True}
        )

    def initialize(self, *args, **kwargs):
        return self.result

    def track(self, *args, **kwargs):
        return self.result


class _Policy:
    def __init__(self, config, checkpoint=None, device=None):
        self.checkpoint = checkpoint or "fake.pth"
        self.checkpoint_sha256 = "fake"

    def step(self, **kwargs):
        return {"action": np.full(18, 100.0)}


def _passing_dressing_qa(**overrides):
    value = {
        "valid": True,
        "surface_containment_ratio": 0.95,
        "sections": [
            {"name": name, "valid": True, "containment_ratio": 0.875}
            for name in ("toes", "forefoot", "heel", "ankle")
        ],
        "cuff_progress_toward_ankle_m": 0.03,
        "maximum_cuff_reverse_step_m": 0.0,
        "cuff_beyond_distal_toe_m": 0.0,
        "maximum_cloth_foot_penetration_m": 0.001,
    }
    value.update(overrides)
    return value


def test_demo_records_bounded_closed_loop_actions(tmp_path):
    config = load_config()
    config["assets"]["output_dir"] = str(tmp_path)
    result = run_demo(
        config,
        prepared={"runtime_urdf": str(tmp_path / "robot.urdf")},
        output_root=tmp_path / "episodes",
        sock_points=[[1, 1]],
        leg_points=[[3, 2]],
        max_steps=2,
        environment_factory=_Environment,
        perception_factory=_Perception,
        policy_factory=_Policy,
    )
    assert result["ok"]
    episode = result["episode"]
    with open(f"{episode}/applied_action.csv", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 2
    assert float(rows[0][0]) == 0.08
    with open(f"{episode}/angle.csv", newline="") as stream:
        recorded_angles = list(csv.reader(stream))
    assert float(recorded_angles[0][0]) == 0.08
    metadata = json.loads(open(f"{episode}/metadata.json").read())
    assert metadata["stop_reason"] == "max_steps"
    assert metadata["frames"] == 2
    assert metadata["video_camera"]["source"] == "recording_camera"
    assert metadata["video_camera"]["position"] == [-1.8, 1.05, -1.5]
    assert Path(result["video"]).is_file()


def test_reference_actions_are_linearly_interpolated_across_demo_frames():
    actions = np.stack([np.zeros(18), np.ones(18), np.full(18, 2.0)])

    np.testing.assert_allclose(
        _reference_action_at_frame(
            actions,
            frame=1,
            steps=5,
            hold_steps=10,
            interpolation="linear",
        ),
        np.full(18, 0.5),
    )
    np.testing.assert_allclose(
        _reference_action_at_frame(
            actions,
            frame=4,
            steps=5,
            hold_steps=10,
            interpolation="linear",
        ),
        np.full(18, 2.0),
    )


def test_recording_crop_is_resized_to_video_frame_shape():
    class Video:
        def write(self, frame):
            self.frame = frame

    video = Video()
    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[3:9, 4:12] = [255, 0, 0]

    _write_video_frame(video, rgb, 0, crop_xywh=[4, 3, 8, 6])

    assert video.frame.shape == rgb.shape
    assert video.frame[..., 2].mean() > 200


def test_grasp_frame_report_checks_both_attachment_and_opening_edges():
    config = load_config()
    observation = {
        "diagnostics": {
            "grasp_state": [
                {"side": "left", "attached": True},
                {"side": "right", "attached": True},
            ],
            "scene_geometry": {
                "left_grasp_position": [0.0, 0.0, 0.0],
                "left_opening_edge": [0.01, 0.0, 0.0],
                "right_grasp_position": [0.0, 0.1, 0.0],
                "right_opening_edge": [0.0, 0.11, 0.0],
                "opening_span_m": 0.11,
            },
        }
    }

    report = _grasp_frame_report(observation, config)

    assert report["ok"]
    assert report["attached_sides"] == ["left", "right"]
    assert report["maximum_edge_error_m"] == 0.01
    assert report["opening_span_m"] == 0.11


def test_grasp_frame_report_uses_pin_error_when_target_has_local_offset():
    config = load_config()
    observation = {
        "diagnostics": {
            "grasp_state": [
                {"side": "left", "attached": True, "constraint_error": 0.02},
                {"side": "right", "attached": True, "constraint_error": 0.01},
            ],
            "scene_geometry": {
                "left_grasp_position": [0.0, 0.0, 0.0],
                "left_opening_edge": [0.09, 0.0, 0.0],
                "right_grasp_position": [0.0, 0.1, 0.0],
                "right_opening_edge": [0.0, 0.12, 0.0],
            },
        }
    }

    report = _grasp_frame_report(observation, config)

    assert report["ok"]
    assert report["maximum_edge_error_m"] == 0.02
    assert report["target_to_edge_distances_m"]["left"] == 0.09


def test_cloth_frame_report_rejects_distal_end_that_does_not_follow_grasps():
    config = load_config(Path("config/custom_player.yaml"))
    config["scenario"]["sock"]["radial_segments"] = 4
    opening = np.asarray(
        [[-0.04, 0, 0], [0, 0.04, 0], [0.04, 0, 0], [0, -0.04, 0]]
    )
    toe = opening + np.asarray([0, 0, 0.30])
    particles = np.vstack([opening, toe, [[0, 0, 0.30]]])
    edges = np.asarray([[index, (index + 1) % 4] for index in range(4)])
    rest = np.linalg.norm(
        particles[edges[:, 0]] - particles[edges[:, 1]], axis=1
    )

    def observation(grasp_offset, toe_offset, *, foot_contact=False):
        moved = particles.copy()
        moved[:4] += grasp_offset
        moved[-5:] += toe_offset
        return {
            "cloth": {
                "particles": moved.tolist(),
                "particle_edges": edges.tolist(),
                "particle_rest_edge_lengths": rest.tolist(),
            },
            "diagnostics": {
                "scene_geometry": {
                        "left_grasp_position": (
                            np.asarray([-0.04, 0, 0]) + grasp_offset
                        ).tolist(),
                        "right_grasp_position": (
                            np.asarray([0.04, 0, 0]) + grasp_offset
                        ).tolist(),
                }
            },
            "contact_force": (
                [{"collider_id": 2104}] if foot_contact else []
            ),
        }

    baseline_observation = observation(np.zeros(3), np.zeros(3))
    baseline = _cloth_following_state(baseline_observation, config)
    following = _cloth_frame_report(
        observation(np.asarray([0.10, 0, 0]), np.asarray([0.09, 0, 0])),
        config,
        baseline,
    )
    collapsed = _cloth_frame_report(
        observation(np.asarray([0.10, 0, 0]), np.zeros(3)),
        config,
        baseline,
    )
    anchored_on_foot = _cloth_frame_report(
        observation(
            np.asarray([0.10, 0, 0]),
            np.zeros(3),
            foot_contact=True,
        ),
        config,
        baseline,
    )

    assert following["following_ok"]
    assert following["distal_follow_ratio"] == pytest.approx(0.9)
    assert not collapsed["following_ok"]
    assert collapsed["distal_follow_ratio"] == 0.0
    assert anchored_on_foot["following_ok"]
    assert anchored_on_foot["foot_contact_allows_distal_anchoring"]


def test_task_success_rejects_any_intermediate_bimanual_grasp_failure(monkeypatch):
    config = load_config(Path("config/custom_player.yaml"))
    observation = {
        "diagnostics": {
            "grasp_attachments": [
                {"side": "left", "verified": True},
                {"side": "right", "verified": True},
            ]
        },
        "cloth": {},
    }
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )

    report = _task_success(
        observation,
        [{"ok": True}, {"ok": True}],
        [0.0, 0.2],
        config,
        application={"initial_pose_contract": {"ok": True}},
        grasp_quality=[
            {
                "ok": True,
                "attached_grippers": 2,
                "maximum_edge_error_m": 0.01,
            },
            {
                "ok": False,
                "attached_grippers": 1,
                "maximum_edge_error_m": 0.04,
            },
        ],
        rigid_collision_qa_by_frame=[
            {
                "ignored_pair_count": 0,
                "maximum_penetration_m": 0.0,
            }
        ],
    )

    assert not report["success"]
    assert not report["continuous_grasp_ok"]
    assert report["minimum_attached_grippers"] == 1
    assert report["maximum_grasp_edge_error_m"] == 0.04


def test_task_success_requires_observed_foot_contact(monkeypatch):
    config = load_config(Path("config/custom_player.yaml"))
    observation = {
        "diagnostics": {
            "grasp_attachments": [
                {"side": "left", "verified": True},
                {"side": "right", "verified": True},
            ]
        },
        "cloth": {},
    }
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )
    kwargs = {
        "application": {"initial_pose_contract": {"ok": True}},
        "grasp_quality": [
            {
                "ok": True,
                "attached_grippers": 2,
                "maximum_edge_error_m": 0.01,
                "opening_span_m": 0.08,
            }
        ],
        "cloth_quality": [
            {
                "stretch": {
                    "passes": True,
                    "bounds_world": {
                        "minimum": [0.0, 0.0, 0.0],
                        "maximum": [0.1, 0.1, 0.2],
                    },
                },
                "following_ok": True,
            }
        ],
        "rigid_collision_qa_by_frame": [
            {
                "ignored_pair_count": 0,
                "maximum_penetration_m": 0.0,
            }
        ],
        "human_chair_lock_qa_by_frame": [
            {
                "valid": True,
                "right_toe_drift_m": 0.0,
                "chair_drift_m": 0.0,
                "human_root_drift_m": 0.0,
                "human_anchor_drift_m": 0.0,
            }
        ],
        "dressing_quality": [_passing_dressing_qa() for _ in range(3)],
    }

    missing = _task_success(
        observation,
        [{"ok": True}],
        [0.0, 0.2],
        config,
        foot_contact_ids_by_frame=[[]],
        **kwargs,
    )
    touching = _task_success(
        observation,
        [{"ok": True}],
        [0.0, 0.2],
        config,
        foot_contact_ids_by_frame=[[2104]],
        **kwargs,
    )

    assert not missing["success"]
    assert not missing["foot_contact_ok"]
    assert touching["success"]
    assert touching["foot_contact_collider_ids"] == [2104]

    wide_opening = _task_success(
        observation,
        [{"ok": True}],
        [0.0, 0.2],
        config,
        foot_contact_ids_by_frame=[[2104]],
        **{
            **kwargs,
            "grasp_quality": [
                {
                    "ok": True,
                    "attached_grippers": 2,
                    "maximum_edge_error_m": 0.01,
                    "opening_span_m": 0.13,
                }
            ],
        },
    )
    assert not wide_opening["success"]
    assert not wide_opening["continuous_opening_span_ok"]


@pytest.mark.parametrize(
    ("override", "failed_gate"),
    [
        (
            {"surface_containment_ratio": 0.89},
            "final_surface_containment_ok",
        ),
        (
            {
                "sections": [
                    {
                        "name": "toes",
                        "valid": True,
                        "containment_ratio": 0.79,
                    }
                ]
            },
            "final_section_containment_ok",
        ),
        (
            {"maximum_cuff_reverse_step_m": 0.003},
            "cuff_progress_ok",
        ),
        (
            {"cuff_beyond_distal_toe_m": 0.003},
            "cuff_progress_ok",
        ),
        (
            {"maximum_cloth_foot_penetration_m": 0.003},
            "cloth_foot_penetration_ok",
        ),
    ],
)
def test_task_success_rejects_invalid_final_dressing_state(
    monkeypatch, override, failed_gate
):
    config = load_config(Path("config/custom_player.yaml"))
    config["scene"]["initial_pose_contract"]["lock_human_and_chair"] = False
    observation = {
        "diagnostics": {
            "grasp_attachments": [
                {"side": "left", "verified": True},
                {"side": "right", "verified": True},
            ]
        },
        "cloth": {},
    }
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )
    frames = [_passing_dressing_qa() for _ in range(3)]
    frames[-1] = _passing_dressing_qa(**override)
    report = _task_success(
        observation,
        [{"ok": True}],
        [0.0, 0.2],
        config,
        application={"initial_pose_contract": {"ok": True}},
        grasp_quality=[
            {
                "ok": True,
                "attached_grippers": 2,
                "maximum_edge_error_m": 0.01,
                "opening_span_m": 0.08,
            }
        ],
        cloth_quality=[
            {
                "stretch": {
                    "passes": True,
                    "bounds_world": {
                        "minimum": [0.0, 0.0, 0.0],
                        "maximum": [0.1, 0.1, 0.2],
                    },
                },
                "following_ok": True,
            }
        ],
        dressing_quality=frames,
        foot_contact_ids_by_frame=[[2104]],
        rigid_collision_qa_by_frame=[
            {"ignored_pair_count": 0, "maximum_penetration_m": 0.0}
        ],
    )

    assert not report["success"]
    assert not report[failed_gate]


def test_task_success_rejects_robot_human_penetration(monkeypatch):
    config = load_config(Path("config/custom_player.yaml"))
    observation = {
        "diagnostics": {
            "grasp_attachments": [
                {"side": "left", "verified": True},
                {"side": "right", "verified": True},
            ]
        },
        "cloth": {},
    }
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )
    report = _task_success(
        observation,
        [{"ok": True}],
        [0.0, 0.2],
        config,
        application={"initial_pose_contract": {"ok": True}},
        grasp_quality=[
            {
                "ok": True,
                "attached_grippers": 2,
                "maximum_edge_error_m": 0.01,
            }
        ],
        foot_contact_ids_by_frame=[[2104]],
        rigid_collision_qa_by_frame=[
            {
                "ignored_pair_count": 0,
                "maximum_penetration_m": 0.006,
                "maximum_enabled_penetration_m": 0.006,
            }
        ],
    )

    assert not report["success"]
    assert not report["rigid_collision_ok"]
    assert report["maximum_enabled_robot_human_penetration_m"] == pytest.approx(
        0.006
    )


def test_task_success_ignores_disabled_pair_penetration(monkeypatch):
    config = load_config(Path("config/custom_player.yaml"))
    observation = {
        "diagnostics": {
            "grasp_attachments": [
                {"side": "left", "verified": True},
                {"side": "right", "verified": True},
            ]
        },
        "cloth": {},
    }
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )

    report = _task_success(
        observation,
        [{"ok": True}],
        [0.0, 0.2],
        config,
        application={"initial_pose_contract": {"ok": True}},
        grasp_quality=[
            {
                "ok": True,
                "attached_grippers": 2,
                "maximum_edge_error_m": 0.01,
            }
        ],
        rigid_collision_qa_by_frame=[
            {
                "ignored_pair_count": 0,
                "maximum_penetration_m": 0.05,
                "maximum_enabled_penetration_m": 0.001,
                "maximum_ignored_penetration_m": 0.05,
            }
        ],
    )

    assert report["rigid_collision_ok"]
    assert report["maximum_robot_human_penetration_m"] == pytest.approx(0.001)
    assert report[
        "maximum_ignored_robot_human_penetration_m"
    ] == pytest.approx(0.05)
