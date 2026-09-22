import csv
import json
from pathlib import Path

import numpy as np

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.demo import (
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
        }

    def command(self, command, previous):
        from sock_dressing_simulation.joints import JointMap

        bounded = JointMap.from_config(self.config).bound(command, previous)
        self.angle = bounded
        self.commands.append(bounded)
        return bounded


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
            },
        }
    }

    report = _grasp_frame_report(observation, config)

    assert report["ok"]
    assert report["attached_sides"] == ["left", "right"]
    assert report["maximum_edge_error_m"] == 0.01


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
            }
        ],
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
