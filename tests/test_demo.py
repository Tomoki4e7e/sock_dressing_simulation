import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.demo import (
    _cloth_following_state,
    _cloth_frame_report,
    _contact_rebound_report,
    _grasp_frame_report,
    _reference_action_at_frame,
    _sock_tip_geometry_report,
    _task_success,
    _tip_drape_sample,
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

    def stabilize_cloth_constraints(self):
        self.stabilizations = getattr(self, "stabilizations", 0) + 1


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


def test_cartesian_projection_settles_seam_before_observation():
    source = Path("sock_dressing_simulation/demo.py").read_text()
    projection = source.split(
        "alignment = environment.move_grippers_to_targets(", 1
    )[1].split("observation = environment.observe()", 1)[0]

    assert "environment.stabilize_cloth_constraints()" in projection


def test_sock_tip_geometry_report_preserves_world_y_and_loop_offsets():
    report = _sock_tip_geometry_report(
        7,
        {
            "sock_tip_center": [0.1, 0.42, 0.3],
            "opening_center": [0.0, 0.5, 0.2],
            "sock_tip_span_axis_offset_m": 0.04,
            "sock_tip_cross_axis_offset_m": 0.14,
            "sock_tip_opening_depth_m": 0.09,
            "opening_ring_maximum_sag_m": 0.001,
            "opening_ring_area_retention": 0.96,
            "opening_target_normal_alignment": 1.0,
            "opening_ring_target_alignment": 0.99,
        },
    )

    assert report["frame"] == 7
    assert report["sock_tip_world_y_m"] == pytest.approx(0.42)
    assert report["opening_center_world_y_m"] == pytest.approx(0.5)
    assert report["sock_tip_cross_axis_offset_m"] == pytest.approx(0.14)
    assert report["sock_tip_opening_depth_m"] == pytest.approx(0.09)


class _Policy:
    def __init__(self, config, checkpoint=None, device=None):
        self.checkpoint = checkpoint or "fake.pth"
        self.checkpoint_sha256 = "fake"

    def step(self, **kwargs):
        return {"action": np.full(18, 100.0)}


def _tip_drape_observation(
    tip_y=0.55,
    span_offset=0.0,
    cross_offset=-0.20,
    opening_depth=-0.02,
    opening_y=0.5,
):
    return {
        "camera": {
            "rgb": np.full((8, 8, 3), 10, np.uint8),
            "sock_mask": np.eye(8, dtype=bool),
            "leg_mask": np.fliplr(np.eye(8, dtype=bool)),
        },
        "recording_camera": {"rgb": np.full((8, 8, 3), 20, np.uint8)},
        "angle": np.zeros(18),
        "torque": np.zeros(18),
        "external_torque": np.zeros(18),
        "cloth": {},
        "diagnostics": {
            "grasp_state": [
                {
                    "side": side,
                    "attached": True,
                    "constraint_error": 0.0,
                }
                for side in ("left", "right")
            ],
            "scene_geometry": {
                "sock_tip_center": [0.0, tip_y, 0.0],
                "opening_center": [0.0, opening_y, 0.0],
                "left_grasp_position": [-0.055, opening_y, 0.0],
                "right_grasp_position": [0.055, opening_y, 0.0],
                "opening_target_normal": [0.0, 0.0, 1.0],
                "sock_tip_span_axis_offset_m": span_offset,
                "sock_tip_cross_axis_offset_m": cross_offset,
                "sock_tip_opening_depth_m": opening_depth,
                "opening_span_m": 0.11,
                "opening_ring_maximum_sag_m": 0.001,
                "opening_ring_area_retention": 0.95,
                "opening_target_normal_alignment": 1.0,
                "opening_ring_target_alignment": 1.0,
                "sock_body_gravity_alignment": 0.95,
            },
        },
        "dressing_qa": _passing_dressing_qa(),
    }


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
        "obi_maximum_cloth_foot_penetration_m": 0.0008,
        "geometric_maximum_cloth_foot_penetration_m": 0.0006,
    }
    value.update(overrides)
    return value


def test_tip_drape_sample_requires_drop_centered_span_and_physics_qa(monkeypatch):
    config = load_config(Path("config/autonomous_real_only_plate_normal_taut_rim.yaml"))
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )

    sample = _tip_drape_sample(
        4,
        _tip_drape_observation(
            tip_y=0.45, span_offset=0.06, opening_y=0.50
        ),
        config,
        release_y=0.52,
        previous_y=0.46,
        settings=config["inference"]["tip_drape_wait"],
    )

    assert sample["ok"]
    assert sample["tip_drop_m"] == pytest.approx(0.07)
    assert sample["tip_below_opening_m"] == pytest.approx(0.05)
    assert sample["span_limit_m"] == pytest.approx(0.065)
    assert sample["grasp_ok"]
    assert sample["rim_ok"]
    assert sample["stretch_ok"]
    assert sample["cross_ok"]
    assert sample["depth_ok"]
    assert sample["gravity_alignment_ok"]

    outside = _tip_drape_sample(
        5,
        _tip_drape_observation(
            tip_y=0.44,
            span_offset=0.01,
            cross_offset=0.02,
            opening_depth=0.09,
            opening_y=0.50,
        ),
        config,
        release_y=0.52,
        previous_y=0.45,
        settings=config["inference"]["tip_drape_wait"],
    )
    assert not outside["ok"]
    assert not outside["cross_ok"]
    assert not outside["depth_ok"]


def test_tip_drape_sample_can_report_stretch_without_gating(monkeypatch):
    config = load_config(
        Path("config/autonomous_real_only_plate_normal_taut_rim.yaml")
    )
    settings = dict(config["inference"]["tip_drape_wait"])
    settings["require_stretch"] = False
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": False}),
    )

    sample = _tip_drape_sample(
        4,
        _tip_drape_observation(
            tip_y=0.45, span_offset=0.06, opening_y=0.50
        ),
        config,
        release_y=0.52,
        previous_y=0.46,
        settings=settings,
    )

    assert sample["ok"]
    assert not sample["stretch_required"]
    assert not sample["stretch_ok"]
    assert sample["stretch_gate_ok"]


def test_tip_drape_sample_can_use_gravity_aligned_frame(monkeypatch):
    config = load_config(Path("config/autonomous_real_only_plate_normal_taut_rim.yaml"))
    settings = dict(config["inference"]["tip_drape_wait"])
    settings["coordinate_frame"] = "gravity_aligned"
    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )

    sample = _tip_drape_sample(
        4,
        _tip_drape_observation(
            tip_y=0.45,
            span_offset=999.0,
            cross_offset=999.0,
            opening_depth=999.0,
            opening_y=0.65,
        ),
        config,
        release_y=0.52,
        previous_y=0.46,
        settings=settings,
    )

    assert sample["ok"]
    assert sample["coordinate_frame"] == "gravity_aligned"
    assert sample["sock_tip_span_axis_offset_m"] == pytest.approx(0.0)
    assert sample["sock_tip_cross_axis_offset_m"] == pytest.approx(-0.20)
    assert sample["sock_tip_opening_depth_m"] == pytest.approx(0.0)
    assert sample["sock_body_gravity_alignment"] == pytest.approx(0.95)


def test_demo_waits_for_tip_drape_before_policy_inference(tmp_path, monkeypatch):
    config = load_config(Path("config/autonomous_real_only_plate_normal_taut_rim.yaml"))
    config["assets"]["output_dir"] = str(tmp_path)
    config["inference"]["recording_crop_xywh"] = None
    config["inference"]["tip_drape_wait"].update(
        minimum_tip_drop_m=0.01,
        consecutive_steps=2,
        maximum_steps=5,
    )
    events = []

    class Environment(_Environment):
        def __init__(self, config):
            super().__init__(config)
            self.tip_y = 0.65
            self.released = False

        def release_initial_tip_guidance(self):
            events.append("release")
            self.released = True

        def observe(self):
            if self.released:
                self.tip_y -= 0.02
                events.append("wait_observe")
            return _tip_drape_observation(tip_y=self.tip_y, opening_y=0.70)

    class Policy(_Policy):
        def step(self, **kwargs):
            events.append("policy")
            return super().step(**kwargs)

    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )
    result = run_demo(
        config,
        prepared={"runtime_urdf": str(tmp_path / "robot.urdf")},
        output_root=tmp_path / "episodes",
        sock_points=[[1, 1]],
        leg_points=[[3, 2]],
        max_steps=1,
        environment_factory=Environment,
        perception_factory=_Perception,
        policy_factory=Policy,
    )

    assert result["ok"]
    assert events.index("release") < events.index("policy")
    assert events.count("wait_observe") >= 2
    metadata = json.loads(Path(result["episode"], "metadata.json").read_text())
    assert metadata["tip_drape_wait"]["passed"]
    assert metadata["tip_drape_wait"]["reason"] == "conditions_met"
    assert Path(result["tip_drape_video"]).is_file()


def test_demo_does_not_run_policy_when_tip_drape_times_out(tmp_path, monkeypatch):
    config = load_config(Path("config/autonomous_real_only_plate_normal_taut_rim.yaml"))
    config["assets"]["output_dir"] = str(tmp_path)
    config["inference"]["recording_crop_xywh"] = None
    config["inference"]["tip_drape_wait"].update(
        minimum_tip_drop_m=0.05,
        consecutive_steps=2,
        maximum_steps=2,
    )
    policy_calls = []

    class Environment(_Environment):
        def release_initial_tip_guidance(self):
            pass

        def observe(self):
            return _tip_drape_observation(tip_y=0.65)

    class Policy(_Policy):
        def step(self, **kwargs):
            policy_calls.append(kwargs)
            return super().step(**kwargs)

    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )
    result = run_demo(
        config,
        prepared={"runtime_urdf": str(tmp_path / "robot.urdf")},
        output_root=tmp_path / "episodes",
        sock_points=[[1, 1]],
        leg_points=[[3, 2]],
        max_steps=1,
        environment_factory=Environment,
        perception_factory=_Perception,
        policy_factory=Policy,
    )

    assert not result["ok"]
    assert result["frames"] == 0
    assert result["stop_reason"].startswith("fail_closed: tip drape wait timed out")
    assert policy_calls == []
    metadata = json.loads(Path(result["episode"], "metadata.json").read_text())
    assert not metadata["tip_drape_wait"]["passed"]
    assert metadata["tip_drape_wait"]["reason"] == "timeout"


@pytest.mark.parametrize(
    ("alignment", "expected_ok"),
    [(1.0, True), (0.5, False)],
)
def test_demo_prepares_signed_drape_before_policy_and_fails_closed(
    tmp_path, monkeypatch, alignment, expected_ok
):
    config = load_config(
        Path(
            "config/"
            "autonomous_real_only_opening_reverse_270deg_positive_180_drape.yaml"
        )
    )
    config["assets"]["output_dir"] = str(tmp_path)
    config["inference"]["recording_crop_xywh"] = None
    config["inference"]["tip_drape_wait"]["consecutive_steps"] = 1
    config["inference"]["pre_inference_drape"].update(
        rotation_steps=2,
        settle_steps=2,
    )
    config["inference"]["post_pre_drape_prompt_mode"] = "single_centroid"
    events = []

    @dataclass
    class DrapeGeometry:
        signed_opening_span_rotation_degrees: float
        opening_target_normal_alignment: float
        opening_ring_target_alignment: float
        opening_body_barrier_maximum_penetration_m: float = 0.0
        opening_body_barrier_violation_count: int = 0

    class Environment(_Environment):
        def __init__(self, environment_config):
            super().__init__(environment_config)
            self.rotation = 0.0

        def release_initial_tip_guidance(self):
            events.append("release")

        def observe(self):
            events.append("observe")
            return _tip_drape_observation(
                tip_y=0.2,
                opening_y=0.5,
            )

        def arm_opening_body_barrier_predictive_skin(self, armed):
            events.append(("barrier", armed))

        def rotate_grasped_opening_about_span(self, delta):
            events.append(("rotate", delta))
            self.rotation += delta

        def step_physics(self):
            events.append("rotation_physics")

        def frame_side_camera_on_sock(self, distance, reverse=False):
            events.append("side_camera")
            return {"distance_m": distance, "reverse": reverse}

        def observe_drape_cameras(self):
            events.append("settle")
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            return {
                "geometry": DrapeGeometry(
                    self.rotation,
                    alignment,
                    alignment,
                ),
                "overview": {"rgb": rgb},
                "side": {"rgb": rgb},
            }

    class Policy(_Policy):
        def step(self, **kwargs):
            events.append("policy")
            return super().step(**kwargs)

    monkeypatch.setattr(
        SockDressingEnv,
        "cloth_radius_qa",
        staticmethod(lambda *args, **kwargs: {"passes": True}),
    )
    result = run_demo(
        config,
        prepared={"runtime_urdf": str(tmp_path / "robot.urdf")},
        output_root=tmp_path / f"episodes-{alignment}",
        sock_points=[[1, 1]],
        leg_points=[[3, 2]],
        max_steps=1,
        environment_factory=Environment,
        perception_factory=_Perception,
        policy_factory=Policy,
    )

    assert result["ok"] is expected_ok
    assert events.count("rotation_physics") == 2
    assert [event for event in events if isinstance(event, tuple) and event[0] == "rotate"] == [
        ("rotate", 90.0),
        ("rotate", 90.0),
    ]
    metadata = json.loads(Path(result["episode"], "metadata.json").read_text())
    report = metadata["pre_inference_drape"]
    assert report["passed"] is expected_ok
    assert report["final_geometry"][
        "signed_opening_span_rotation_degrees"
    ] == pytest.approx(180.0)
    assert Path(result["pre_inference_drape_video"]).is_file()
    if expected_ok:
        assert metadata["post_pre_drape_prompt_mode"] == "single_centroid"
        assert events.index("policy") > max(
            index for index, event in enumerate(events) if event == "settle"
        )
    else:
        assert "policy" not in events
        assert result["stop_reason"].startswith(
            "fail_closed: pre-inference drape geometry QA failed"
        )


def test_demo_records_bounded_closed_loop_actions(tmp_path):
    config = load_config()
    config["assets"]["output_dir"] = str(tmp_path)
    config["inference"]["record_inference_camera_video"] = True
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
    assert Path(result["inference_camera_video"]).is_file()
    assert metadata["inference_camera_video"] == result[
        "inference_camera_video"
    ]


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
                "opening_ring_area_retention": 0.97,
                "right_toe_position": [0.0, -0.02, 0.0],
            },
        }
    }

    report = _grasp_frame_report(observation, config)

    assert report["ok"]
    assert report["attached_sides"] == ["left", "right"]
    assert report["maximum_edge_error_m"] == 0.01
    assert report["opening_span_m"] == 0.11
    assert report["opening_ring_area_retention"] == 0.97
    assert report["left_grasp_position"] == [0.0, 0.0, 0.0]
    assert report["right_grasp_position"] == [0.0, 0.1, 0.0]
    assert report["left_minus_right_vertical_m"] == pytest.approx(-0.1)
    assert report["grasp_toe_vertical_clearance_m"] == pytest.approx(0.02)


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
    assert following["responding_ok"]
    assert following["distal_follow_ratio"] == pytest.approx(0.9)
    assert following["distal_follow_direction_alignment"] == pytest.approx(1.0)
    assert not collapsed["following_ok"]
    assert not collapsed["responding_ok"]
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
                "responding_ok": True,
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

    config["inference"]["minimum_opening_span_m"] = 0.09
    config["inference"]["minimum_opening_ring_area_retention"] = 0.95
    narrow_opening = _task_success(
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
                    "opening_span_m": 0.08,
                    "opening_ring_area_retention": 0.94,
                    "left_minus_right_vertical_m": 0.01,
                }
            ],
        },
    )
    assert not narrow_opening["continuous_opening_span_ok"]
    assert narrow_opening["minimum_opening_span_m"] == pytest.approx(0.08)
    assert narrow_opening[
        "minimum_opening_ring_area_retention"
    ] == pytest.approx(0.94)

    crossed_grasps = _task_success(
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
                    "opening_span_m": 0.10,
                    "opening_ring_area_retention": 0.97,
                    "left_minus_right_vertical_m": 0.01,
                },
                {
                    "ok": True,
                    "attached_grippers": 2,
                    "maximum_edge_error_m": 0.01,
                    "opening_span_m": 0.10,
                    "opening_ring_area_retention": 0.97,
                    "left_minus_right_vertical_m": -0.01,
                },
            ],
        },
    )
    assert not crossed_grasps["continuous_opening_span_ok"]
    assert not crossed_grasps["grasp_vertical_order_preserved"]

    config["inference"]["minimum_grasp_toe_vertical_clearance_m"] = 0.0
    below_toe = _task_success(
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
                    "opening_span_m": 0.10,
                    "opening_ring_area_retention": 0.97,
                    "left_minus_right_vertical_m": 0.01,
                    "grasp_toe_vertical_clearance_m": 0.01,
                },
                {
                    "ok": True,
                    "attached_grippers": 2,
                    "maximum_edge_error_m": 0.01,
                    "opening_span_m": 0.10,
                    "opening_ring_area_retention": 0.97,
                    "left_minus_right_vertical_m": 0.01,
                    "grasp_toe_vertical_clearance_m": -0.005,
                },
            ],
        },
    )
    assert not below_toe["grasp_toe_clearance_ok"]
    assert below_toe[
        "minimum_grasp_toe_vertical_clearance_m"
    ] == pytest.approx(-0.005)
    assert below_toe["first_grasp_below_toe_frame"] == 1


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
                "responding_ok": True,
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
    assert report["maximum_obi_cloth_foot_penetration_m"] == pytest.approx(
        0.0008
    )
    assert report[
        "maximum_geometric_cloth_foot_penetration_m"
    ] == pytest.approx(0.0006)


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


def test_task_success_reports_gripper_foot_region_passage(monkeypatch):
    config = load_config(Path("config/custom_player.yaml"))
    config["inference"]["maximum_gripper_foot_region_penetration_m"] = 0.001
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
                "maximum_penetration_m": 0.0002,
                "penetrating_pairs": [
                    {
                        "human_region": "forefoot",
                        "ignored": False,
                        "penetration_m": 0.0002,
                        "robot_collider_path": "robot/left_gripper/finger",
                    }
                ],
            },
            {
                "ignored_pair_count": 0,
                "maximum_penetration_m": 0.0012,
                "penetrating_pairs": [
                    {
                        "human_region": "ankle",
                        "ignored": False,
                        "penetration_m": 0.0012,
                        "robot_collider_path": "robot/right_gripper/finger",
                    },
                    {
                        "human_region": "calf",
                        "ignored": False,
                        "penetration_m": 0.004,
                        "robot_collider_path": "robot/right_gripper/finger",
                    },
                ],
            },
        ],
    )

    assert not report["gripper_foot_passage_ok"]
    assert report["gripper_foot_region_contact_frames"] == [0, 1]
    assert report["gripper_foot_region_penetration_frames"] == [1]
    assert report["first_gripper_foot_region_contact_frame"] == 0
    assert report[
        "maximum_gripper_foot_region_penetration_m"
    ] == pytest.approx(0.0012)


def test_contact_rebound_report_uses_only_post_contact_window():
    dressing = [
        {
            "foot_contact_count": 0,
            "maximum_cuff_reverse_step_m": 0.02,
            "cuff_progress_toward_ankle_m": 0.0,
        },
        {
            "foot_contact_count": 1,
            "maximum_cuff_reverse_step_m": 0.001,
            "cuff_progress_toward_ankle_m": 0.01,
        },
        {
            "foot_contact_count": 1,
            "maximum_cuff_reverse_step_m": 0.003,
            "cuff_progress_toward_ankle_m": 0.006,
        },
        {
            "foot_contact_count": 1,
            "maximum_cuff_reverse_step_m": 0.001,
            "cuff_progress_toward_ankle_m": 0.009,
        },
    ]
    grasp = [
        {"opening_span_m": 0.115, "opening_ring_area_retention": 1.0},
        {"opening_span_m": 0.100, "opening_ring_area_retention": 0.98},
        {"opening_span_m": 0.092, "opening_ring_area_retention": 0.95},
        {"opening_span_m": 0.105, "opening_ring_area_retention": 1.01},
    ]

    report = _contact_rebound_report(
        dressing, grasp, [[], [2105], [2105], [2105]], window_frames=2
    )

    assert report["foot_contact_onset_frame"] == 1
    assert report["maximum_post_contact_cuff_reverse_m"] == pytest.approx(0.003)
    assert report["maximum_post_contact_cuff_progress_drop_m"] == pytest.approx(
        0.004
    )
    assert report["minimum_post_contact_opening_span_m"] == pytest.approx(0.092)
    assert report[
        "minimum_post_contact_opening_ring_area_retention"
    ] == pytest.approx(0.95)
    assert report["maximum_post_contact_opening_span_overshoot_m"] == pytest.approx(
        0.005
    )
    assert report["maximum_post_contact_opening_area_overshoot"] == pytest.approx(
        0.03
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
