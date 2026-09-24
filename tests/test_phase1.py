import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.episode import EpisodeWriter
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.pipeline import run_phase1_pipeline
from sock_dressing_simulation.quality import assess_observation_quality
from sock_dressing_simulation.scenario import scenario_from_config


def test_scenario_sampling_is_seeded_and_validates_mimic():
    config = load_config()
    config["scenario"]["randomization"]["enabled"] = True
    first = scenario_from_config(config, seed=12)
    second = scenario_from_config(config, seed=12)
    assert first == second
    assert first.sock_mesh.length_m == pytest.approx(0.30)
    assert first.sock_mesh.radius_m == pytest.approx(0.04)
    assert first.foot_ik_index == 3
    expected_arm = np.deg2rad([35.0, -85.0, -60.0, 135.0, 150.0, 25.0, 23.0])
    np.testing.assert_allclose(first.initial_joints[:7], expected_arm, atol=5e-6)
    np.testing.assert_allclose(first.initial_joints[9:16], expected_arm, atol=5e-6)
    np.testing.assert_allclose(
        np.asarray(first.initial_joints)[[7, 8, 16, 17]], np.zeros(4)
    )
    assert first.initial_joints_source == "shareset-change_pose/ka"

    config["scenario"]["initial_joints"][8] = 0.01
    with pytest.raises(ValueError, match="mimic"):
        scenario_from_config(config)


def test_observation_quality_is_fail_closed_for_null_renderer():
    camera = {
        "rgb": np.full((4, 5, 3), 205, dtype=np.uint8),
        "camera_depth": np.full((4, 5), 205, dtype=np.uint8),
        "sock_mask": np.ones((4, 5), dtype=bool),
        "leg_mask": np.ones((4, 5), dtype=bool),
    }
    report = assess_observation_quality(
        [camera, camera],
        expected_width=5,
        expected_height=4,
        exact_foot_colliders=False,
    )
    assert not report["learning_ready"]
    assert "depth_nonconstant" in report["failed_checks"]
    assert "exact_foot_colliders" in report["failed_checks"]


class _Backend:
    def step(self):
        pass

    def close(self):
        pass

    def InstanceObject(self, name, id=None, **kwargs):
        return _Anchor(id)


def _safe_cloth_data():
    return {
        "particles": [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.0, 0.01, 0.0],
            [0.01, 0.01, 0.0],
        ],
        "particle_edges": [[0, 1], [0, 2], [1, 3], [2, 3]],
        "particle_rest_edge_lengths": [0.01, 0.01, 0.01, 0.01],
        "opening_particle_indices": [0, 1, 2, 3],
        "grasp_state": [],
    }


class _Robot:
    def __init__(self):
        self.id = 1100
        self.data = {
            "joint_positions": np.zeros(29),
            "joint_force": np.zeros(29),
        }

    def SetJointPosition(self, values):
        self.data["joint_positions"] = np.asarray(values)

    def SetJointPositionDirectly(self, values):
        self.data["joint_positions"] = np.asarray(values)


class _Cloth:
    def __init__(self):
        self.attachments = []

    def SetTransform(self, **kwargs):
        self.transform = kwargs

    def reset(self):
        self.was_reset = True

    def align_grasp_targets_to_opening(self):
        self.was_aligned = True

    def clamp_grasp_target_span(self, maximum_span_m):
        self.maximum_grasp_span = maximum_span_m

    def align_sock_opening_to_grasp_targets(self, toe_target):
        self.sock_was_aligned_to_grippers = True
        self.sock_toe_target = list(toe_target)

    def arm_slip_detection(self, armed=True):
        self.slip_detection_armed = armed

    def request_configuration(self):
        self.configuration_requested = True

    def stop_foot_clearance_tracking(self):
        self.clearance_tracking_stopped = True

    def lock_human_and_chair(self, chair_id=-1):
        self.human_and_chair_locked = True

    def align_human_visual_foot_to_sock(self, distance):
        self.visual_foot_distance = distance

    def AddAttach(self, id, max_dis):
        self.attachments.append((id, max_dis))


class _Anchor:
    def __init__(self, id):
        self.id = id

    def SetParent(self, parent_id, parent_name):
        self.parent = (parent_id, parent_name)

    def SetTransform(self, **kwargs):
        self.transform = kwargs


class _Human:
    def __init__(self):
        self.calls = []

    def HumanIKTargetDoMove(self, **kwargs):
        self.calls.append(("move", kwargs))

    def HumanIKTargetDoRotate(self, **kwargs):
        self.calls.append(("rotate", kwargs))

    def HumanIKTargetDoComplete(self, index):
        self.calls.append(("complete", index))


def test_inference_camera_is_parented_to_right_see3cam(monkeypatch):
    attributes = ModuleType("pyrcareworld.attributes")
    attributes.CameraAttr = object
    package = ModuleType("pyrcareworld")
    package.attributes = attributes
    monkeypatch.setitem(sys.modules, "pyrcareworld", package)
    monkeypatch.setitem(sys.modules, "pyrcareworld.attributes", attributes)
    config = load_config(Path("config/custom_player.yaml"))
    environment = SockDressingEnv(config, backend=_Backend())
    environment.robot = _Robot()

    environment._create_native_cameras()

    assert environment.camera.parent == (
        environment.robot.id,
        "head/see3cam_right/camera_color_frame",
    )
    assert environment.camera.transform == {
        "position": [0.0, 0.0, 0.0],
        "rotation": [20.0, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "is_world": False,
    }
    assert environment._camera_mount_report["mode"] == "robot_link"


def test_inference_camera_world_pose_remains_an_explicit_fallback(monkeypatch):
    attributes = ModuleType("pyrcareworld.attributes")
    attributes.CameraAttr = object
    package = ModuleType("pyrcareworld")
    package.attributes = attributes
    monkeypatch.setitem(sys.modules, "pyrcareworld", package)
    monkeypatch.setitem(sys.modules, "pyrcareworld.attributes", attributes)
    config = load_config()
    config["scene"]["camera_parent_link"] = None
    environment = SockDressingEnv(config, backend=_Backend())
    environment.robot = _Robot()

    environment._create_native_cameras()

    assert not hasattr(environment.camera, "parent")
    assert environment.camera.transform["position"] == config["scene"]["camera_position"]
    assert environment._camera_mount_report["mode"] == "world"


def test_environment_applies_sock_foot_and_grasp_scenario():
    scenario = scenario_from_config(load_config())
    environment = SockDressingEnv(load_config(), backend=_Backend())
    environment.robot = _Robot()
    environment.cloth = _Cloth()
    environment.human = _Human()
    report = environment.apply_scenario(scenario)
    assert environment.cloth.transform["position"] == list(scenario.sock_position)
    assert [call[0] for call in environment.human.calls] == [
        "move",
        "rotate",
        "complete",
        "move",
        "rotate",
        "complete",
    ]
    assert report["foot_ik_applied"]
    assert report["support_foot_ik_applied"]
    assert [item["side"] for item in report["grasp_attachments"]] == [
        "left",
        "right",
    ]


def test_custom_foot_calibration_sets_clearance_before_pose_is_locked():
    config = load_config(Path("config/custom_player.yaml"))
    environment = SockDressingEnv(config, backend=_Backend())

    class Cloth:
        def __init__(self):
            self.clearance = None
            self.locked = False

        def set_foot_clearance_target(self, distance, chair_id):
            self.clearance = (distance, chair_id)

        def lock_human_and_chair(self, chair_id=-1):
            self.locked = True

        def request_scene_geometry(self):
            pass

        def scene_geometry(self):
            return SimpleNamespace(
                foot_to_opening_plane_m=0.1,
                foot_to_opening_lateral_m=0.0,
            )

    environment.sock_cloth = Cloth()
    environment._calibrate_foot_to_sock(None, 0.1)

    assert environment.sock_cloth.clearance == (0.1, 2301)
    assert not environment.sock_cloth.locked


def test_post_calibration_toe_offset_articulates_leg_to_world_target():
    config = load_config(Path("config/custom_player.yaml"))
    config["scene"]["initial_pose_contract"].update(
        {
            "right_toe_offset_world_m": [0.05, -0.05, 0.0],
            "right_toe_offset_tolerance_m": 0.005,
        }
    )
    environment = SockDressingEnv(config, backend=_Backend())

    class Cloth:
        def __init__(self):
            self.positions = [
                (-0.18, 0.52, 0.59),
                (-0.13, 0.47, 0.59),
            ]
            self.requested_target = None

        def request_scene_geometry(self):
            pass

        def scene_geometry(self):
            return SimpleNamespace(right_toe_position=self.positions.pop(0))

        def set_task_right_toe_position_articulated(self, position):
            self.requested_target = tuple(position)

    environment.sock_cloth = Cloth()
    report = environment._apply_post_calibration_right_toe_offset()

    np.testing.assert_allclose(
        environment.sock_cloth.requested_target, [-0.13, 0.47, 0.59]
    )
    np.testing.assert_allclose(report["achieved_offset_m"], [0.05, -0.05, 0.0])
    assert report["frame"] == "unity_world"
    assert report["ok"]


@pytest.mark.parametrize(
    ("toe_alignment", "left_depth", "offset_report", "expected"),
    [
        (0.95, 0.0, None, True),
        (0.5, 0.0, None, False),
        (0.5, 0.0, {"ok": True}, False),
    ],
)
def test_initial_pose_contract_requires_toe_facing_opening_edge_grasp(
    toe_alignment, left_depth, offset_report, expected
):
    config = load_config(Path("config/custom_player.yaml"))
    environment = SockDressingEnv(config, backend=_Backend())

    class Cloth:
        data = _safe_cloth_data()

        def request_scene_geometry(self):
            pass

        def request_grasp_state(self):
            pass

        def request_particles(self):
            pass

        def request_configuration(self):
            pass

        def request_visual_diagnostics(self, chair_id):
            assert chair_id == 2301

        def grasp_states(self):
            return (
                SimpleNamespace(
                    side="left", attached=True, particle_indices=(1, 2, 5, 6)
                ),
                SimpleNamespace(
                    side="right", attached=True, particle_indices=(3, 4, 7, 8)
                ),
            )

        def scene_geometry(self):
            return SimpleNamespace(
                foot_to_opening_plane_m=0.1,
                foot_to_opening_lateral_m=0.0,
                right_leg_raise_degrees=90.0,
                right_knee_flexion_degrees=0.0,
                sock_body_gravity_alignment=0.0,
                opening_to_toe_alignment=toe_alignment,
                left_cuff_insertion_depth_m=left_depth,
                right_cuff_insertion_depth_m=0.025,
                opening_center=(0.0, 0.0, 0.0),
                opening_normal=(0.0, 0.0, 1.0),
                opening_outward_normal=(0.0, 0.0, -1.0),
                opening_target_normal=(0.0, 0.0, 1.0),
                sock_body_direction=(0.0, -1.0, 0.0),
                right_toe_position=(0.0, 0.0, -0.1),
                left_grasp_position=(-0.04, 0.0, 0.03),
                right_grasp_position=(0.04, 0.0, 0.03),
                left_opening_edge=(-0.04, 0.0, 0.03),
                right_opening_edge=(0.04, 0.0, 0.03),
                left_grasp_corner_negative=(-0.04, -0.02, 0.03),
                left_grasp_corner_positive=(-0.04, 0.02, 0.03),
                right_grasp_corner_negative=(0.04, -0.02, 0.03),
                right_grasp_corner_positive=(0.04, 0.02, 0.03),
                left_grasp_patch_span_m=0.04,
                right_grasp_patch_span_m=0.04,
                maximum_grasp_corner_error_m=0.0,
                grasp_thickness_axis_alignment=1.0,
                left_grasp_thickness_axis=(0.0, 1.0, 0.0),
                right_grasp_thickness_axis=(0.0, 1.0, 0.0),
                left_grasp_inward_axis=(0.0, 0.0, 1.0),
                right_grasp_inward_axis=(0.0, 0.0, 1.0),
            )

        def visual_diagnostics(self):
            return [
                {"role": "human_task_pose", "valid": True},
                {
                    "role": "human_chair_lock",
                    "valid": True,
                    "locked": True,
                    "right_toe_drift_m": 0.0,
                    "chair_drift_m": 0.0,
                    "human_root_drift_m": 0.0,
                    "human_anchor_drift_m": 0.0,
                },
            ]

    environment.sock_cloth = Cloth()
    environment._right_toe_offset_report = offset_report
    report = environment._request_initial_pose_contract()

    assert report["ok"] is expected
    assert report["opening_to_toe_alignment"] == pytest.approx(toe_alignment)
    assert report["left_cuff_insertion_depth_m"] == pytest.approx(left_depth)
    assert report["opening_edges_at_grippers"]
    assert report["bimanual_grasp_attached"]
    assert report["grasp_particle_indices"] == {
        "left": [1, 2, 5, 6],
        "right": [3, 4, 7, 8],
    }
    assert report["left_grasp_inward_axis"] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("right_edge", "right_attached", "expected"),
    [
        ((0.04, 0.0, 0.03), True, True),
        ((0.06, 0.0, 0.03), True, False),
        ((0.04, 0.0, 0.03), False, False),
    ],
)
def test_initial_pose_contract_requires_edges_at_attached_grippers(
    right_edge, right_attached, expected
):
    config = load_config(Path("config/custom_player.yaml"))
    environment = SockDressingEnv(config, backend=_Backend())

    class Cloth:
        data = _safe_cloth_data()

        def request_scene_geometry(self):
            pass

        def request_grasp_state(self):
            pass

        def request_particles(self):
            pass

        def request_configuration(self):
            pass

        def request_visual_diagnostics(self, chair_id):
            pass

        def grasp_states(self):
            return (
                SimpleNamespace(
                    side="left", attached=True, particle_indices=(1, 2, 5, 6)
                ),
                SimpleNamespace(
                    side="right",
                    attached=right_attached,
                    particle_indices=(3, 4, 7, 8) if right_attached else (),
                ),
            )

        def scene_geometry(self):
            return SimpleNamespace(
                foot_to_opening_plane_m=0.1,
                foot_to_opening_lateral_m=0.0,
                right_leg_raise_degrees=90.0,
                right_knee_flexion_degrees=0.0,
                sock_body_gravity_alignment=0.0,
                opening_to_toe_alignment=0.95,
                left_cuff_insertion_depth_m=0.0,
                right_cuff_insertion_depth_m=0.0,
                opening_center=(0.0, 0.0, 0.0),
                opening_normal=(0.0, 0.0, 1.0),
                opening_outward_normal=(0.0, 0.0, -1.0),
                opening_target_normal=(0.0, 0.0, 1.0),
                sock_body_direction=(0.0, -1.0, 0.0),
                right_toe_position=(0.0, 0.0, -0.1),
                left_grasp_position=(-0.04, 0.0, 0.03),
                right_grasp_position=(0.04, 0.0, 0.03),
                left_opening_edge=(-0.04, 0.0, 0.03),
                right_opening_edge=right_edge,
                left_grasp_corner_negative=(-0.04, -0.02, 0.03),
                left_grasp_corner_positive=(-0.04, 0.02, 0.03),
                right_grasp_corner_negative=(0.04, -0.02, 0.03),
                right_grasp_corner_positive=(0.04, 0.02, 0.03),
                left_grasp_patch_span_m=0.04,
                right_grasp_patch_span_m=0.04,
                maximum_grasp_corner_error_m=0.0,
                grasp_thickness_axis_alignment=1.0,
                left_grasp_thickness_axis=(0.0, 1.0, 0.0),
                right_grasp_thickness_axis=(0.0, 1.0, 0.0),
                left_grasp_inward_axis=(0.0, 0.0, 1.0),
                right_grasp_inward_axis=(0.0, 0.0, 1.0),
            )

        def visual_diagnostics(self):
            return [
                {"role": "human_task_pose", "valid": True},
                {
                    "role": "human_chair_lock",
                    "valid": True,
                    "locked": True,
                    "right_toe_drift_m": 0.0,
                    "chair_drift_m": 0.0,
                    "human_root_drift_m": 0.0,
                    "human_anchor_drift_m": 0.0,
                },
            ]

    environment.sock_cloth = Cloth()
    report = environment._request_initial_pose_contract()

    assert report["ok"] is expected
    assert report["opening_edges_at_grippers"] is (expected or not right_attached)
    assert report["bimanual_grasp_attached"] is right_attached


def test_custom_environment_starts_with_verified_bimanual_grasp(monkeypatch):
    config = load_config(Path("config/custom_player.yaml"))
    scenario = scenario_from_config(config)
    environment = SockDressingEnv(config, backend=_Backend())
    environment.robot = _Robot()
    environment.cloth = _Cloth()
    environment.sock_cloth = environment.cloth
    environment.human = _Human()
    calls = []
    monkeypatch.setattr(
        environment.sock_cloth,
        "align_sock_opening_to_grasp_targets_and_grasp",
        lambda toe_target, distance: (
            calls.append(("align_and_grasp", tuple(toe_target), distance))
            or setattr(environment.cloth, "sock_was_aligned_to_grippers", True)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        environment.sock_cloth, "request_grasp_state", lambda: None, raising=False
    )
    monkeypatch.setattr(
        environment.sock_cloth,
        "grasp_states",
        lambda: tuple(
            SimpleNamespace(
                side=side,
                attached=True,
                particle_indices=(1, 2, 5, 6),
                constraint_error=0.0,
                peak_constraint_error=0.0,
                over_threshold_steps=0,
                release_reason="",
            )
            for side in ("left", "right")
        ),
        raising=False,
    )
    monkeypatch.setattr(environment, "_calibrate_foot_to_sock", lambda *args: None)
    monkeypatch.setattr(
        environment,
        "_request_grasp_target_positions",
        lambda: (
            np.asarray([-0.10, 0.0, 0.0]),
            np.asarray([0.10, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(
        environment,
        "move_grippers_to_targets",
        lambda left, right: {"ok": True, "left": left, "right": right},
    )
    final_toe = tuple(config["scene"]["visuals"]["task_right_toe_position"])
    monkeypatch.setattr(
        environment,
        "_request_scene_geometry",
        lambda: SimpleNamespace(right_toe_position=final_toe),
    )
    monkeypatch.setattr(
        environment,
        "_request_initial_pose_contract",
        lambda: {
            "ok": True,
            "foot_to_sock_m": 0.1,
            "right_leg_raise_degrees": 90,
        },
    )
    report = environment.apply_scenario(scenario)
    assert environment.cloth.was_reset
    assert not hasattr(environment.cloth, "was_aligned")
    assert environment.cloth.sock_was_aligned_to_grippers
    assert not hasattr(environment.cloth, "maximum_grasp_span")
    assert environment.cloth.slip_detection_armed
    assert getattr(environment.cloth, "visual_foot_distance", None) is None
    assert calls == [
        (
            "align_and_grasp",
            final_toe,
            config["scene"]["grasp_anchors"]["max_distance_m"],
        ),
    ]
    assert all(item["attached"] for item in report["initial_grasp"])
    assert report["grasp_alignment"]["ok"]
    assert report["initial_pose_contract"]["ok"]


def test_bimanual_cartesian_alignment_updates_both_arm_chains(monkeypatch):
    config = load_config(Path("config/custom_player.yaml"))
    config["scene"]["grasp_alignment"].update(
        {
            "tolerance_m": 0.001,
            "finite_difference_rad": 0.01,
            "damping": 0.0001,
            "maximum_joint_step_rad": 0.1,
            "maximum_iterations": 1,
        }
    )
    environment = SockDressingEnv(config, backend=_Backend())
    angle = np.zeros(18)
    driven = []

    def robot_signals():
        return {"angle": angle.copy()}

    def drive(target):
        angle[:] = np.asarray(target, dtype=float)
        driven.append(angle.copy())
        return angle.copy()

    def grasp_positions():
        return (
            np.array([angle[0], 0.0, 0.0]),
            np.array([angle[9], 0.0, 0.0]),
        )

    monkeypatch.setattr(environment, "robot_signals", robot_signals)
    monkeypatch.setattr(environment, "_drive_joint_target", drive)
    monkeypatch.setattr(environment, "_request_grasp_target_positions", grasp_positions)

    report = environment.move_grippers_to_targets(
        [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]
    )

    assert report["ok"]
    assert angle[0] > 0
    assert angle[9] > 0
    assert driven[-1][0] > 0 and driven[-1][9] > 0


def test_scene_contract_rejects_noncanonical_rest_mesh_and_left_target():
    config = load_config()
    config["scenario"]["sock"]["radius_m"] = 0.05
    with pytest.raises(ValueError, match="rest mesh radius"):
        scenario_from_config(config)

    config = load_config()
    config["scenario"]["foot"]["ik_index"] = 2
    with pytest.raises(ValueError, match="right foot"):
        scenario_from_config(config)


def test_coverage_measurement_fails_closed_for_degenerate_masks():
    degenerate = {
        "sock_mask": np.ones((5, 6), dtype=bool),
        "leg_mask": np.ones((5, 6), dtype=bool),
    }
    assert SockDressingEnv.measured_coverage(degenerate) is None

    valid = {
        "sock_mask": np.array([[1, 1, 0, 0]], dtype=bool),
        "leg_mask": np.array([[0, 1, 1, 1]], dtype=bool),
    }
    assert SockDressingEnv.measured_coverage(valid) == pytest.approx(1 / 3)


def test_cloth_radius_qa_uses_mesh_rings_not_global_tube_axis():
    rows = []
    for z in (0.0, 0.15, 0.30):
        rows.extend(
            (0.04 * np.cos(angle), 0.04 * np.sin(angle), z)
            for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        )
    report = SockDressingEnv.cloth_radius_qa(
        {"particles": rows}, radial_segments=8
    )
    assert report["passes"]
    assert report["maximum_radius_m"] == pytest.approx(0.04)
    assert report["method"] == "mesh-ring maximum edge stretch"


def test_cloth_radius_qa_prefers_obi_topology_over_particle_array_order():
    report = SockDressingEnv.cloth_radius_qa(
        {
            "particles": [[0, 0, 0], [10, 0, 0], [0.01, 0, 0]],
            "particle_edges": [[0, 2]],
            "particle_rest_edge_lengths": [0.01],
        },
        radial_segments=3,
    )
    assert report["passes"]
    assert report["circumferential_stretch_proxy"] == pytest.approx(1.0)
    assert report["method"] == "Obi topology structural edge stretch"


def test_cloth_radius_qa_reports_body_pin_edge_classes():
    report = SockDressingEnv.cloth_radius_qa(
        {
            "particles": [[0, 0, 0], [0.01, 0, 0], [0.024, 0, 0], [0.04, 0, 0]],
            "particle_edges": [[0, 1], [1, 2], [2, 3]],
            "particle_rest_edge_lengths": [0.01, 0.01, 0.01],
            "grasp_state": [
                {
                    "attached": True,
                    "particle_indices": [2, 3],
                }
            ],
        },
        maximum_circumferential_stretch=1.5,
    )

    assert report["edge_classes"]["body_body"]["maximum_stretch"] == pytest.approx(1.0)
    assert report["edge_classes"]["pin_body"]["maximum_stretch"] == pytest.approx(1.4)
    assert report["edge_classes"]["pin_pin"]["maximum_stretch"] == pytest.approx(1.6)
    assert not report["passes"]


def test_topology_stretch_uses_rest_ratio_not_legacy_absolute_radius():
    report = SockDressingEnv.cloth_radius_qa(
        {
            "particles": [[0, 0, 0], [0.08, 0, 0], [0, 0.01, 0]],
            "particle_edges": [[0, 1]],
            "particle_rest_edge_lengths": [0.08],
        },
        maximum_circumferential_stretch=1.5,
    )

    assert report["passes"]
    assert report["maximum_current_edge_m"] == pytest.approx(0.08)


def test_synthetic_episode_passes_feature_visualization_and_audit(tmp_path):
    config = load_config()
    dataset = config["dataset"]["name"]
    episode_name = "synthetic"
    episode = tmp_path / dataset / "train" / episode_name
    with EpisodeWriter(episode, config["joints"]["names"], {"mode": "synthetic-regression"}) as writer:
        for frame in range(3):
            height, width = 120, 180
            leg = np.zeros((height, width), dtype=bool)
            sock = np.zeros((height, width), dtype=bool)
            leg[35 + frame : 85, 70:155] = True
            sock[25:95, 15 : 80 + frame] = True
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[..., 1] = leg * 180
            rgb[..., 0] = sock * 220
            writer.append(
                rgb=rgb,
                sock_mask=sock,
                leg_mask=leg,
                camera_depth=np.full((height, width), 100, dtype=np.uint8),
                angle=np.zeros(18),
                torque=np.zeros(18),
                external_torque=np.zeros(18),
            )
    manifest = tmp_path / "dataset_phase1.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {dataset: {"train": {episode_name: {"start": 0, "end": 3}}, "test": {}}},
            sort_keys=False,
        )
    )
    audit = tmp_path / "audit.json"
    result = run_phase1_pipeline(
        episode,
        data_root=tmp_path,
        manifest=manifest,
        dataset_name=dataset,
        audit_output=audit,
    )
    assert result["ok"]
    assert result["features_ok"] == 3
    assert np.load(episode / "features/features_ok.npy").all()
    assert json.loads(audit.read_text())["n_usable"] == 1
    validation = json.loads(
        (episode / "features/viz/validation.json").read_text()
    )
    assert validation["all_checks_ok"]
