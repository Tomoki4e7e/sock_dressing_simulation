import json
from pathlib import Path

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

    def InstanceObject(self, name, id=None):
        return _Anchor(id)


class _Robot:
    def __init__(self):
        self.id = 1100
        self.data = {
            "joint_positions": np.zeros(29),
            "joint_force": np.zeros(29),
        }

    def SetJointPosition(self, values):
        self.data["joint_positions"] = np.asarray(values)


class _Cloth:
    def __init__(self):
        self.attachments = []

    def SetTransform(self, **kwargs):
        self.transform = kwargs

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
    assert report["method"] == "mesh-ring centroid radial distance"


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
