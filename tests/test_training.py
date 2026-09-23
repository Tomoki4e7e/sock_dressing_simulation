import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.quality import compare_learning_domains
from sock_dressing_simulation.training import (
    audit_training_data,
    load_episode_arrays,
    load_training_specs,
)


def _teacher_data(tmp_path: Path):
    dataset = tmp_path / "data" / "sock"
    manifest = {"sock": {"train": {"a": {"start": 0, "end": 3}}, "test": {"b": {"start": 0, "end": 3}}}}
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    for split, name in (("train", "a"), ("test", "b")):
        episode = dataset / split / name
        for directory in (
            "camera_right",
            "camera_depth",
            "depth_mask/sock_depth",
            "depth_mask/foot_depth",
        ):
            (episode / directory).mkdir(parents=True)
        angle = np.arange(54, dtype=float).reshape(3, 18)
        np.savetxt(episode / "angle.csv", angle, delimiter=",")
        np.savetxt(episode / "torque.csv", angle + 1, delimiter=",")
        for index in range(3):
            Image.fromarray(np.full((8, 9, 3), index + 5, np.uint8)).save(
                episode / "camera_right" / f"{index}.png"
            )
            for directory in ("camera_depth", "depth_mask/sock_depth", "depth_mask/foot_depth"):
                Image.fromarray(np.full((8, 9), index + 5, np.uint8)).save(
                    episode / directory / f"{index}.png"
                )
    config = load_config()
    config["inference"]["training_data_root"] = str(tmp_path / "data")
    config["inference"]["training_manifest"] = str(manifest_path)
    config["inference"]["training_dataset_name"] = "sock"
    return config


def test_training_audit_and_foot_to_leg_mapping(tmp_path):
    config = _teacher_data(tmp_path)
    report = audit_training_data(config)
    assert report["ok"]
    specs = load_training_specs(config)
    rgb, joints, sock, leg = load_episode_arrays(
        specs[0], image_size=4, smooth_torque=1, skip_num=1
    )
    assert rgb.shape == (3, 3, 4, 4)
    assert joints.shape == (3, 36)
    assert sock.shape == leg.shape == (3, 1, 4, 4)


def test_training_resamples_episode_to_fixed_sequence_length(tmp_path):
    config = _teacher_data(tmp_path)
    spec = load_training_specs(config)[0]
    rgb, joints, sock, leg = load_episode_arrays(
        spec,
        image_size=4,
        smooth_torque=1,
        skip_num=1,
        sequence_length=2,
    )
    assert rgb.shape == (2, 3, 4, 4)
    assert joints.shape == (2, 36)
    assert sock.shape == leg.shape == (2, 1, 4, 4)
    np.testing.assert_array_equal(joints[:, 0], [0.0, 36.0])


def test_training_rejects_sequence_longer_than_episode(tmp_path):
    config = _teacher_data(tmp_path)
    spec = load_training_specs(config)[0]
    try:
        load_episode_arrays(
            spec,
            image_size=4,
            smooth_torque=1,
            skip_num=1,
            sequence_length=4,
        )
    except ValueError as error:
        assert "exceeds episode length" in str(error)
    else:
        raise AssertionError("oversized sequence_length was accepted")


def test_autonomous_comparison_config_resolves_nested_extends():
    config = load_config(Path("config/autonomous_real_only.yaml"))
    assert config["rcareworld"]["profile"] == "custom_player"
    assert config["inference"]["training_sequence_length"] == 50
    assert config["inference"]["reference_actions"] is None
    assert config["inference"]["reference_action_blend"] == 0.0
    assert config["inference"]["reference_cartesian_pull_m"] == 0.0


def test_training_audit_rejects_missing_visual_frame(tmp_path):
    config = _teacher_data(tmp_path)
    missing = (
        tmp_path / "data" / "sock" / "train" / "a" / "depth_mask" / "foot_depth" / "1.png"
    )
    missing.unlink()
    report = audit_training_data(config)
    assert not report["ok"]
    assert any("foot_depth missing" in error for error in report["episodes"][0]["errors"])


def test_training_audit_rejects_zero_torque(tmp_path):
    config = _teacher_data(tmp_path)
    episode = tmp_path / "data" / "sock" / "train" / "a"
    np.savetxt(episode / "torque.csv", np.zeros((3, 18)), delimiter=",")

    report = audit_training_data(config)

    assert not report["ok"]
    assert any("all zero" in error for error in report["episodes"][0]["errors"])


def test_domain_audit_maps_foot_and_leg_and_requires_head_mount(tmp_path):
    _teacher_data(tmp_path)
    real = tmp_path / "data" / "sock" / "train" / "a"
    simulation = tmp_path / "simulation"
    shutil.copytree(real, simulation)
    (simulation / "camera_right_mask" / "leg_mask").mkdir(parents=True)
    for index in range(3):
        Image.fromarray(np.eye(8, 9, dtype=np.uint8) * 255).save(
            simulation / "camera_right_mask" / "leg_mask" / f"{index}.png"
        )
    (simulation / "depth_mask" / "leg_depth").mkdir(parents=True)
    shutil.rmtree(simulation / "depth_mask" / "foot_depth")
    for index in range(3):
        Image.fromarray(np.eye(8, 9, dtype=np.uint8) * 100).save(
            simulation / "depth_mask" / "leg_depth" / f"{index}.png"
        )
    (simulation / "metadata.json").write_text(
        json.dumps({"diagnostics": {"camera_mount": {"mode": "robot_link"}}})
    )

    report = compare_learning_domains([real], [simulation])

    assert report["ok"]
    assert report["simulation"][0]["limb_alias"] == "leg"
    assert report["simulation"][0]["camera_mount"]["mode"] == "robot_link"
