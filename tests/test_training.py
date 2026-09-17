import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from sock_dressing_simulation.config import load_config
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


def test_training_audit_rejects_missing_visual_frame(tmp_path):
    config = _teacher_data(tmp_path)
    missing = (
        tmp_path / "data" / "sock" / "train" / "a" / "depth_mask" / "foot_depth" / "1.png"
    )
    missing.unlink()
    report = audit_training_data(config)
    assert not report["ok"]
    assert any("foot_depth missing" in error for error in report["episodes"][0]["errors"])
