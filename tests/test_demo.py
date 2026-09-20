import csv
import json
from pathlib import Path

import numpy as np

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.demo import run_demo
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
    metadata = json.loads(open(f"{episode}/metadata.json").read())
    assert metadata["stop_reason"] == "max_steps"
    assert metadata["frames"] == 2
    assert metadata["video_camera"]["source"] == "recording_camera"
    assert metadata["video_camera"]["position"] == [-1.8, 1.05, -1.5]
    assert Path(result["video"]).is_file()
