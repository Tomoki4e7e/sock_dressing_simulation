import json

import numpy as np
import pytest

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.perception import (
    SAMDepthPerception,
    assess_masks,
    masked_depth,
)


class _Predictor:
    def __init__(self, logits):
        self.logits = logits

    def load_first_frame(self, frame):
        self.frame = frame

    def add_new_points(self, *args):
        return None, None, self.logits

    def track(self, frame):
        return None, self.logits


class _Depth:
    def infer_image(self, frame, input_size):
        return np.arange(frame.shape[0] * frame.shape[1], dtype=np.float32).reshape(frame.shape[:2])


def _config(tmp_path):
    config = load_config()
    calibration = tmp_path / "depth.json"
    calibration.write_text(json.dumps({"depth_min": 0.0, "depth_max": 20.0}))
    config["inference"]["depth_anything"]["calibration"] = str(calibration)
    config["inference"]["mask_min_fraction"] = 0.01
    return config


def test_sam_depth_perception_creates_masked_depth(tmp_path):
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    sock = np.zeros((1, 1, 4, 5), dtype=np.float32)
    leg = np.zeros((1, 1, 4, 5), dtype=np.float32)
    sock[:, :, 1, 1:3] = 1
    leg[:, :, 2:, 3:] = 1
    perception = SAMDepthPerception(
        _config(tmp_path),
        sock_predictor=_Predictor(sock),
        leg_predictor=_Predictor(leg),
        depth_model=_Depth(),
    )
    result = perception.initialize(
        frame, sock_points=[[1, 1]], leg_points=[[3, 3]]
    )
    assert result.quality["ok"]
    assert result.sock_depth.dtype == np.uint8
    assert np.all(result.sock_depth[~result.sock_mask] == 0)
    assert np.all(result.leg_depth[~result.leg_mask] == 0)


def test_degenerate_masks_fail_closed(tmp_path):
    mask = np.ones((1, 1, 4, 5), dtype=np.float32)
    perception = SAMDepthPerception(
        _config(tmp_path),
        sock_predictor=_Predictor(mask),
        leg_predictor=_Predictor(mask),
        depth_model=_Depth(),
    )
    with pytest.raises(RuntimeError, match="quality failed"):
        perception.initialize(
            np.zeros((4, 5, 3), np.uint8),
            sock_points=[[1, 1]],
            leg_points=[[2, 2]],
        )


def test_mask_contract_helpers():
    sock = np.zeros((3, 4), bool)
    leg = np.zeros((3, 4), bool)
    sock[0, 0] = True
    leg[2, 3] = True
    assert assess_masks(sock, leg)["ok"]
    depth = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert masked_depth(depth, sock).sum() == depth[0, 0]
