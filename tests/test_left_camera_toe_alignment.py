import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sock_dressing_simulation.config import load_config


SCRIPT = Path("scripts/search_left_camera_toe_alignment.py")
SPEC = importlib.util.spec_from_file_location("toe_alignment_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
toe_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = toe_search
SPEC.loader.exec_module(toe_search)


def test_toe_tip_x_uses_lowest_leg_mask_band():
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[:80, 30:50] = 255
    mask[80:100, 90:111] = 255

    assert toe_search.toe_tip_x_from_mask(mask) == pytest.approx(100.0)


def test_alignment_measurement_reports_signed_center_error(tmp_path):
    mask = np.zeros((40, 100), dtype=np.uint8)
    mask[5:40, 20:31] = 255
    path = tmp_path / "leg.png"
    Image.fromarray(mask, "L").save(path)

    measurement = toe_search.alignment_measurement(path)

    assert measurement["toe_tip_x_px"] == pytest.approx(25.0)
    assert measurement["center_x_px"] == pytest.approx(49.5)
    assert measurement["signed_error_px"] == pytest.approx(-24.5)
    assert measurement["absolute_error_px"] == pytest.approx(24.5)


def test_alignment_candidates_refine_around_coarse_winner():
    assert toe_search.candidate_values(range(-3, 4)) == [
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
    ]
    assert toe_search.candidate_values(
        (), fine_center=1.0, fine_step=0.5
    ) == [0.5, 1.0, 1.5]
    assert toe_search.candidate_values(
        (), fine_center=3.0, fine_step=0.5
    ) == [2.5, 3.0]


def test_alignment_ranking_requires_pose_and_minimizes_pixel_error():
    centered = {
        "pose_screening_ok": True,
        "frames": 1,
        "stop_reason": "max_steps",
        "absolute_error_px": 4.0,
        "right_cm": 1.0,
        "camera_yaw_degrees": 0.0,
    }
    offset = {
        **centered,
        "absolute_error_px": 20.0,
    }
    invalid = {
        **centered,
        "pose_screening_ok": False,
        "absolute_error_px": 0.0,
    }

    assert toe_search.ranking_key(centered) > toe_search.ranking_key(offset)
    assert toe_search.ranking_key(offset) > toe_search.ranking_key(invalid)


def test_candidate_config_preserves_left_camera_and_down8_contract(tmp_path):
    base = Path(
        "config/autonomous_real_only_opening_reverse_270deg_"
        "positive_180_drape_recorded_pose_gripper_coupled_"
        "head_camera_frame_zero_physics_single_centroid_fast_policy_"
        "human_chair_away2cm_down7cm_left_camera.yaml"
    )
    path = tmp_path / "config.yaml"
    toe_search._candidate_config(
        toe_search.Candidate(right_cm=1.5, camera_yaw_degrees=2.0),
        path,
        base,
    )

    config = load_config(path)
    scene = config["scene"]
    pose = scene["initial_pose_contract"]
    assert scene["camera_parent_link"] == (
        "head/see3cam_left/camera_color_frame"
    )
    assert scene["camera_local_position"] == pytest.approx([0.0, -0.04, 0.1])
    assert scene["camera_local_rotation"] == pytest.approx([60.0, 2.0, 0.0])
    assert pose["away_from_robot_m"] == pytest.approx(0.02)
    assert pose["down_m"] == pytest.approx(0.08)
    assert pose["right_from_robot_m"] == pytest.approx(0.015)
    assert config["inference"]["reference_actions"] is None


def test_selected_toe_centered_profile_keeps_minimal_adjustments():
    config = load_config(
        Path(
            "config/autonomous_real_only_opening_reverse_270deg_"
            "positive_180_drape_recorded_pose_gripper_coupled_"
            "head_camera_frame_zero_physics_single_centroid_fast_policy_"
            "human_chair_away2cm_down8cm_left_camera_toe_centered.yaml"
        )
    )
    scene = config["scene"]
    pose = scene["initial_pose_contract"]

    assert scene["camera_parent_link"] == (
        "head/see3cam_left/camera_color_frame"
    )
    assert scene["camera_local_position"] == pytest.approx([0.0, -0.04, 0.1])
    assert scene["camera_local_rotation"] == pytest.approx([60.0, -3.0, 0.0])
    assert pose["away_from_robot_m"] == pytest.approx(0.02)
    assert pose["down_m"] == pytest.approx(0.08)
    assert pose["right_from_robot_m"] == pytest.approx(0.03)
    assert config["inference"]["reference_actions"] is None
    assert config["inference"]["reference_action_blend"] == pytest.approx(0.0)
