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


def test_leg_axis_measurement_reports_tilt_from_image_vertical():
    mask = np.zeros((200, 200), dtype=np.uint8)
    for y in range(10, 190):
        x = int(round(70 + 0.2 * (y - 10)))
        mask[y, x - 3 : x + 4] = 255

    measurement = toe_search.leg_axis_measurement(mask)

    assert measurement["leg_axis_pca_degrees"] == pytest.approx(
        np.degrees(np.arctan(0.2)), abs=0.5
    )
    assert measurement["leg_axis_ridge_degrees"] == pytest.approx(
        np.degrees(np.arctan(0.2)), abs=0.5
    )
    assert measurement["leg_axis_agreement_degrees"] < 1.0
    assert measurement["leg_axis_anisotropy"] > 5.0
    assert measurement["leg_axis_valid"]


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
        "leg_axis_valid": True,
        "leg_axis_pca_degrees": 1.0,
        "leg_axis_agreement_degrees": 0.2,
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
    tilted = {
        **centered,
        "leg_axis_pca_degrees": 8.0,
        "absolute_error_px": 0.0,
    }

    assert toe_search.ranking_key(centered) > toe_search.ranking_key(offset)
    assert toe_search.ranking_key(offset) > toe_search.ranking_key(invalid)
    assert toe_search.ranking_key(centered) > toe_search.ranking_key(tilted)


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


def test_vertical_foot_axis_profile_preserves_opening_and_drape():
    config = load_config(
        Path(
            "config/autonomous_real_only_opening_reverse_270deg_"
            "positive_180_drape_recorded_pose_gripper_coupled_"
            "head_camera_frame_zero_physics_single_centroid_fast_policy_"
            "human_chair_away2cm_down8cm_left_camera_foot_axis_vertical.yaml"
        )
    )
    scene = config["scene"]
    pose = scene["initial_pose_contract"]
    drape = config["inference"]["pre_inference_drape"]

    assert scene["camera_parent_link"] == (
        "head/see3cam_left/camera_color_frame"
    )
    assert scene["camera_local_position"] == pytest.approx([0.0, -0.04, 0.1])
    assert scene["camera_local_rotation"] == pytest.approx(
        [60.0, -12.0, 0.0]
    )
    assert pose["away_from_robot_m"] == pytest.approx(0.02)
    assert pose["down_m"] == pytest.approx(0.08)
    assert pose["right_from_robot_m"] == pytest.approx(0.0225)
    assert pose["opening_rotation_about_span_degrees"] == pytest.approx(90.0)
    assert not pose["opening_rotation_away_from_toe"]
    assert drape["rotation_degrees"] == pytest.approx(180.0)
    assert drape["rotation_steps"] == 180
    assert drape["settle_steps"] == 250


def test_opening_preserved_profile_changes_only_grasp_span_contract():
    baseline_path = Path(
        "config/autonomous_real_only_opening_reverse_270deg_"
        "positive_180_drape_recorded_pose_gripper_coupled_"
        "head_camera_frame_zero_physics_single_centroid_fast_policy_"
        "human_chair_away2cm_down8cm_left_camera_foot_axis_vertical.yaml"
    )
    tuned_path = baseline_path.with_name(
        baseline_path.stem + "_opening_preserved.yaml"
    )
    baseline = load_config(baseline_path)
    tuned = load_config(tuned_path)

    assert tuned["scenario"]["sock"] == baseline["scenario"]["sock"]
    assert tuned["scene"]["camera_parent_link"] == baseline["scene"][
        "camera_parent_link"
    ]
    assert tuned["scene"]["camera_local_position"] == baseline["scene"][
        "camera_local_position"
    ]
    assert tuned["scene"]["camera_local_rotation"] == baseline["scene"][
        "camera_local_rotation"
    ]
    assert tuned["scene"]["initial_pose_contract"] == baseline["scene"][
        "initial_pose_contract"
    ]
    assert tuned["inference"]["pre_inference_drape"] == baseline["inference"][
        "pre_inference_drape"
    ]
    assert tuned["scene"]["grasp_alignment"]["target_span_m"] == pytest.approx(
        0.115
    )
    expected = tuned["obi"]["expected"]
    assert expected["slip_minimum_opening_span_m"] == pytest.approx(0.092)
    assert expected["slip_opening_span_m"] == pytest.approx(0.115)
    assert tuned["inference"]["minimum_opening_span_m"] == pytest.approx(0.09)
    assert tuned["inference"][
        "minimum_opening_ring_area_retention"
    ] == pytest.approx(0.95)


def test_toe_clearance_profile_changes_only_height_and_ankle_angle():
    baseline_path = Path(
        "config/autonomous_real_only_opening_reverse_270deg_"
        "positive_180_drape_recorded_pose_gripper_coupled_"
        "head_camera_frame_zero_physics_single_centroid_fast_policy_"
        "human_chair_away2cm_down8cm_left_camera_foot_axis_vertical_"
        "opening_preserved.yaml"
    )
    tuned_path = baseline_path.with_name(
        baseline_path.name.replace(
            "down8cm_left_camera",
            "down12cm_plantar30deg_left_camera",
        )
    )
    baseline = load_config(baseline_path)
    tuned = load_config(tuned_path)

    assert tuned["scene"]["camera_parent_link"] == baseline["scene"][
        "camera_parent_link"
    ]
    assert tuned["scene"]["camera_local_position"] == baseline["scene"][
        "camera_local_position"
    ]
    assert tuned["scene"]["camera_local_rotation"] == baseline["scene"][
        "camera_local_rotation"
    ]
    baseline_pose = baseline["scene"]["initial_pose_contract"]
    tuned_pose = tuned["scene"]["initial_pose_contract"]
    for key, value in baseline_pose.items():
        if key != "down_m":
            assert tuned_pose[key] == value
    assert tuned_pose["down_m"] == pytest.approx(0.12)
    assert tuned["scenario"]["sock"] == baseline["scenario"]["sock"]
    assert tuned["scenario"]["foot"]["plantarflexion_degrees"] == pytest.approx(
        30.0
    )
    assert tuned["inference"]["pre_inference_drape"] == baseline["inference"][
        "pre_inference_drape"
    ]
    assert tuned["scene"]["grasp_alignment"] == baseline["scene"][
        "grasp_alignment"
    ]
    assert tuned["obi"]["expected"] == baseline["obi"]["expected"]
    assert tuned["inference"][
        "minimum_grasp_toe_vertical_clearance_m"
    ] == pytest.approx(0.0)


@pytest.mark.parametrize("angle", [5, 10, 15, 18, 20, 21, 22, 24, 25, 27, 35])
def test_foot_passage_profiles_change_only_plantarflexion(angle):
    baseline_path = Path(
        "config/autonomous_real_only_opening_reverse_270deg_"
        "positive_180_drape_recorded_pose_gripper_coupled_"
        "head_camera_frame_zero_physics_single_centroid_fast_policy_"
        "human_chair_away2cm_down12cm_plantar30deg_left_camera_"
        "foot_axis_vertical_opening_preserved.yaml"
    )
    tuned_path = baseline_path.with_name(
        baseline_path.name.replace("plantar30deg", f"plantar{angle}deg")
    )
    baseline = load_config(baseline_path)
    tuned = load_config(tuned_path)

    assert tuned["scenario"]["foot"]["plantarflexion_degrees"] == pytest.approx(
        float(angle)
    )
    assert tuned["scenario"]["sock"] == baseline["scenario"]["sock"]
    assert tuned["scene"] == baseline["scene"]
    assert tuned["obi"] == baseline["obi"]
    assert tuned["joints"] == baseline["joints"]
    assert tuned["inference"]["pre_inference_drape"] == baseline["inference"][
        "pre_inference_drape"
    ]
    for key, value in baseline["inference"].items():
        assert tuned["inference"][key] == value
    assert tuned["inference"][
        "maximum_gripper_foot_region_penetration_m"
    ] == pytest.approx(0.001)
