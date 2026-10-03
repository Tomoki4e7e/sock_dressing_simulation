import importlib.util
import json
import sys
from pathlib import Path

import yaml


SCRIPT = Path("scripts/search_opening_short_edge.py")
WINNING_CONFIG = Path(
    "config/autonomous_real_only_opening_short_edge_75mm.yaml"
)
SPEC = importlib.util.spec_from_file_location("opening_short_edge_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
opening_short_edge_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = opening_short_edge_search
SPEC.loader.exec_module(opening_short_edge_search)


def _metadata(*, passage_ok: bool, cloth_ok: int) -> dict:
    return {
        "task_success": {
            "success": False,
            "continuous_grasp_ok": True,
            "continuous_opening_span_ok": True,
            "grasp_toe_clearance_ok": True,
            "gripper_foot_passage_ok": passage_ok,
            "human_chair_lock_ok": True,
            "coverage_gain": 0.5,
            "cuff_progress_toward_ankle_m": 0.1,
            "maximum_cloth_foot_penetration_m": 0.001,
            "maximum_gripper_foot_region_penetration_m": 0.0005,
            "contact_rebound": {
                "maximum_post_contact_cuff_reverse_m": 0.001
            },
        },
        "cloth_quality_by_frame": [{"ok": True}] * cloth_ok,
    }


def test_short_edge_grid_covers_nominal_fifty_through_eighty_five_mm():
    candidates = opening_short_edge_search.candidates()

    assert len(candidates) == 7
    assert candidates[0].name == "edge_50mm"
    assert candidates[-1].name == "edge_85mm"
    assert candidates[0].nominal_short_edge_m == 0.05
    assert candidates[-1].nominal_short_edge_m == 0.085


def test_candidate_config_changes_only_grasp_half_width(tmp_path):
    candidate = opening_short_edge_search.candidates()[3]
    payload = opening_short_edge_search.candidate_config(
        candidate, tmp_path / "baseline.yaml"
    )

    assert set(payload) == {"extends", "obi"}
    assert payload["obi"] == {
        "expected": {
            "grasp_thickness_half_width_m": candidate.half_width_m
        }
    }


def test_winning_config_overrides_only_short_edge_half_width():
    payload = yaml.safe_load(WINNING_CONFIG.read_text())

    assert set(payload) == {"extends", "obi"}
    assert payload["extends"] == (
        "autonomous_real_only_opening_reverse_270deg_positive_180_drape_"
        "recorded_pose_gripper_coupled_head_camera_frame_zero_physics_"
        "single_centroid_fast_policy_human_chair_away2cm_down12cm_"
        "plantar20deg_left_camera_foot_axis_vertical_opening_preserved.yaml"
    )
    assert payload["obi"] == {
        "expected": {"grasp_thickness_half_width_m": 0.0375}
    }


def test_short_edge_score_prioritizes_foot_passage_over_cloth_count():
    safe = _metadata(passage_ok=True, cloth_ok=1)
    colliding = _metadata(passage_ok=False, cloth_ok=60)

    assert opening_short_edge_search.score_metadata(
        safe
    ) > opening_short_edge_search.score_metadata(colliding)


def test_short_edge_resume_rejects_incomplete_metadata(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"frames": 0, "stop_reason": "fail_closed"}))
    assert opening_short_edge_search.completed_metadata(path, 60) is None

    path.write_text(json.dumps({"frames": 60, "stop_reason": "max_steps"}))
    assert opening_short_edge_search.completed_metadata(path, 60) == path
