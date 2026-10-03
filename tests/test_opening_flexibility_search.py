import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path("scripts/search_opening_flexibility.py")
SPEC = importlib.util.spec_from_file_location("opening_flex_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
opening_flex_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = opening_flex_search
SPEC.loader.exec_module(opening_flex_search)


def _metadata(reverse: float) -> dict:
    return {
        "task_success": {
            "success": False,
            "continuous_grasp_ok": True,
            "continuous_opening_span_ok": True,
            "grasp_toe_clearance_ok": True,
            "human_chair_lock_ok": True,
            "stretch_ok": True,
            "coverage_gain": 0.5,
            "cuff_progress_toward_ankle_m": 0.1,
            "maximum_gripper_foot_region_penetration_m": 0.0005,
            "contact_rebound": {
                "maximum_post_contact_cuff_reverse_m": reverse,
                "maximum_post_contact_cuff_progress_drop_m": reverse / 2,
            },
        },
        "cloth_quality_by_frame": [{"ok": True}, {"ok": False}],
    }


def test_opening_flexibility_grid_is_bounded_and_includes_baseline():
    candidates = opening_flex_search.candidates()

    assert len(candidates) == 7
    assert candidates[0].name == "baseline"
    assert candidates[0].plane_stiffness == 0.75
    assert candidates[-1].name == "minimum_restore"
    assert len({candidate.name for candidate in candidates}) == len(candidates)


def test_candidate_config_changes_only_opening_rim_restoration(tmp_path):
    candidate = opening_flex_search.candidates()[3]
    payload = opening_flex_search.candidate_config(
        candidate, tmp_path / "baseline.yaml"
    )

    assert set(payload) == {"extends", "obi"}
    assert set(payload["obi"]) == {"expected"}
    assert set(payload["obi"]["expected"]) == {
        "opening_rim_plane_stiffness",
        "opening_rim_shape_stiffness",
        "opening_rim_span_shape_stiffness",
        "opening_rim_maximum_correction_m",
    }


def test_opening_flexibility_score_prefers_less_post_contact_rebound():
    low_rebound = _metadata(0.001)
    high_rebound = _metadata(0.004)

    assert opening_flex_search.score_metadata(
        low_rebound
    ) > opening_flex_search.score_metadata(high_rebound)


def test_resume_rejects_incomplete_episode_metadata(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"frames": 0, "stop_reason": "fail_closed"}))

    assert opening_flex_search._completed_metadata(path, 60) is None

    path.write_text(json.dumps({"frames": 60, "stop_reason": "max_steps"}))
    assert opening_flex_search._completed_metadata(path, 60) == path


