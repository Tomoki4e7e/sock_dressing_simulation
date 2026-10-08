import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path("scripts/search_sock_body_flexibility.py")
SPEC = importlib.util.spec_from_file_location("body_flex_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
body_flex_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = body_flex_search
SPEC.loader.exec_module(body_flex_search)


def _metadata(
    *,
    success: bool = False,
    physical: bool = True,
    maximum_stretch: float = 1.2,
    penetration: float = 0.001,
) -> dict:
    return {
        "task_success": {
            "success": success,
            "continuous_grasp_ok": True,
            "continuous_opening_span_ok": True,
            "human_chair_lock_ok": True,
            "continuous_stretch_ok": physical,
            "cloth_foot_penetration_ok": physical,
            "gripper_foot_passage_ok": physical,
            "final_surface_containment_ok": physical,
            "final_section_containment_ok": physical,
            "cuff_progress_ok": physical,
            "maximum_cloth_foot_penetration_m": penetration,
            "coverage_gain": 0.5,
            "cuff_progress_toward_ankle_m": 0.1,
            "contact_rebound": {
                "maximum_post_contact_cuff_reverse_m": 0.001
            },
        },
        "cloth_quality_by_frame": [
            {
                "ok": physical,
                "stretch": {
                    "passes": physical,
                    "edge_classes": {
                        "body_body": {"maximum_stretch": maximum_stretch}
                    },
                },
            }
        ],
    }


def test_material_grid_is_bounded_and_softens_by_higher_compliance():
    candidates = body_flex_search.candidates("material")

    assert len(candidates) == 7
    assert candidates[0].name == "body_baseline"
    assert candidates[0].settings["stretch_compliance"] == 1e-5
    assert any(
        item.settings["stretch_compliance"] == 5e-4
        for item in candidates
    )
    assert any(
        item.settings["bend_compliance"] == 0.20 for item in candidates
    )


def test_material_candidate_changes_only_body_physics(tmp_path):
    candidate = body_flex_search.candidates("material")[2]
    payload = body_flex_search.candidate_config(
        candidate, tmp_path / "baseline.yaml"
    )

    assert set(payload) == {"extends", "obi"}
    assert set(payload["obi"]) == {"expected"}
    assert set(payload["obi"]["expected"]) == {
        "stretch_compliance",
        "bend_compliance",
        "damping",
        "strain_limit_iterations",
    }


def test_barrier_candidate_changes_only_body_barrier_response(tmp_path):
    candidate = body_flex_search.candidates("barrier")[1]
    payload = body_flex_search.candidate_config(
        candidate, tmp_path / "baseline.yaml"
    )

    assert set(payload["obi"]["expected"]) == {
        "opening_body_barrier_stiffness",
        "opening_body_barrier_maximum_correction_m",
    }


def test_score_prioritizes_full_success_then_physical_integrity():
    successful = _metadata(
        success=True, maximum_stretch=1.4, penetration=0.002
    )
    prettier_failure = _metadata(
        success=False, maximum_stretch=1.1, penetration=0.0001
    )
    unsafe = _metadata(physical=False, maximum_stretch=1.05)

    assert body_flex_search.score_metadata(
        successful
    ) > body_flex_search.score_metadata(prettier_failure)
    assert body_flex_search.score_metadata(
        prettier_failure
    ) > body_flex_search.score_metadata(unsafe)


def test_body_stretch_report_uses_body_body_edges():
    report = body_flex_search.body_stretch_report(
        _metadata(maximum_stretch=1.35)
    )

    assert report["maximum_body_body_stretch"] == 1.35
    assert report["stretch_qa_ok_frames"] == 1


def test_resume_rejects_incomplete_episode_metadata(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"frames": 53, "stop_reason": "fail_closed"}))
    assert body_flex_search.completed_metadata(path, 60) is None

    path.write_text(json.dumps({"frames": 60, "stop_reason": "max_steps"}))
    assert body_flex_search.completed_metadata(path, 60) == path
