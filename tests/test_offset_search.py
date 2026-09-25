import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path("scripts/search_human_chair_offsets.py")
SPEC = importlib.util.spec_from_file_location("offset_search", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
offset_search = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = offset_search
SPEC.loader.exec_module(offset_search)


def test_offset_search_builds_complete_deterministic_grid():
    candidates = offset_search.grid_candidates()

    assert len(candidates) == 64
    assert candidates[0].name == "away_01cm_down_01cm"
    assert candidates[-1].name == "away_08cm_down_08cm"
    assert len({candidate.name for candidate in candidates}) == 64


def test_offset_search_score_prioritizes_task_success_then_gates():
    successful = {
        "frames": 10,
        "task_success": {
            "success": True,
            "initial_pose_ok": True,
            "failed_gates": ["coverage_gain_ok"],
        },
    }
    unsuccessful = {
        "frames": 50,
        "task_success": {
            "success": False,
            "initial_pose_ok": True,
            "failed_gates": [],
            "final_surface_containment_ratio": 1.0,
        },
    }
    fewer_failures = {
        "task_success": {
            "success": False,
            "initial_pose_ok": True,
            "failed_gates": ["coverage_gain_ok"],
        }
    }
    more_failures = {
        "task_success": {
            "success": False,
            "initial_pose_ok": True,
            "failed_gates": ["coverage_gain_ok", "distal_follow_ok"],
        }
    }

    assert offset_search.score_metadata(successful) > offset_search.score_metadata(
        unsuccessful
    )
    assert offset_search.score_metadata(
        fewer_failures
    ) > offset_search.score_metadata(more_failures)


def test_offset_search_resume_requires_matching_step_count(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps({"completed": True, "requested_steps": 10}),
        encoding="utf-8",
    )

    assert offset_search.completed_result(result)
    assert offset_search.completed_result(result, 10)
    assert not offset_search.completed_result(result, 50)
