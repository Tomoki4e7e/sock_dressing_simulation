import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


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


def test_offset_search_accepts_vertical_offsets_through_twelve_cm():
    assert offset_search.parse_centimeter_values(
        "7,8,9,10,11,12"
    ) == [7, 8, 9, 10, 11, 12]
    with pytest.raises(argparse.ArgumentTypeError, match="1 through 12"):
        offset_search.parse_centimeter_values("13")


def test_offset_search_allows_height_screening_when_only_full_pose_warns():
    result = {
        "initial_pose_ok": False,
        "four_point_grasp_attached": True,
        "rectangular_opening_ok": True,
        "human_chair_lock_ok": True,
    }

    assert offset_search.initial_screening_ok(result)
    result["human_chair_lock_ok"] = False
    assert not offset_search.initial_screening_ok(result)


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


def test_offset_search_score_requires_complete_horizon():
    complete = {
        "frames": 20,
        "stop_reason": "max_steps",
        "task_success": {"coverage_gain": 0.1},
    }
    incomplete = {
        "frames": 19,
        "stop_reason": "early_stop",
        "task_success": {
            "success": True,
            "coverage_gain": 1.0,
        },
    }

    assert offset_search.score_metadata(
        complete, 20
    ) > offset_search.score_metadata(incomplete, 20)


def test_offset_search_score_prefers_integrity_then_balanced_progress():
    integrity = {
        "continuous_grasp_ok": True,
        "continuous_opening_span_ok": True,
        "human_chair_lock_ok": True,
        "rigid_collision_ok": True,
        "continuous_stretch_ok": True,
        "cloth_foot_penetration_ok": True,
        "opening_body_penetration_ok": True,
    }
    safe = {
        "task_success": {
            **integrity,
            "coverage_gain": 0.2,
            "cuff_progress_toward_ankle_m": 0.06,
        }
    }
    unsafe = {
        "task_success": {
            "coverage_gain": 1.0,
            "cuff_progress_toward_ankle_m": 0.3,
        }
    }
    balanced = {
        "task_success": {
            **integrity,
            "coverage_gain": 0.4,
            "cuff_progress_toward_ankle_m": 0.18,
        }
    }
    coverage_only = {
        "task_success": {
            **integrity,
            "coverage_gain": 0.6,
            "cuff_progress_toward_ankle_m": 0.06,
        }
    }

    assert offset_search.score_metadata(safe) > offset_search.score_metadata(
        unsafe
    )
    assert offset_search.score_metadata(
        balanced
    ) > offset_search.score_metadata(coverage_only)


def test_offset_search_demo_stage_records_left_camera_video(
    tmp_path, monkeypatch
):
    episode = tmp_path / "episode"
    episode.mkdir()
    metadata = episode / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "frames": 20,
                "stop_reason": "max_steps",
                "task_success": {},
            }
        ),
        encoding="utf-8",
    )
    (episode / "demo.mp4").touch()
    (episode / "inference_camera.mp4").touch()
    monkeypatch.setattr(offset_search, "_run", lambda *_args: 0)
    monkeypatch.setattr(
        offset_search, "_latest_metadata", lambda _root: metadata
    )

    result = offset_search._demo_stage(
        offset_search.Candidate(2, 7),
        tmp_path / "candidate",
        tmp_path / "config.yaml",
        "short",
        20,
        0,
        False,
    )

    assert result["frames"] == 20
    assert result["score"][0] == 1
    assert result["inference_camera_video"].endswith(
        "inference_camera.mp4"
    )


def test_offset_search_resume_requires_matching_step_count(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(
        json.dumps(
            {
                "completed": True,
                "requested_steps": 10,
                "frames": 10,
                "stop_reason": "max_steps",
            }
        ),
        encoding="utf-8",
    )

    assert offset_search.completed_result(result)
    assert offset_search.completed_result(result, 10)
    assert not offset_search.completed_result(result, 50)
    result.write_text(
        json.dumps(
            {
                "completed": True,
                "requested_steps": 10,
                "frames": 0,
                "stop_reason": "fail_closed",
            }
        ),
        encoding="utf-8",
    )
    assert not offset_search.completed_result(result, 10)
