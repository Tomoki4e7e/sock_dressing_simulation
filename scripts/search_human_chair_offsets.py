#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "autonomous_real_only_human_chair_offset_search.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase4" / "human-chair-offset-search"


@dataclass(frozen=True)
class Candidate:
    away_cm: int
    down_cm: int

    @property
    def name(self) -> str:
        return f"away_{self.away_cm:02d}cm_down_{self.down_cm:02d}cm"


def grid_candidates(
    away_values: Iterable[int] = range(1, 9),
    down_values: Iterable[int] = range(1, 9),
) -> list[Candidate]:
    return [
        Candidate(int(away), int(down))
        for away in away_values
        for down in down_values
    ]


def parse_centimeter_values(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 or item > 12 for item in values):
        raise argparse.ArgumentTypeError(
            "offset values must be comma-separated integers from 1 through 12"
        )
    return values


def _metric(
    mapping: Mapping[str, Any], key: str, default: float
) -> float:
    value = mapping.get(key)
    return default if value is None else float(value)


def score_metadata(
    metadata: Mapping[str, Any],
    expected_steps: Optional[int] = None,
) -> tuple:
    task = dict(metadata.get("task_success", {}) or {})
    failed = list(task.get("failed_gates", ()) or ())
    frames = int(metadata.get("frames", 0))
    complete_horizon = (
        expected_steps is None
        or (
            frames == int(expected_steps)
            and metadata.get("stop_reason") == "max_steps"
        )
    )
    integrity_gates = (
        "continuous_grasp_ok",
        "continuous_opening_span_ok",
        "human_chair_lock_ok",
        "rigid_collision_ok",
        "continuous_stretch_ok",
        "cloth_foot_penetration_ok",
        "opening_body_penetration_ok",
    )
    integrity_passes = sum(
        bool(task.get(name, False)) for name in integrity_gates
    )
    coverage_gain = _metric(task, "coverage_gain", 0.0)
    cuff_progress = _metric(task, "cuff_progress_toward_ankle_m", 0.0)
    normalized_cuff_progress = min(max(cuff_progress / 0.30, 0.0), 1.0)
    balanced_progress = 0.5 * (coverage_gain + normalized_cuff_progress)
    cloth_qa = dict(task.get("cloth_qa", {}) or {})
    return (
        int(complete_horizon),
        int(bool(task.get("success", False))),
        integrity_passes,
        -len(failed),
        balanced_progress,
        _metric(task, "final_surface_containment_ratio", 0.0),
        coverage_gain,
        cuff_progress,
        -_metric(task, "maximum_cloth_foot_penetration_m", 1e9),
        -_metric(task, "maximum_distal_follow_error_m", 1e9),
        -_metric(cloth_qa, "circumferential_stretch_proxy", 1e9),
    )


def completed_result(path: Path, expected_steps: Optional[int] = None) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if expected_steps is None:
        return bool(result.get("completed", False))
    return (
        bool(result.get("completed", False))
        and int(result.get("requested_steps", -1)) == int(expected_steps)
        and int(result.get("frames", -1)) == int(expected_steps)
        and result.get("stop_reason") == "max_steps"
    )


def initial_screening_ok(result: Mapping[str, Any]) -> bool:
    return bool(
        result.get("initial_pose_ok", False)
        or (
            result.get("four_point_grasp_attached", False)
            and result.get("rectangular_opening_ok", False)
            and result.get("human_chair_lock_ok", False)
        )
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_config(
    candidate: Candidate,
    directory: Path,
    base_config: Path,
) -> Path:
    path = directory / "config.yaml"
    payload = {
        "extends": str(base_config.resolve()),
        "scene": {
            "initial_pose_contract": {
                "away_from_robot_m": candidate.away_cm / 100.0,
                "down_m": candidate.down_cm / 100.0,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _run(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _latest_metadata(output_root: Path) -> Optional[Path]:
    matches = sorted(
        output_root.glob("data_sock_sim_smoke/train/phase4_*/metadata.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _initial_stage(
    candidate: Candidate,
    directory: Path,
    config_path: Path,
    resume: bool,
) -> Dict[str, Any]:
    result_path = directory / "initial" / "result.json"
    if resume and completed_result(result_path):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        acceptance = directory / "initial" / "acceptance.json"
        if acceptance.is_file():
            payload = json.loads(acceptance.read_text(encoding="utf-8"))
            grid_offset = (
                payload.get("application", {}).get(
                    "human_chair_grid_offset", {}
                )
                or {}
            )
            result["effective_coordinates"] = dict(
                grid_offset.get("translated_locked_pose_baseline", {}) or {}
            )
            result["actual_right_toe_position"] = grid_offset.get(
                "actual_right_toe_position"
            )
            result["away_axis_xz"] = grid_offset.get("away_axis_xz")
            result["translation_m"] = grid_offset.get("translation_m")
        else:
            log_path = directory / "initial" / "run.log"
            if log_path.is_file():
                for line in reversed(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                ):
                    if line.startswith("RuntimeError:"):
                        result["rejection_reason"] = line.partition(":")[2].strip()
                        break
        result["screening_ok"] = initial_screening_ok(result)
        result["initial_pose_warning"] = (
            None
            if result.get("initial_pose_ok", False)
            else "initial_pose_contract"
        )
        if result["screening_ok"]:
            result["rejection_reason"] = None
        _write_json(result_path, result)
        return result
    acceptance = directory / "initial" / "acceptance.json"
    returncode = _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "live_acceptance.py"),
            "--config",
            str(config_path),
            "--output",
            str(acceptance),
            "--hold-steps",
            "1",
            "--insertion-steps",
            "0",
            "--pull-steps",
            "0",
        ],
        directory / "initial" / "run.log",
    )
    payload: Dict[str, Any] = {}
    if acceptance.is_file():
        payload = json.loads(acceptance.read_text(encoding="utf-8"))
    contract = (
        payload.get("application", {}).get("initial_pose_contract", {})
        if payload
        else {}
    )
    grid_offset = (
        payload.get("application", {}).get("human_chair_grid_offset", {})
        if payload
        else {}
    )
    effective_coordinates = dict(
        grid_offset.get("translated_locked_pose_baseline", {}) or {}
    )
    result = {
        "completed": True,
        "candidate": candidate.name,
        "away_cm": candidate.away_cm,
        "down_cm": candidate.down_cm,
        "returncode": returncode,
        "acceptance": str(acceptance) if acceptance.is_file() else None,
        "initial_pose_ok": bool(contract.get("ok", False)),
        "four_point_grasp_attached": bool(
            contract.get("four_point_grasp_attached", False)
        ),
        "rectangular_opening_ok": bool(
            contract.get("rectangular_opening_ok", False)
        ),
        "human_chair_lock_ok": bool(
            contract.get("human_chair_lock_ok", False)
        ),
        "opening_to_toe_alignment": contract.get(
            "opening_to_toe_alignment"
        ),
        "effective_coordinates": effective_coordinates,
        "actual_right_toe_position": grid_offset.get(
            "actual_right_toe_position"
        ),
        "away_axis_xz": grid_offset.get("away_axis_xz"),
        "translation_m": grid_offset.get("translation_m"),
        "rejection_reason": None,
    }
    if not result["initial_pose_ok"]:
        result["rejection_reason"] = "initial_pose_contract"
    result["screening_ok"] = initial_screening_ok(result)
    result["initial_pose_warning"] = (
        None if result["initial_pose_ok"] else "initial_pose_contract"
    )
    if result["screening_ok"]:
        result["rejection_reason"] = None
    _write_json(result_path, result)
    return result


def _fill_effective_coordinates(
    candidates: Sequence[Candidate],
    state: Dict[str, Dict[str, Any]],
    output_root: Path,
) -> None:
    sample = next(
        (
            item
            for item in state.values()
            if item.get("effective_coordinates")
            and item.get("away_axis_xz")
        ),
        None,
    )
    if sample is None:
        return
    axis = [float(value) for value in sample["away_axis_xz"]]
    sample_delta = [
        axis[0] * float(sample["away_cm"]) / 100.0,
        -float(sample["down_cm"]) / 100.0,
        axis[2] * float(sample["away_cm"]) / 100.0,
    ]
    baseline = {
        key: [
            float(position[index]) - sample_delta[index]
            for index in range(3)
        ]
        for key, position in sample["effective_coordinates"].items()
    }
    for candidate in candidates:
        record = state[candidate.name]
        if record.get("effective_coordinates"):
            continue
        delta = [
            axis[0] * candidate.away_cm / 100.0,
            -candidate.down_cm / 100.0,
            axis[2] * candidate.away_cm / 100.0,
        ]
        coordinates = {
            key: [
                float(position[index]) + delta[index]
                for index in range(3)
            ]
            for key, position in baseline.items()
        }
        record["effective_coordinates"] = coordinates
        result_path = (
            output_root
            / "candidates"
            / candidate.name
            / "initial"
            / "result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["effective_coordinates"] = coordinates
        result["away_axis_xz"] = axis
        result["translation_m"] = delta
        _write_json(result_path, result)


def _demo_stage(
    candidate: Candidate,
    directory: Path,
    config_path: Path,
    stage: str,
    steps: int,
    seed: int,
    resume: bool,
) -> Dict[str, Any]:
    stage_dir = directory / stage
    result_path = stage_dir / "result.json"
    if resume and completed_result(result_path, steps):
        return json.loads(result_path.read_text(encoding="utf-8"))
    output_root = stage_dir / "episodes"
    returncode = _run(
        [
            sys.executable,
            "-m",
            "sock_dressing_simulation.cli",
            "--config",
            str(config_path),
            "demo",
            "--graphics",
            "--max-steps",
            str(steps),
            "--seed",
            str(seed),
            "--output-root",
            str(output_root),
        ],
        stage_dir / "run.log",
    )
    metadata_path = _latest_metadata(output_root)
    metadata: Dict[str, Any] = {}
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    task = dict(metadata.get("task_success", {}) or {})
    episode = metadata_path.parent if metadata_path is not None else None
    video = episode / "demo.mp4" if episode is not None else None
    inference_camera_video = (
        episode / "inference_camera.mp4" if episode is not None else None
    )
    result = {
        "completed": True,
        "candidate": candidate.name,
        "away_cm": candidate.away_cm,
        "down_cm": candidate.down_cm,
        "requested_steps": steps,
        "returncode": returncode,
        "metadata": str(metadata_path) if metadata_path else None,
        "episode": str(episode) if episode else None,
        "video": str(video) if video and video.is_file() else None,
        "inference_camera_video": (
            str(inference_camera_video)
            if inference_camera_video
            and inference_camera_video.is_file()
            else None
        ),
        "frames": int(metadata.get("frames", 0)),
        "stop_reason": metadata.get("stop_reason"),
        "task_success": bool(task.get("success", False)),
        "failed_gates": list(task.get("failed_gates", ()) or ()),
        "score": (
            list(score_metadata(metadata, steps)) if metadata else []
        ),
    }
    _write_json(result_path, result)
    return result


def _ranking_key(result: Mapping[str, Any]) -> tuple:
    return (
        *tuple(result.get("score", ())),
        -int(result.get("away_cm", 0)),
        -int(result.get("down_cm", 0)),
    )


def _ranked_records(
    records: Iterable[Mapping[str, Any]],
    score_field: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        (record for record in records if record.get(score_field)),
        key=lambda record: _ranking_key(
            {
                **record,
                "score": record[score_field],
            }
        ),
        reverse=True,
    )


def _write_summary(
    output_root: Path,
    records: Sequence[Mapping[str, Any]],
    intermediate_candidates: Sequence[str],
    finalists: Sequence[str],
    best: Optional[Mapping[str, Any]],
) -> None:
    summary = {
        "grid_size": len(records),
        "intermediate_candidates": list(intermediate_candidates),
        "finalists": list(finalists),
        "best": dict(best) if best else None,
        "results": list(records),
    }
    _write_json(output_root / "search_summary.json", summary)
    columns = (
        "candidate",
        "away_cm",
        "down_cm",
        "initial_pose_ok",
        "screening_ok",
        "initial_pose_warning",
        "rejection_reason",
        "effective_coordinates",
        "short_frames",
        "short_task_success",
        "intermediate_frames",
        "intermediate_task_success",
        "final_frames",
        "final_task_success",
        "failed_gates",
        "score",
        "video",
        "inference_camera_video",
    )
    with (output_root / "search_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(record.get(key))
                    if isinstance(record.get(key), (list, dict))
                    else record.get(key)
                    for key in columns
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--short-steps", type=int, default=10)
    parser.add_argument("--intermediate-steps", type=int, default=0)
    parser.add_argument("--intermediate-finalists", type=int, default=2)
    parser.add_argument("--final-steps", type=int, default=50)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--away-values",
        type=parse_centimeter_values,
        default=list(range(1, 9)),
    )
    parser.add_argument(
        "--down-values",
        type=parse_centimeter_values,
        default=list(range(1, 9)),
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if (
        args.short_steps < 1
        or args.intermediate_steps < 0
        or args.intermediate_finalists < 1
        or args.final_steps < 1
        or args.finalists < 1
    ):
        parser.error(
            "step counts must be non-negative and finalist counts positive"
        )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume
    candidates = grid_candidates(args.away_values, args.down_values)
    state: Dict[str, Dict[str, Any]] = {}

    for candidate in candidates:
        directory = output_root / "candidates" / candidate.name
        config_path = _candidate_config(candidate, directory, args.config)
        initial = _initial_stage(
            candidate, directory, config_path, resume
        )
        state[candidate.name] = {
            "candidate": candidate.name,
            "away_cm": candidate.away_cm,
            "down_cm": candidate.down_cm,
            "initial_pose_ok": bool(initial["initial_pose_ok"]),
            "screening_ok": bool(initial["screening_ok"]),
            "initial_pose_warning": initial.get("initial_pose_warning"),
            "rejection_reason": initial.get("rejection_reason"),
            "effective_coordinates": initial.get(
                "effective_coordinates", {}
            ),
            "away_axis_xz": initial.get("away_axis_xz"),
            "translation_m": initial.get("translation_m"),
        }

    _fill_effective_coordinates(candidates, state, output_root)

    for candidate in candidates:
        record = state[candidate.name]
        if not record["screening_ok"]:
            continue
        directory = output_root / "candidates" / candidate.name
        short = _demo_stage(
            candidate,
            directory,
            directory / "config.yaml",
            "short",
            args.short_steps,
            args.seed,
            resume,
        )
        record.update(
            {
                "short_frames": short["frames"],
                "short_task_success": short["task_success"],
                "short_failed_gates": short["failed_gates"],
                "short_score": short["score"],
                "short_video": short["video"],
            }
        )

    short_ranked = _ranked_records(state.values(), "short_score")
    by_name = {candidate.name: candidate for candidate in candidates}
    intermediate_names: list[str] = []
    intermediate_ranked: list[Mapping[str, Any]] = []
    if args.intermediate_steps > 0:
        intermediate_names = [
            item["candidate"]
            for item in short_ranked[: args.intermediate_finalists]
        ]
        for name in intermediate_names:
            candidate = by_name[name]
            directory = output_root / "candidates" / name
            intermediate = _demo_stage(
                candidate,
                directory,
                directory / "config.yaml",
                "intermediate",
                args.intermediate_steps,
                args.seed,
                resume,
            )
            state[name].update(
                {
                    "intermediate_frames": intermediate["frames"],
                    "intermediate_task_success": intermediate[
                        "task_success"
                    ],
                    "intermediate_failed_gates": intermediate[
                        "failed_gates"
                    ],
                    "intermediate_score": intermediate["score"],
                    "intermediate_video": intermediate["video"],
                    "intermediate_inference_camera_video": intermediate[
                        "inference_camera_video"
                    ],
                }
            )
        intermediate_ranked = _ranked_records(
            state.values(), "intermediate_score"
        )
        finalist_names = [
            item["candidate"]
            for item in intermediate_ranked[: args.finalists]
        ]
    else:
        finalist_names = [
            item["candidate"] for item in short_ranked[: args.finalists]
        ]
    for name in finalist_names:
        candidate = by_name[name]
        directory = output_root / "candidates" / name
        final = _demo_stage(
            candidate,
            directory,
            directory / "config.yaml",
            "final",
            args.final_steps,
            args.seed,
            resume,
        )
        state[name].update(
            {
                "final_frames": final["frames"],
                "final_task_success": final["task_success"],
                "failed_gates": final["failed_gates"],
                "score": final["score"],
                "video": final["video"],
                "inference_camera_video": final[
                    "inference_camera_video"
                ],
                "metadata": final["metadata"],
            }
        )

    final_ranked = sorted(
        (record for record in state.values() if record.get("score")),
        key=_ranking_key,
        reverse=True,
    )
    best = final_ranked[0] if final_ranked else (
        intermediate_ranked[0]
        if intermediate_ranked
        else (short_ranked[0] if short_ranked else None)
    )
    records = [state[candidate.name] for candidate in candidates]
    _write_summary(
        output_root,
        records,
        intermediate_names,
        finalist_names,
        best,
    )
    if best and best.get("video"):
        best_dir = output_root / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best["video"], best_dir / "demo.mp4")
        if best.get("inference_camera_video"):
            shutil.copy2(
                best["inference_camera_video"],
                best_dir / "inference_camera.mp4",
            )
        _write_json(best_dir / "result.json", best)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "candidates": len(candidates),
                "initial_passed": sum(
                    bool(item["screening_ok"]) for item in records
                ),
                "short_completed": len(short_ranked),
                "intermediate_completed": len(intermediate_ranked),
                "final_completed": len(final_ranked),
                "best": best,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if best else 2


if __name__ == "__main__":
    raise SystemExit(main())
