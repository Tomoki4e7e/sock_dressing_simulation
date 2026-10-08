#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from sock_dressing_simulation.demo import _contact_rebound_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / (
    "autonomous_real_only_opening_reverse_270deg_positive_180_drape_"
    "recorded_pose_gripper_coupled_head_camera_frame_zero_physics_"
    "single_centroid_fast_policy_human_chair_away2cm_down12cm_"
    "plantar20deg_left_camera_foot_axis_vertical_opening_preserved.yaml"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase4" / "opening-flexibility-search"


@dataclass(frozen=True)
class Candidate:
    name: str
    plane_stiffness: float
    shape_stiffness: float
    span_shape_stiffness: float
    maximum_correction_m: float


def candidates() -> list[Candidate]:
    return [
        Candidate("baseline", 0.75, 0.50, 0.25, 0.010),
        Candidate("moderate_a", 0.65, 0.40, 0.20, 0.0075),
        Candidate("moderate_b", 0.50, 0.35, 0.20, 0.0075),
        Candidate("soft_a", 0.50, 0.25, 0.15, 0.005),
        Candidate("soft_b", 0.35, 0.25, 0.15, 0.005),
        Candidate("soft_plane", 0.35, 0.35, 0.20, 0.005),
        Candidate("minimum_restore", 0.20, 0.15, 0.10, 0.003),
    ]


def candidate_config(candidate: Candidate, base_config: Path) -> dict:
    return {
        "extends": str(base_config.resolve()),
        "obi": {
            "expected": {
                "opening_rim_plane_stiffness": candidate.plane_stiffness,
                "opening_rim_shape_stiffness": candidate.shape_stiffness,
                "opening_rim_span_shape_stiffness": (
                    candidate.span_shape_stiffness
                ),
                "opening_rim_maximum_correction_m": (
                    candidate.maximum_correction_m
                ),
            }
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rebound_report(metadata: Mapping[str, Any]) -> dict:
    task = dict(metadata.get("task_success", {}) or {})
    saved = task.get("contact_rebound")
    if isinstance(saved, Mapping):
        return dict(saved)
    return _contact_rebound_report(
        metadata.get("dressing_quality_by_frame", ()) or (),
        metadata.get("grasp_quality_by_frame", ()) or (),
        metadata.get("foot_contact_ids_by_frame", ()) or (),
    )


def score_metadata(metadata: Mapping[str, Any]) -> tuple:
    task = dict(metadata.get("task_success", {}) or {})
    rebound = rebound_report(metadata)
    cloth_ok = sum(
        bool(item.get("ok", False))
        for item in metadata.get("cloth_quality_by_frame", ()) or ()
    )
    required_gates = (
        "continuous_grasp_ok",
        "continuous_opening_span_ok",
        "grasp_toe_clearance_ok",
        "human_chair_lock_ok",
    )
    safe = all(bool(task.get(name, False)) for name in required_gates)
    stretch_safe = bool(task.get("stretch_ok", False))
    reverse = rebound.get("maximum_post_contact_cuff_reverse_m")
    progress_drop = rebound.get("maximum_post_contact_cuff_progress_drop_m")
    return (
        int(safe and stretch_safe),
        int(bool(task.get("success", False))),
        -float(reverse if reverse is not None else 1e9),
        -float(progress_drop if progress_drop is not None else 1e9),
        cloth_ok,
        float(task.get("coverage_gain") or 0.0),
        float(task.get("cuff_progress_toward_ankle_m") or 0.0),
        -float(
            task.get("maximum_gripper_foot_region_penetration_m") or 0.0
        ),
    )


def _latest_metadata(output: Path) -> Path | None:
    matches = sorted(
        output.glob("data_sock_sim_smoke/train/phase4_*/metadata.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _completed_metadata(path: Path | None, expected_steps: int) -> Path | None:
    if path is None:
        return None
    try:
        metadata = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if (
        int(metadata.get("frames", -1)) != int(expected_steps)
        or metadata.get("stop_reason") != "max_steps"
    ):
        return None
    return path


def _run_candidate(
    candidate: Candidate,
    *,
    base_config: Path,
    output_root: Path,
    max_steps: int,
    seed: int,
    resume: bool,
) -> dict:
    directory = output_root / candidate.name
    config_path = directory / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            candidate_config(candidate, base_config), sort_keys=False
        )
    )
    run_root = directory / f"autonomous-{max_steps}-seed{seed}"
    metadata_path = (
        _completed_metadata(_latest_metadata(run_root), max_steps)
        if resume
        else None
    )
    if metadata_path is None:
        log_path = run_root / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "sock_dressing_simulation.cli",
            "--config",
            str(config_path),
            "demo",
            "--graphics",
            "--max-steps",
            str(max_steps),
            "--seed",
            str(seed),
            "--output-root",
            str(run_root),
        ]
        with log_path.open("w", encoding="utf-8") as stream:
            returncode = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        metadata_path = _completed_metadata(
            _latest_metadata(run_root), max_steps
        )
        if returncode or metadata_path is None:
            return {
                "candidate": candidate.name,
                "completed": False,
                "returncode": returncode,
            }
    metadata = json.loads(metadata_path.read_text())
    task = dict(metadata.get("task_success", {}) or {})
    result = {
        "candidate": candidate.name,
        "settings": asdict(candidate),
        "completed": True,
        "metadata": str(metadata_path.resolve()),
        "episode": str(metadata_path.parent.resolve()),
        "frames": int(metadata.get("frames", 0)),
        "stop_reason": metadata.get("stop_reason"),
        "task_success": bool(task.get("success", False)),
        "failed_gates": list(task.get("failed_gates", ()) or ()),
        "coverage_gain": task.get("coverage_gain"),
        "cuff_progress_toward_ankle_m": task.get(
            "cuff_progress_toward_ankle_m"
        ),
        "minimum_opening_span_m": task.get("minimum_opening_span_m"),
        "minimum_opening_ring_area_retention": task.get(
            "minimum_opening_ring_area_retention"
        ),
        "maximum_gripper_foot_region_penetration_m": task.get(
            "maximum_gripper_foot_region_penetration_m"
        ),
        "cloth_qa_ok_frames": sum(
            bool(item.get("ok", False))
            for item in metadata.get("cloth_quality_by_frame", ()) or ()
        ),
        "contact_rebound": rebound_report(metadata),
    }
    result["score"] = list(score_metadata(metadata))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--names", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    selected = candidates()
    if args.names:
        names = {item.strip() for item in args.names.split(",") if item.strip()}
        selected = [item for item in selected if item.name in names]
        missing = names - {item.name for item in selected}
        if missing:
            parser.error(f"unknown candidates: {sorted(missing)}")
    results = [
        _run_candidate(
            item,
            base_config=args.base_config,
            output_root=args.output_root,
            max_steps=args.max_steps,
            seed=args.seed,
            resume=args.resume,
        )
        for item in selected
    ]
    completed = [item for item in results if item.get("completed")]
    completed.sort(key=lambda item: tuple(item["score"]), reverse=True)
    summary = {
        "base_config": str(args.base_config.resolve()),
        "max_steps": args.max_steps,
        "seed": args.seed,
        "ranking": completed,
        "incomplete": [
            item for item in results if not item.get("completed")
        ],
    }
    _write_json(
        args.output_root / f"summary-{args.max_steps}-seed{args.seed}.json",
        summary,
    )
    print(json.dumps(summary, indent=2))
    return 0 if len(completed) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
