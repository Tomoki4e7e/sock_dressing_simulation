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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / (
    "autonomous_real_only_opening_reverse_270deg_positive_180_drape_"
    "recorded_pose_gripper_coupled_head_camera_frame_zero_physics_"
    "single_centroid_fast_policy_human_chair_away2cm_down12cm_"
    "plantar20deg_left_camera_foot_axis_vertical_opening_preserved.yaml"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "phase4" / "opening-short-edge-search"


@dataclass(frozen=True)
class Candidate:
    name: str
    half_width_m: float

    @property
    def nominal_short_edge_m(self) -> float:
        return 2.0 * self.half_width_m


def candidates() -> list[Candidate]:
    return [
        Candidate(f"edge_{int(round(value * 2000)):02d}mm", value)
        for value in (0.025, 0.030, 0.0325, 0.035, 0.0375, 0.040, 0.0425)
    ]


def candidate_config(candidate: Candidate, base_config: Path) -> dict:
    return {
        "extends": str(base_config.resolve()),
        "obi": {
            "expected": {
                "grasp_thickness_half_width_m": candidate.half_width_m
            }
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _latest_metadata(output: Path) -> Path | None:
    matches = sorted(
        output.glob("data_sock_sim_smoke/train/phase4_*/metadata.json"),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def completed_metadata(path: Path | None, expected_steps: int) -> Path | None:
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


def geometry_report(metadata: Mapping[str, Any]) -> dict:
    geometry = dict(
        metadata.get("diagnostics", {}).get("scene_geometry", {}) or {}
    )
    left = geometry.get("left_grasp_patch_span_m")
    right = geometry.get("right_grasp_patch_span_m")
    measured = [
        float(value) for value in (left, right) if value is not None
    ]
    return {
        "left_grasp_patch_span_m": left,
        "right_grasp_patch_span_m": right,
        "mean_grasp_patch_span_m": (
            sum(measured) / len(measured) if measured else None
        ),
        "opening_ring_area_m2": geometry.get("opening_ring_area_m2"),
        "opening_ring_area_retention": geometry.get(
            "opening_ring_area_retention"
        ),
        "opening_span_m": geometry.get("opening_span_m"),
        "rectangular_opening_ok": (
            metadata.get("scenario_application", {})
            .get("initial_pose_contract", {})
            .get("rectangular_opening_ok")
        ),
    }


def score_metadata(metadata: Mapping[str, Any]) -> tuple:
    task = dict(metadata.get("task_success", {}) or {})
    rebound = dict(task.get("contact_rebound", {}) or {})
    cloth_ok = sum(
        bool(item.get("ok", False))
        for item in metadata.get("cloth_quality_by_frame", ()) or ()
    )
    required = (
        "continuous_grasp_ok",
        "continuous_opening_span_ok",
        "grasp_toe_clearance_ok",
        "gripper_foot_passage_ok",
        "human_chair_lock_ok",
    )
    hard_pass = all(bool(task.get(name, False)) for name in required)
    reverse = rebound.get("maximum_post_contact_cuff_reverse_m")
    return (
        int(bool(task.get("success", False))),
        int(hard_pass),
        cloth_ok,
        -float(reverse if reverse is not None else 1e9),
        float(task.get("coverage_gain") or 0.0),
        float(task.get("cuff_progress_toward_ankle_m") or 0.0),
        -float(task.get("maximum_cloth_foot_penetration_m") or 1e9),
        -float(
            task.get("maximum_gripper_foot_region_penetration_m") or 1e9
        ),
    )


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
        completed_metadata(_latest_metadata(run_root), max_steps)
        if resume
        else None
    )
    if metadata_path is None:
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
        log_path = run_root / "run.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            returncode = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        metadata_path = completed_metadata(
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
        "nominal_short_edge_m": candidate.nominal_short_edge_m,
        "completed": True,
        "metadata": str(metadata_path.resolve()),
        "episode": str(metadata_path.parent.resolve()),
        "frames": int(metadata.get("frames", 0)),
        "stop_reason": metadata.get("stop_reason"),
        "geometry": geometry_report(metadata),
        "task_success": bool(task.get("success", False)),
        "failed_gates": list(task.get("failed_gates", ()) or ()),
        "continuous_grasp_ok": task.get("continuous_grasp_ok"),
        "continuous_opening_span_ok": task.get(
            "continuous_opening_span_ok"
        ),
        "grasp_toe_clearance_ok": task.get("grasp_toe_clearance_ok"),
        "gripper_foot_passage_ok": task.get("gripper_foot_passage_ok"),
        "human_chair_lock_ok": task.get("human_chair_lock_ok"),
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
        "maximum_cloth_foot_penetration_m": task.get(
            "maximum_cloth_foot_penetration_m"
        ),
        "contact_rebound": task.get("contact_rebound"),
        "cloth_qa_ok_frames": sum(
            bool(item.get("ok", False))
            for item in metadata.get("cloth_quality_by_frame", ()) or ()
        ),
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
