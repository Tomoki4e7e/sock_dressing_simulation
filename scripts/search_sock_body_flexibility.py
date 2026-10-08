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
    "autonomous_real_only_opening_short_edge_75mm.yaml"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts" / "phase4" / "sock-body-flexibility-search"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    settings: Mapping[str, float | int]


def candidates(stage: str = "material") -> list[Candidate]:
    if stage == "barrier":
        return [
            Candidate(
                "barrier_baseline",
                {
                    "opening_body_barrier_stiffness": 1.0,
                    "opening_body_barrier_maximum_correction_m": 0.030,
                },
            ),
            Candidate(
                "barrier_moderate",
                {
                    "opening_body_barrier_stiffness": 0.5,
                    "opening_body_barrier_maximum_correction_m": 0.015,
                },
            ),
            Candidate(
                "barrier_soft",
                {
                    "opening_body_barrier_stiffness": 0.25,
                    "opening_body_barrier_maximum_correction_m": 0.0075,
                },
            ),
        ]
    if stage != "material":
        raise ValueError(f"unknown stage: {stage}")
    return [
        Candidate(
            "body_baseline",
            {
                "stretch_compliance": 1e-5,
                "bend_compliance": 0.10,
                "damping": 0.95,
                "strain_limit_iterations": 240,
            },
        ),
        Candidate(
            "stretch_5e5",
            {
                "stretch_compliance": 5e-5,
                "bend_compliance": 0.10,
                "damping": 0.95,
                "strain_limit_iterations": 240,
            },
        ),
        Candidate(
            "stretch_1e4",
            {
                "stretch_compliance": 1e-4,
                "bend_compliance": 0.10,
                "damping": 0.95,
                "strain_limit_iterations": 240,
            },
        ),
        Candidate(
            "stretch_5e4",
            {
                "stretch_compliance": 5e-4,
                "bend_compliance": 0.10,
                "damping": 0.95,
                "strain_limit_iterations": 240,
            },
        ),
        Candidate(
            "bend_20_stretch_1e4",
            {
                "stretch_compliance": 1e-4,
                "bend_compliance": 0.20,
                "damping": 0.95,
                "strain_limit_iterations": 240,
            },
        ),
        Candidate(
            "damping_75_stretch_1e4",
            {
                "stretch_compliance": 1e-4,
                "bend_compliance": 0.10,
                "damping": 0.75,
                "strain_limit_iterations": 240,
            },
        ),
        Candidate(
            "strain_320_stretch_1e4",
            {
                "stretch_compliance": 1e-4,
                "bend_compliance": 0.10,
                "damping": 0.95,
                "strain_limit_iterations": 320,
            },
        ),
    ]


def candidate_config(candidate: Candidate, base_config: Path) -> dict:
    return {
        "extends": str(base_config.resolve()),
        "obi": {"expected": dict(candidate.settings)},
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


def body_stretch_report(metadata: Mapping[str, Any]) -> dict:
    values = []
    passing_frames = 0
    for item in metadata.get("cloth_quality_by_frame", ()) or ():
        stretch = dict(item.get("stretch", {}) or {})
        if stretch.get("passes"):
            passing_frames += 1
        body = dict(
            (stretch.get("edge_classes", {}) or {}).get("body_body", {}) or {}
        )
        value = body.get("maximum_stretch")
        if value is not None:
            values.append(float(value))
    return {
        "maximum_body_body_stretch": max(values, default=None),
        "stretch_qa_ok_frames": passing_frames,
    }


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
    stretch = body_stretch_report(metadata)
    rebound = rebound_report(metadata)
    integrity = all(
        bool(task.get(name, False))
        for name in (
            "continuous_grasp_ok",
            "continuous_opening_span_ok",
            "human_chair_lock_ok",
        )
    )
    physical = all(
        bool(task.get(name, False))
        for name in (
            "continuous_stretch_ok",
            "cloth_foot_penetration_ok",
            "gripper_foot_passage_ok",
        )
    )
    containment = all(
        bool(task.get(name, False))
        for name in (
            "final_surface_containment_ok",
            "final_section_containment_ok",
            "cuff_progress_ok",
        )
    )
    cloth_ok = sum(
        bool(item.get("ok", False))
        for item in metadata.get("cloth_quality_by_frame", ()) or ()
    )
    maximum_stretch = stretch["maximum_body_body_stretch"]
    cloth_penetration = task.get("maximum_cloth_foot_penetration_m")
    reverse = rebound.get("maximum_post_contact_cuff_reverse_m")
    return (
        int(bool(task.get("success", False))),
        int(integrity and physical and containment),
        int(physical),
        cloth_ok,
        stretch["stretch_qa_ok_frames"],
        -float(maximum_stretch if maximum_stretch is not None else 1e9),
        -float(cloth_penetration if cloth_penetration is not None else 1e9),
        -float(reverse if reverse is not None else 1e9),
        float(task.get("coverage_gain") or 0.0),
        float(task.get("cuff_progress_toward_ankle_m") or 0.0),
    )


def _write_candidate_config(
    candidate: Candidate, base_config: Path, output_root: Path
) -> Path:
    path = output_root / candidate.name / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(candidate_config(candidate, base_config), sort_keys=False)
    )
    return path


def _run_preflight(
    candidate: Candidate, *, base_config: Path, output_root: Path
) -> dict:
    config_path = _write_candidate_config(candidate, base_config, output_root)
    output = output_root / candidate.name / "acceptance.json"
    log = output_root / candidate.name / "acceptance.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "live_acceptance.py"),
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    with log.open("w", encoding="utf-8") as stream:
        returncode = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    report = json.loads(output.read_text()) if output.exists() else {}
    return {
        "candidate": candidate.name,
        "settings": dict(candidate.settings),
        "completed": bool(report),
        "acceptance_ok": bool(report.get("ok", False)),
        "returncode": returncode,
        "output": str(output.resolve()),
        "maximum_particle_ring_stretch_proxy": report.get(
            "maximum_particle_ring_stretch_proxy"
        ),
        "continuous_grasp": report.get("continuous_grasp"),
        "dressing_qa_finite": report.get("dressing_qa_finite"),
        "final_dressing_qa": report.get("final_dressing_qa"),
    }


def _run_candidate(
    candidate: Candidate,
    *,
    base_config: Path,
    output_root: Path,
    max_steps: int,
    seed: int,
    resume: bool,
) -> dict:
    config_path = _write_candidate_config(candidate, base_config, output_root)
    run_root = output_root / candidate.name / f"autonomous-{max_steps}-seed{seed}"
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
        log = run_root / "run.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as stream:
            returncode = subprocess.run(
                command,
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        metadata_path = completed_metadata(_latest_metadata(run_root), max_steps)
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
        "settings": dict(candidate.settings),
        "completed": True,
        "metadata": str(metadata_path.resolve()),
        "episode": str(metadata_path.parent.resolve()),
        "frames": int(metadata.get("frames", 0)),
        "stop_reason": metadata.get("stop_reason"),
        "task_success": bool(task.get("success", False)),
        "failed_gates": list(task.get("failed_gates", ()) or ()),
        "continuous_grasp_ok": task.get("continuous_grasp_ok"),
        "continuous_opening_span_ok": task.get("continuous_opening_span_ok"),
        "continuous_stretch_ok": task.get("continuous_stretch_ok"),
        "cloth_foot_penetration_ok": task.get("cloth_foot_penetration_ok"),
        "gripper_foot_passage_ok": task.get("gripper_foot_passage_ok"),
        "final_surface_containment_ok": task.get(
            "final_surface_containment_ok"
        ),
        "final_section_containment_ok": task.get(
            "final_section_containment_ok"
        ),
        "cuff_progress_ok": task.get("cuff_progress_ok"),
        "coverage_gain": task.get("coverage_gain"),
        "cuff_progress_toward_ankle_m": task.get(
            "cuff_progress_toward_ankle_m"
        ),
        "maximum_cloth_foot_penetration_m": task.get(
            "maximum_cloth_foot_penetration_m"
        ),
        "maximum_gripper_foot_region_penetration_m": task.get(
            "maximum_gripper_foot_region_penetration_m"
        ),
        "cloth_qa_ok_frames": sum(
            bool(item.get("ok", False))
            for item in metadata.get("cloth_quality_by_frame", ()) or ()
        ),
        "body_stretch": body_stretch_report(metadata),
        "contact_rebound": rebound_report(metadata),
    }
    result["score"] = list(score_metadata(metadata))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("material", "barrier"), default="material")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--names", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    selected = candidates(args.stage)
    if args.names:
        names = {item.strip() for item in args.names.split(",") if item.strip()}
        selected = [item for item in selected if item.name in names]
        missing = names - {item.name for item in selected}
        if missing:
            parser.error(f"unknown candidates: {sorted(missing)}")
    if args.preflight_only:
        results = [
            _run_preflight(
                item,
                base_config=args.base_config,
                output_root=args.output_root,
            )
            for item in selected
        ]
        summary = {
            "base_config": str(args.base_config.resolve()),
            "stage": args.stage,
            "preflight": results,
        }
        _write_json(
            args.output_root / f"preflight-{args.stage}.json", summary
        )
        print(json.dumps(summary, indent=2))
        return 0 if all(item["completed"] for item in results) else 1
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
        "stage": args.stage,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "ranking": completed,
        "incomplete": [
            item for item in results if not item.get("completed")
        ],
    }
    _write_json(
        args.output_root
        / f"summary-{args.stage}-{args.max_steps}-seed{args.seed}.json",
        summary,
    )
    print(json.dumps(summary, indent=2))
    return 0 if len(completed) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
