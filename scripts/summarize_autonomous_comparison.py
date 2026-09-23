#!/usr/bin/env python3
"""Aggregate autonomous rollouts and render synchronized comparison videos."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "artifacts" / "autonomous-comparison"
EVALUATION = BASE / "evaluation"
REPORT = BASE / "report"
VARIANTS = ("real_only", "real_sim", "sim_only")
SEEDS = (0, 1, 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episode(variant: str, seed: int) -> Path:
    candidates = sorted(
        (EVALUATION / variant / f"seed-{seed}").glob(
            "**/phase4_*/metadata.json"
        )
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one rollout for {variant} seed {seed}, found {len(candidates)}"
        )
    return candidates[0].parent


def _read_actions(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)


def _record(variant: str, seed: int) -> dict:
    episode = _episode(variant, seed)
    metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
    projection = metadata.get("reference_action_projection", {})
    predicted = _read_actions(episode / "predicted_action.csv")
    applied = _read_actions(episode / "applied_action.csv")
    compared = min(len(predicted), len(applied))
    task = metadata.get("task_success", {})
    video = episode / "demo.mp4"
    return {
        "variant": variant,
        "seed": seed,
        "success": bool(task.get("success", False)),
        "frames": compared,
        "stop_reason": metadata.get("stop_reason"),
        "coverage_gain": task.get("coverage_gain"),
        "minimum_attached_grippers": task.get("minimum_attached_grippers"),
        "maximum_grasp_edge_error_m": task.get("maximum_grasp_edge_error_m"),
        "maximum_stretch": task.get("cloth_qa", {}).get(
            "circumferential_stretch_proxy"
        ),
        "semantic_masks_ok": task.get("semantic_masks_ok"),
        "continuous_grasp_ok": task.get("continuous_grasp_ok"),
        "coverage_ok": task.get("coverage_ok"),
        "initial_pose_ok": task.get("initial_pose_ok"),
        "stretch_ok": task.get("stretch_ok"),
        "foot_contact_ok": task.get("foot_contact_ok"),
        "reference_path": projection.get("path"),
        "reference_blend": projection.get("blend"),
        "reference_cartesian_pull_m": projection.get("cartesian_pull_m"),
        "maximum_prediction_application_delta": (
            float(np.max(np.abs(predicted[:compared] - applied[:compared])))
            if compared
            else None
        ),
        "checkpoint": metadata.get("checkpoint"),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "episode": str(episode),
        "video": str(video),
        "video_sha256": _sha256(video),
    }


def _comparison_video(seed: int, records: list[dict]) -> Path:
    captures = [cv2.VideoCapture(item["video"]) for item in records]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError(f"could not open all videos for seed {seed}")
    fps = min(capture.get(cv2.CAP_PROP_FPS) or 5.0 for capture in captures)
    width = 480
    height = 360
    output = REPORT / f"comparison_seed_{seed}.mp4"
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * len(captures), height),
    )
    try:
        while True:
            frames = []
            for capture, item in zip(captures, records):
                ok, frame = capture.read()
                if not ok:
                    return output
                frame = cv2.resize(frame, (width, height))
                cv2.putText(
                    frame,
                    item["variant"],
                    (18, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                frames.append(frame)
            writer.write(np.concatenate(frames, axis=1))
    finally:
        writer.release()
        for capture in captures:
            capture.release()


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    records = [
        _record(variant, seed) for variant in VARIANTS for seed in SEEDS
    ]
    for item in records:
        if (
            item["reference_path"] is not None
            or item["reference_blend"] != 0.0
            or item["reference_cartesian_pull_m"] != 0.0
        ):
            raise RuntimeError(f"rollout was not fully autonomous: {item}")
    summary = {}
    for variant in VARIANTS:
        selected = [item for item in records if item["variant"] == variant]
        successes = sum(item["success"] for item in selected)
        summary[variant] = {
            "successes": successes,
            "trials": len(selected),
            "success_rate": successes / len(selected),
            "mean_coverage_gain": float(
                np.mean([item["coverage_gain"] for item in selected])
            ),
            "mean_maximum_stretch": float(
                np.mean([item["maximum_stretch"] for item in selected])
            ),
            "continuous_grasp_passes": sum(
                bool(item["continuous_grasp_ok"]) for item in selected
            ),
            "stretch_passes": sum(bool(item["stretch_ok"]) for item in selected),
        }
    payload = {
        "protocol": {
            "steps": 50,
            "seeds": list(SEEDS),
            "fixed_initial_condition": True,
            "reference_action_blend": 0.0,
            "reference_cartesian_pull_m": 0.0,
        },
        "summary": summary,
        "runs": records,
    }
    (REPORT / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (REPORT / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ["# SAMDAMSARNN fully autonomous comparison", ""]
    for variant in VARIANTS:
        item = summary[variant]
        lines.append(
            f"- {variant}: {item['successes']}/{item['trials']} "
            f"({100.0 * item['success_rate']:.1f}%); mean coverage gain "
            f"{item['mean_coverage_gain']:.4f}; mean maximum stretch "
            f"{item['mean_maximum_stretch']:.4f}; continuous grasp "
            f"{item['continuous_grasp_passes']}/{item['trials']}; stretch QA "
            f"{item['stretch_passes']}/{item['trials']}"
        )
    lines.extend(
        [
            "",
            "All runs used fixed initial conditions and no reference-action or "
            "Cartesian projection. Three trials measure repeatability, not "
            "generalization across initial conditions.",
            "",
        ]
    )
    (REPORT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    for seed in SEEDS:
        _comparison_video(
            seed, [item for item in records if item["seed"] == seed]
        )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
