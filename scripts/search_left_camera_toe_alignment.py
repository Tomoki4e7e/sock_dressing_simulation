#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / (
    "autonomous_real_only_opening_reverse_270deg_positive_180_drape_"
    "recorded_pose_gripper_coupled_head_camera_frame_zero_physics_"
    "single_centroid_fast_policy_human_chair_away2cm_down7cm_"
    "left_camera.yaml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "phase4"
    / "left-camera-toe-center-alignment-away2cm-down8cm"
)


@dataclass(frozen=True)
class Candidate:
    right_cm: float
    camera_yaw_degrees: float = 0.0

    @property
    def name(self) -> str:
        right = f"{abs(self.right_cm):04.1f}".replace(".", "p")
        yaw = f"{abs(self.camera_yaw_degrees):04.1f}".replace(".", "p")
        right_sign = "pos" if self.right_cm >= 0 else "neg"
        yaw_sign = "pos" if self.camera_yaw_degrees >= 0 else "neg"
        return (
            f"right_{right_sign}_{right}cm_"
            f"yaw_{yaw_sign}_{yaw}deg"
        )


def parse_float_values(value: str) -> list[float]:
    values = [
        float(item.strip()) for item in value.split(",") if item.strip()
    ]
    if not values or not all(np.isfinite(values)):
        raise argparse.ArgumentTypeError(
            "values must be comma-separated finite numbers"
        )
    return values


def toe_tip_x_from_mask(
    mask: np.ndarray,
    tip_fraction: float = 0.08,
) -> float:
    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("leg mask must be a 2-D array")
    foreground = values > 0
    ys, xs = np.nonzero(foreground)
    if xs.size == 0:
        raise ValueError("leg mask has no foreground")
    minimum_y = int(ys.min())
    maximum_y = int(ys.max())
    span = maximum_y - minimum_y + 1
    band_height = max(4, int(round(span * float(tip_fraction))))
    tip_xs = xs[ys >= maximum_y - band_height + 1]
    if tip_xs.size == 0:
        raise ValueError("leg mask toe band has no foreground")
    return float(np.median(tip_xs))


def alignment_measurement(mask_path: Path) -> Dict[str, float]:
    mask = np.asarray(Image.open(mask_path).convert("L"))
    toe_x = toe_tip_x_from_mask(mask)
    center_x = (float(mask.shape[1]) - 1.0) * 0.5
    signed_error = toe_x - center_x
    return {
        "image_width_px": int(mask.shape[1]),
        "image_height_px": int(mask.shape[0]),
        "toe_tip_x_px": toe_x,
        "center_x_px": center_x,
        "signed_error_px": signed_error,
        "absolute_error_px": abs(signed_error),
    }


def candidate_values(
    coarse_values: Iterable[float],
    fine_center: Optional[float] = None,
    fine_step: float = 0.5,
    minimum: float = -3.0,
    maximum: float = 3.0,
) -> list[float]:
    if fine_center is None:
        values = coarse_values
    else:
        values = (
            fine_center - fine_step,
            fine_center,
            fine_center + fine_step,
        )
    return sorted(
        {
            round(float(item), 6)
            for item in values
            if minimum <= float(item) <= maximum
        }
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _candidate_config(
    candidate: Candidate,
    path: Path,
    base_config: Path,
) -> None:
    payload = {
        "extends": str(base_config.resolve()),
        "scene": {
            "camera_local_rotation": [
                60.0,
                float(candidate.camera_yaw_degrees),
                0.0,
            ],
            "initial_pose_contract": {
                "away_from_robot_m": 0.02,
                "down_m": 0.08,
                "right_from_robot_m": float(candidate.right_cm) / 100.0,
            },
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _latest_metadata(output_root: Path) -> Optional[Path]:
    matches = sorted(
        output_root.glob(
            "data_sock_sim_smoke/train/phase4_*/metadata.json"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    return matches[-1] if matches else None


def _completed_result(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        result.get("completed", False)
        and result.get("frames") == 1
        and result.get("stop_reason") == "max_steps"
        and result.get("mask")
    )


def _pose_screening(contract: Mapping[str, Any]) -> bool:
    return bool(
        contract.get("four_point_grasp_attached", False)
        and contract.get("rectangular_opening_ok", False)
        and contract.get("human_chair_lock_ok", False)
    )


def _run_candidate(
    candidate: Candidate,
    directory: Path,
    base_config: Path,
    seed: int,
    resume: bool,
) -> Dict[str, Any]:
    result_path = directory / "result.json"
    if resume and _completed_result(result_path):
        return json.loads(result_path.read_text(encoding="utf-8"))
    config_path = directory / "config.yaml"
    _candidate_config(candidate, config_path, base_config)
    output_root = directory / "episodes"
    log_path = directory / "run.log"
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
        "1",
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
    ]
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    metadata_path = _latest_metadata(output_root)
    metadata: Dict[str, Any] = {}
    if metadata_path is not None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    episode = metadata_path.parent if metadata_path is not None else None
    mask_path = (
        episode / "camera_right_mask" / "leg_mask" / "0.png"
        if episode is not None
        else None
    )
    measurement: Dict[str, Any] = {}
    if mask_path is not None and mask_path.is_file():
        measurement = alignment_measurement(mask_path)
    contract = dict(
        metadata.get("scenario_application", {}).get(
            "initial_pose_contract", {}
        )
        or {}
    )
    result = {
        "completed": True,
        "candidate": candidate.name,
        "right_cm": candidate.right_cm,
        "camera_yaw_degrees": candidate.camera_yaw_degrees,
        "returncode": int(completed.returncode),
        "config": str(config_path),
        "metadata": str(metadata_path) if metadata_path else None,
        "episode": str(episode) if episode else None,
        "mask": str(mask_path) if mask_path and mask_path.is_file() else None,
        "frames": int(metadata.get("frames", 0)),
        "stop_reason": metadata.get("stop_reason"),
        "pose_screening_ok": _pose_screening(contract),
        **measurement,
    }
    _write_json(result_path, result)
    return result


def ranking_key(result: Mapping[str, Any]) -> tuple:
    return (
        int(bool(result.get("pose_screening_ok", False))),
        int(
            result.get("frames") == 1
            and result.get("stop_reason") == "max_steps"
        ),
        -float(result.get("absolute_error_px", 1e9)),
        -abs(float(result.get("camera_yaw_degrees", 0.0))),
        -abs(float(result.get("right_cm", 0.0))),
    )


def _run_stage(
    stage: str,
    candidates: Sequence[Candidate],
    output_root: Path,
    base_config: Path,
    seed: int,
    resume: bool,
) -> list[Dict[str, Any]]:
    records = [
        _run_candidate(
            candidate,
            output_root / stage / candidate.name,
            base_config,
            seed,
            resume,
        )
        for candidate in candidates
    ]
    return sorted(records, key=ranking_key, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coarse-right-values",
        type=parse_float_values,
        default=[-3, -2, -1, 0, 1, 2, 3],
    )
    parser.add_argument(
        "--yaw-values",
        type=parse_float_values,
        default=[-3, -2, -1, 0, 1, 2, 3],
    )
    parser.add_argument("--fine-step-cm", type=float, default=0.5)
    parser.add_argument("--yaw-threshold-px", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.fine_step_cm <= 0 or args.yaw_threshold_px < 0:
        parser.error("fine step must be positive and threshold non-negative")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_config = args.config.expanduser().resolve()
    resume = not args.no_resume

    coarse = _run_stage(
        "coarse",
        [Candidate(value) for value in args.coarse_right_values],
        output_root,
        base_config,
        args.seed,
        resume,
    )
    if not coarse:
        return 2
    coarse_best = coarse[0]
    fine = _run_stage(
        "fine",
        [
            Candidate(value)
            for value in candidate_values(
                (),
                float(coarse_best["right_cm"]),
                args.fine_step_cm,
            )
        ],
        output_root,
        base_config,
        args.seed,
        resume,
    )
    best = fine[0] if fine else coarse_best
    yaw: list[Dict[str, Any]] = []
    if float(best.get("absolute_error_px", 1e9)) > args.yaw_threshold_px:
        yaw = _run_stage(
            "yaw",
            [
                Candidate(
                    float(best["right_cm"]),
                    float(value),
                )
                for value in args.yaw_values
            ],
            output_root,
            base_config,
            args.seed,
            resume,
        )
        if yaw and ranking_key(yaw[0]) > ranking_key(best):
            best = yaw[0]

    summary = {
        "base_config": str(base_config),
        "coarse": coarse,
        "fine": fine,
        "yaw": yaw,
        "best": best,
    }
    _write_json(output_root / "alignment_search.json", summary)
    best_config = Path(str(best["config"]))
    shutil.copy2(best_config, output_root / "best_config.yaml")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if best.get("pose_screening_ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
