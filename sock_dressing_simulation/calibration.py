from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def discover_episodes(data_root: Path) -> List[Path]:
    return sorted(
        path.parent
        for path in Path(data_root).rglob("angle.csv")
        if (path.parent / "camera_right_mask/sock_mask").is_dir()
        and (path.parent / "camera_right_mask/leg_mask").is_dir()
    )


def _quantiles(values: np.ndarray, low: float, high: float) -> Dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    finite_rows = values[np.all(np.isfinite(values), axis=1)]
    if not finite_rows.size:
        return {"low": None, "high": None, "n": 0}
    return {
        "low": np.quantile(finite_rows, low, axis=0).tolist(),
        "high": np.quantile(finite_rows, high, axis=0).tolist(),
        "n": int(finite_rows.shape[0]),
    }


def calibrate_randomization(
    data_root: Path,
    *,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    max_episodes: int = 0,
) -> Dict[str, Any]:
    if not 0 <= quantile_low < quantile_high <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    episodes = discover_episodes(data_root)
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
    if not episodes:
        raise FileNotFoundError(f"no ShareSet-compatible episodes below {data_root}")

    processing = Path(__file__).resolve().parents[2] / "dress_regrasping" / "data-processing"
    sys.path.insert(0, str(processing))
    try:
        from feature_package import compute_episode_features
        from shareset_io import EpisodeReader

        angles: List[np.ndarray] = []
        coverage: List[np.ndarray] = []
        foot_axis_deg: List[np.ndarray] = []
        opening_norm: List[np.ndarray] = []
        angle_csv_widths: Dict[int, int] = {}
        feature_frames = 0
        for episode in episodes:
            angle = np.loadtxt(str(episode / "angle.csv"), delimiter=",", ndmin=2)
            angle_csv_widths[angle.shape[1]] = angle_csv_widths.get(angle.shape[1], 0) + 1
            if angle.shape[1] == 18:
                angles.append(angle)
            bundle = compute_episode_features(
                EpisodeReader(episode), baseline_mode="episode_max", n_opening=8
            )
            ok = bundle.features_ok
            feature_frames += int(ok.sum())
            coverage.append(bundle.coverage.coverage[bundle.coverage.ok])
            vectors = bundle.keypoints.heel_xy[ok] - bundle.keypoints.toe_xy[ok]
            if vectors.size:
                foot_axis_deg.append(
                    np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
                )
                opening_norm.append(bundle.keypoints.opening_norm[ok].reshape(-1, 2))
    finally:
        sys.path.remove(str(processing))

    all_angles = np.concatenate(angles, axis=0) if angles else np.empty((0, 18))
    all_coverage = np.concatenate(coverage) if coverage else np.empty(0)
    all_axis = np.concatenate(foot_axis_deg) if foot_axis_deg else np.empty(0)
    all_opening = (
        np.concatenate(opening_norm, axis=0) if opening_norm else np.empty((0, 2))
    )
    return {
        "schema": "sock-sim-randomization-calibration-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "data_root": str(Path(data_root).resolve()),
            "episodes": [str(path.relative_to(data_root)) for path in episodes],
            "read_only": True,
            "angle_csv_widths": {
                str(width): count for width, count in sorted(angle_csv_widths.items())
            },
        },
        "quantiles": [quantile_low, quantile_high],
        "proxies": {
            "joint_state_rad_or_m": _quantiles(
                all_angles, quantile_low, quantile_high
            ),
            "gripper_opening_m": _quantiles(
                all_angles[:, [7, 16]], quantile_low, quantile_high
            ),
            "initial_coverage_image_proxy": _quantiles(
                all_coverage, quantile_low, quantile_high
            ),
            "foot_axis_image_degrees": _quantiles(
                all_axis, quantile_low, quantile_high
            ),
            "opening_normalized": _quantiles(
                all_opening, quantile_low, quantile_high
            ),
        },
        "feature_frames_ok": feature_frames,
        "unmeasured": {
            "sock_length_m": "No metric garment measurement is stored in ShareSet.",
            "sock_radius_m": "No metric garment measurement is stored in ShareSet.",
            "cartesian_grasp_position_m": "ShareSet stores joints, not calibrated end-effector poses.",
            "physical_foot_euler_degrees": "Image foot-axis angle is only a proxy.",
            "joint_state_18d": (
                None
                if angles
                else "Sample angle.csv files are not the current 18-D contract; no implicit column mapping was guessed."
            ),
        },
    }


def save_calibration(report: Dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
