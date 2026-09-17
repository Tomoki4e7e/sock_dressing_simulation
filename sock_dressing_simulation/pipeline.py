from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def run_phase1_pipeline(
    episode: Path,
    *,
    data_root: Path,
    manifest: Path,
    dataset_name: str,
    audit_output: Path,
    n_opening: int = 8,
) -> Dict[str, Any]:
    episode = Path(episode).resolve()
    project = Path(__file__).resolve().parents[2]
    processing = project / "dress_regrasping" / "data-processing"
    residual_root = project / "dress_regrasping"
    sys.path[:0] = [str(processing), str(residual_root)]
    try:
        from feature_package import compute_episode_features, save_episode_features
        from residual_flow.data import audit_dataset, load_manifest, save_audit
        from shareset_io import EpisodeReader
        from visualize_features import visualize_episode

        bundle = compute_episode_features(
            EpisodeReader(episode),
            n_opening=n_opening,
            baseline_mode="episode_max",
        )
        save_episode_features(bundle, episode / "features", episode_dir=episode)
        visualization = visualize_episode(episode)
        reports = audit_dataset(load_manifest(data_root, manifest, dataset_name))
        save_audit(reports, audit_output)
    finally:
        del sys.path[:2]

    metadata_path = episode / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    measured_coverage = (
        float(bundle.coverage.coverage[0])
        if bundle.coverage.coverage.size and np.isfinite(bundle.coverage.coverage[0])
        else None
    )
    target = metadata.get("scenario", {}).get("initial_coverage_target")
    tolerance = metadata.get("scenario", {}).get("initial_coverage_tolerance", 0.0)
    coverage_target_ok: Optional[bool] = None
    if target is not None and measured_coverage is not None:
        coverage_target_ok = abs(measured_coverage - float(target)) <= float(tolerance)
    elif target is not None:
        coverage_target_ok = False
    metadata["phase1_features"] = {
        "n_frames": int(bundle.features_ok.size),
        "n_features_ok": int(bundle.features_ok.sum()),
        "initial_coverage_measured": measured_coverage,
        "initial_coverage_target_ok": coverage_target_ok,
        "missing_values_policy": "NaN/False; no imputation",
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    matching = [
        report for report in reports if report.directory == episode
    ] if reports and hasattr(reports[0], "directory") else []
    if not matching:
        matching = [report for report in reports if report.episode == episode.name]
    audit_ok = bool(matching) and all(
        not report.errors and report.features_present and report.usable_for_residual
        for report in matching
    )
    observation_quality = metadata.get("observation_quality", {})
    live_quality_ok = (
        bool(observation_quality.get("learning_ready"))
        if metadata.get("phase") == 1
        else True
    )
    ok = (
        visualization.ok
        and audit_ok
        and int(bundle.features_ok.sum()) > 0
        and live_quality_ok
        and coverage_target_ok is not False
    )
    result = {
        "ok": ok,
        "episode": str(episode),
        "features_dir": str(episode / "features"),
        "features_ok": int(bundle.features_ok.sum()),
        "frames": int(bundle.features_ok.size),
        "visualization_ok": bool(visualization.ok),
        "audit_ok": audit_ok,
        "audit_output": str(Path(audit_output).resolve()),
        "live_observation_quality_ok": live_quality_ok,
        "initial_coverage_measured": measured_coverage,
        "initial_coverage_target_ok": coverage_target_ok,
    }
    gate_path = episode / "features" / "viz" / "phase1_gate.json"
    gate_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata["phase1_gate"] = result
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
