#!/usr/bin/env python3
"""Summarize sock foot-conformity search episodes into one JSON table."""

import argparse
import json
from pathlib import Path

import numpy as np


def _values(frames, key):
    values = [
        frame.get("foot_conformity", {}).get(key)
        for frame in frames
        if frame.get("foot_conformity", {}).get("available")
    ]
    return [float(value) for value in values if value is not None]


def _edge_class_max(cloth_quality, name, field):
    values = [
        item.get("stretch", {})
        .get("edge_classes", {})
        .get(name, {})
        .get(field)
        for item in cloth_quality
    ]
    values = [float(value) for value in values if value is not None]
    return max(values) if values else None


def summarize(episode: Path) -> dict:
    metadata = json.loads((episode / "metadata.json").read_text())
    success = metadata.get("task_success", {})
    frames = metadata.get("dressing_quality_by_frame", [])
    cloth_quality = metadata.get("cloth_quality_by_frame", [])
    contact = _values(frames, "contact_fraction")
    stretch_std = _values(frames, "near_foot_stretch_std")
    stretch_max = _values(frames, "near_foot_stretch_max")
    return {
        "episode": str(episode),
        "frames": len(frames),
        "task_success": bool(success.get("success", False)),
        "failed_gates": success.get("failed_gates", []),
        "maximum_geometric_cloth_foot_penetration_m": success.get(
            "maximum_geometric_cloth_foot_penetration_m"
        ),
        "maximum_gripper_foot_region_penetration_m": success.get(
            "maximum_gripper_foot_region_penetration_m"
        ),
        "coverage_gain": success.get("coverage_gain"),
        "cuff_progress_toward_ankle_m": success.get(
            "cuff_progress_toward_ankle_m"
        ),
        "final_surface_containment_ratio": success.get(
            "final_surface_containment_ratio"
        ),
        "minimum_opening_ring_area_retention": success.get(
            "minimum_opening_ring_area_retention"
        ),
        "continuous_stretch_ok_frames": sum(
            bool(item.get("stretch", {}).get("passes", False))
            for item in cloth_quality
        ),
        "maximum_body_body_stretch": _edge_class_max(
            cloth_quality, "body_body", "maximum_stretch"
        ),
        "maximum_opening_body_excess_length_m": _edge_class_max(
            cloth_quality, "opening_body", "maximum_excess_length_m"
        ),
        "mean_contact_fraction": (
            float(np.mean(contact)) if contact else None
        ),
        "peak_contact_fraction": max(contact) if contact else None,
        "mean_near_foot_stretch_std": (
            float(np.mean(stretch_std)) if stretch_std else None
        ),
        "maximum_near_foot_stretch": (
            max(stretch_max) if stretch_max else None
        ),
        "maximum_opening_rim_stretch": _edge_class_max(
            cloth_quality, "opening_rim", "maximum_stretch"
        ),
        "opening_rim_elastic_qa": success.get("opening_rim_elastic_qa"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = {}
    for metadata in sorted(args.root.glob("*/**/metadata.json")):
        episode = metadata.parent
        candidate = episode.relative_to(args.root).parts[0]
        rows[candidate] = summarize(episode)
    text = json.dumps(rows, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
