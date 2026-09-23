from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image


def assess_observation_quality(
    cameras: Sequence[Mapping[str, np.ndarray]],
    *,
    expected_width: int,
    expected_height: int,
    exact_foot_colliders: bool,
    player_diagnostics: Mapping[str, Any] = None,
) -> Dict[str, Any]:
    if not cameras:
        raise ValueError("at least one camera observation is required")
    expected_shape = (int(expected_height), int(expected_width))
    checks: Dict[str, bool] = {
        "resolution_contract": all(
            np.asarray(camera["camera_depth"]).shape == expected_shape for camera in cameras
        ),
        "rgb_nonconstant": any(np.unique(camera["rgb"]).size > 1 for camera in cameras),
        "depth_nonconstant": any(
            np.unique(camera["camera_depth"]).size > 1 for camera in cameras
        ),
        "sock_mask_nontrivial": any(
            0 < np.count_nonzero(camera["sock_mask"]) < np.asarray(camera["sock_mask"]).size
            for camera in cameras
        ),
        "leg_mask_nontrivial": any(
            0 < np.count_nonzero(camera["leg_mask"]) < np.asarray(camera["leg_mask"]).size
            for camera in cameras
        ),
        "masks_distinct": any(
            not np.array_equal(camera["sock_mask"], camera["leg_mask"])
            for camera in cameras
        ),
        "temporal_variation": len(cameras) > 1
        and any(
            not np.array_equal(cameras[0]["rgb"], camera["rgb"])
            or not np.array_equal(cameras[0]["sock_mask"], camera["sock_mask"])
            for camera in cameras[1:]
        ),
        "exact_foot_colliders": bool(exact_foot_colliders),
    }
    diagnostics = dict(player_diagnostics or {})
    if diagnostics:
        obi = diagnostics.get("obi_contract", {})
        checks.update(
            {
                "player_contract_verified": bool(obi.get("ok", False)),
                "robot_obi_collider_verified": bool(
                    diagnostics.get("robot_obi_collider_verified", False)
                ),
                "registered_foot_colliders": _foot_colliders_verified(diagnostics),
            }
        )
    required = (
        "resolution_contract",
        "rgb_nonconstant",
        "depth_nonconstant",
        "sock_mask_nontrivial",
        "leg_mask_nontrivial",
        "masks_distinct",
        "temporal_variation",
        "exact_foot_colliders",
    )
    if diagnostics:
        required += (
            "player_contract_verified",
            "robot_obi_collider_verified",
            "registered_foot_colliders",
        )
    failed = [name for name in required if not checks[name]]
    return {
        "checks": checks,
        "failed_checks": failed,
        "learning_ready": not failed,
        "policy": "fail-closed; failed observations must not be used as Phase 1 live data",
    }


def _foot_colliders_verified(diagnostics: Mapping[str, Any]) -> bool:
    required = {"calf", "ankle", "heel", "forefoot", "toes"}
    enabled = {
        str(item.get("region"))
        for item in diagnostics.get("registered_obi_colliders", ())
        if item.get("enabled", False)
    }
    return required.issubset(enabled)


def _numbered_pngs(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    paths = []
    for path in directory.iterdir():
        if path.suffix.lower() != ".png":
            continue
        try:
            index = int(path.stem)
        except ValueError:
            continue
        paths.append((index, path))
    return [path for _, path in sorted(paths)]


def _sample_paths(paths: Sequence[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return list(paths)
    indices = np.rint(np.linspace(0, len(paths) - 1, maximum)).astype(int)
    return [paths[index] for index in indices]


def _image_summary(paths: Sequence[Path], *, mask: bool, maximum: int) -> Dict[str, Any]:
    samples = _sample_paths(paths, maximum)
    if not samples:
        return {"count": len(paths), "samples": 0}
    means = []
    standard_deviations = []
    foreground = []
    centroids = []
    shapes = []
    for path in samples:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L" if mask else "RGB"), dtype=np.float32)
        shapes.append(list(array.shape[:2]))
        means.append(float(array.mean() / 255.0))
        standard_deviations.append(float(array.std() / 255.0))
        if mask:
            binary = array > 0
            foreground.append(float(binary.mean()))
            points = np.argwhere(binary)
            if points.size:
                y, x = points.mean(axis=0)
                centroids.append(
                    [float(x / max(array.shape[1] - 1, 1)), float(y / max(array.shape[0] - 1, 1))]
                )
    result: Dict[str, Any] = {
        "count": len(paths),
        "samples": len(samples),
        "shapes": sorted({tuple(shape) for shape in shapes}),
        "mean": float(np.mean(means)),
        "standard_deviation": float(np.mean(standard_deviations)),
    }
    result["shapes"] = [list(shape) for shape in result["shapes"]]
    if mask:
        result["foreground_fraction"] = float(np.mean(foreground))
        result["centroid"] = (
            np.mean(np.asarray(centroids), axis=0).tolist() if centroids else None
        )
    return result


def summarize_learning_episode(
    episode: Path, *, maximum_image_samples: int = 16
) -> Dict[str, Any]:
    """Summarize one ShareSet-style episode without loading it into training."""
    episode = Path(episode).resolve()
    sock_mask = _numbered_pngs(episode / "camera_right_mask" / "sock_mask")
    limb_dir = episode / "camera_right_mask" / "foot_mask"
    limb_name = "foot"
    if not limb_dir.is_dir():
        limb_dir = episode / "camera_right_mask" / "leg_mask"
        limb_name = "leg"
    limb_mask = _numbered_pngs(limb_dir)
    sock_depth = _numbered_pngs(episode / "depth_mask" / "sock_depth")
    limb_depth_dir = episode / "depth_mask" / f"{limb_name}_depth"
    limb_depth = _numbered_pngs(limb_depth_dir)
    modalities = {
        "rgb": _image_summary(
            _numbered_pngs(episode / "camera_right"),
            mask=False,
            maximum=maximum_image_samples,
        ),
        "camera_depth": _image_summary(
            _numbered_pngs(episode / "camera_depth"),
            mask=False,
            maximum=maximum_image_samples,
        ),
        "sock_mask": _image_summary(
            sock_mask, mask=True, maximum=maximum_image_samples
        ),
        "limb_mask": _image_summary(
            limb_mask, mask=True, maximum=maximum_image_samples
        ),
        "sock_depth": _image_summary(
            sock_depth, mask=True, maximum=maximum_image_samples
        ),
        "limb_depth": _image_summary(
            limb_depth, mask=True, maximum=maximum_image_samples
        ),
    }
    signals = {}
    for filename in ("angle.csv", "torque.csv", "external_torque.csv"):
        path = episode / filename
        if not path.is_file():
            signals[filename] = {"missing": True}
            continue
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        signals[filename] = {
            "shape": list(values.shape),
            "minimum": float(values.min()) if values.size else None,
            "maximum": float(values.max()) if values.size else None,
            "standard_deviation": float(values.std()) if values.size else None,
            "all_zero": bool(values.size and not np.any(values)),
        }
    metadata = {}
    metadata_path = episode / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "episode": str(episode),
        "limb_alias": limb_name,
        "modalities": modalities,
        "signals": signals,
        "camera_mount": metadata.get("diagnostics", {}).get("camera_mount"),
    }


def compare_learning_domains(
    real_episodes: Sequence[Path],
    simulation_episodes: Sequence[Path],
    *,
    maximum_image_samples: int = 16,
) -> Dict[str, Any]:
    """Compare gross ShareSet modality properties and fail on unusable sim data."""
    if not real_episodes or not simulation_episodes:
        raise ValueError("real and simulation episode lists must both be non-empty")
    real = [
        summarize_learning_episode(path, maximum_image_samples=maximum_image_samples)
        for path in real_episodes
    ]
    simulation = [
        summarize_learning_episode(path, maximum_image_samples=maximum_image_samples)
        for path in simulation_episodes
    ]
    errors = []
    warnings = []
    real_shapes = {
        tuple(shape)
        for item in real
        for shape in item["modalities"]["rgb"].get("shapes", [])
    }
    for item in simulation:
        modalities = item["modalities"]
        sim_shapes = {tuple(shape) for shape in modalities["rgb"].get("shapes", [])}
        missing = [
            name for name in ("rgb", "camera_depth", "sock_depth", "limb_depth")
            if not modalities[name].get("count")
        ]
        if missing:
            errors.append(f"{item['episode']}: missing modalities {missing}")
        if real_shapes and sim_shapes != real_shapes:
            errors.append(
                f"{item['episode']}: RGB shapes {sorted(sim_shapes)} differ from real {sorted(real_shapes)}"
            )
        torque = item["signals"].get("torque.csv", {})
        if torque.get("missing") or torque.get("all_zero"):
            errors.append(f"{item['episode']}: torque.csv is missing or all zero")
        mount = item.get("camera_mount")
        if not isinstance(mount, Mapping) or mount.get("mode") != "robot_link":
            errors.append(f"{item['episode']}: inference camera is not robot-link mounted")
        counts = {
            name: modalities[name].get("count", 0)
            for name in ("rgb", "camera_depth", "sock_depth", "limb_depth")
        }
        if max(counts.values(), default=0) != min(counts.values(), default=0):
            warnings.append(f"{item['episode']}: modality frame counts differ: {counts}")
    real_frame_median = float(
        np.median([item["modalities"]["rgb"].get("count", 0) for item in real])
    )
    for item in simulation:
        count = item["modalities"]["rgb"].get("count", 0)
        if real_frame_median and count < real_frame_median / 3:
            warnings.append(
                f"{item['episode']}: {count} RGB frames versus real median {real_frame_median:g}"
            )
    differences = {}
    for name in ("rgb", "camera_depth", "sock_mask", "limb_mask"):
        for metric in ("mean", "standard_deviation", "foreground_fraction"):
            real_values = [
                item["modalities"][name][metric]
                for item in real
                if metric in item["modalities"][name]
            ]
            sim_values = [
                item["modalities"][name][metric]
                for item in simulation
                if metric in item["modalities"][name]
            ]
            if real_values and sim_values:
                key = f"{name}.{metric}"
                real_mean = float(np.mean(real_values))
                sim_mean = float(np.mean(sim_values))
                differences[key] = {
                    "real": real_mean,
                    "simulation": sim_mean,
                    "absolute": abs(sim_mean - real_mean),
                }
                if name == "rgb" and metric == "mean" and abs(sim_mean - real_mean) > 0.2:
                    warnings.append(
                        f"{key}: normalized mean differs by {abs(sim_mean - real_mean):.3g}"
                    )
                if metric == "foreground_fraction":
                    denominator = max(real_mean, 1e-6)
                    ratio = sim_mean / denominator
                    differences[key]["ratio"] = ratio
                    if ratio > 5.0 or ratio < 0.2:
                        warnings.append(f"{key}: simulation/real ratio is {ratio:.3g}")
    for name in ("sock_mask", "limb_mask"):
        real_centroids = [
            item["modalities"][name]["centroid"]
            for item in real
            if item["modalities"][name].get("centroid") is not None
        ]
        sim_centroids = [
            item["modalities"][name]["centroid"]
            for item in simulation
            if item["modalities"][name].get("centroid") is not None
        ]
        if real_centroids and sim_centroids:
            real_centroid = np.mean(np.asarray(real_centroids), axis=0)
            sim_centroid = np.mean(np.asarray(sim_centroids), axis=0)
            distance = float(np.linalg.norm(sim_centroid - real_centroid))
            differences[f"{name}.centroid"] = {
                "real": real_centroid.tolist(),
                "simulation": sim_centroid.tolist(),
                "normalized_distance": distance,
            }
            if distance > 0.35:
                warnings.append(f"{name}.centroid: normalized distance is {distance:.3g}")
    return {
        "ok": not errors,
        "status": "fail" if errors else ("warn" if warnings else "pass"),
        "errors": errors,
        "warnings": warnings,
        "differences": differences,
        "real": real,
        "simulation": simulation,
    }
