from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


def assess_observation_quality(
    cameras: Sequence[Mapping[str, np.ndarray]],
    *,
    expected_width: int,
    expected_height: int,
    exact_foot_colliders: bool,
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
    failed = [name for name in required if not checks[name]]
    return {
        "checks": checks,
        "failed_checks": failed,
        "learning_ready": not failed,
        "policy": "fail-closed; failed observations must not be used as Phase 1 live data",
    }
