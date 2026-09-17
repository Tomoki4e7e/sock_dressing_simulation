from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def _vector(name: str, value: Sequence[float], size: int) -> Tuple[float, ...]:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {size} finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class SockMesh:
    length_m: float
    radius_m: float
    radial_segments: int
    length_segments: int

    def validate(self) -> None:
        if self.length_m <= 0 or self.radius_m <= 0:
            raise ValueError("sock mesh dimensions must be positive")
        if self.radial_segments < 3 or self.length_segments < 1:
            raise ValueError("sock mesh segment counts are invalid")


@dataclass(frozen=True)
class Scenario:
    seed: int
    sock_mesh: SockMesh
    sock_position: Tuple[float, float, float]
    sock_rotation: Tuple[float, float, float]
    foot_ik_index: Optional[int]
    foot_position: Tuple[float, float, float]
    foot_rotation: Tuple[float, float, float]
    initial_joints: Tuple[float, ...]
    settle_steps: int
    initial_coverage_target: Optional[float]
    initial_coverage_tolerance: float

    def to_metadata(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sock_mesh"]["units"] = "metres"
        payload["sock_position_units"] = "metres"
        payload["sock_rotation_units"] = "degrees"
        payload["foot_position_units"] = "metres"
        payload["foot_rotation_units"] = "degrees"
        payload["initial_joint_units"] = "arm radians, gripper metres"
        return payload


def _uniform_scalar(
    rng: np.random.Generator,
    nominal: float,
    bounds: Optional[Sequence[float]],
    name: str,
) -> float:
    if bounds is None:
        return float(nominal)
    low, high = _vector(name, bounds, 2)
    if low > high:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return float(rng.uniform(low, high))


def _uniform_vector(
    rng: np.random.Generator,
    nominal: Sequence[float],
    bounds: Optional[Sequence[Sequence[float]]],
    name: str,
) -> Tuple[float, ...]:
    nominal_array = np.asarray(nominal, dtype=float)
    if bounds is None:
        return _vector(name, nominal_array, nominal_array.size)
    bound_array = np.asarray(bounds, dtype=float)
    if bound_array.shape != (nominal_array.size, 2) or not np.all(
        np.isfinite(bound_array)
    ):
        raise ValueError(f"{name} bounds must have shape ({nominal_array.size}, 2)")
    if np.any(bound_array[:, 0] > bound_array[:, 1]):
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return tuple(
        float(item)
        for item in rng.uniform(bound_array[:, 0], bound_array[:, 1])
    )


def scenario_from_config(config: Mapping[str, Any], seed: Optional[int] = None) -> Scenario:
    from .joints import JointMap

    settings = config.get("scenario", {})
    sock = settings.get("sock", {})
    foot = settings.get("foot", {})
    randomization = settings.get("randomization", {})
    randomization_enabled = bool(randomization.get("enabled", False))
    ranges = dict(randomization.get("ranges", {})) if randomization_enabled else {}
    calibration_path = randomization.get("calibration_file")
    if (
        randomization_enabled
        and randomization.get("use_calibrated_joint_ranges", True)
        and calibration_path
    ):
        from .config import resolve_package_path

        resolved = resolve_package_path(calibration_path)
        if resolved.is_file():
            calibration = json.loads(resolved.read_text(encoding="utf-8"))
            joint_proxy = calibration.get("proxies", {}).get("joint_state_rad_or_m", {})
            low, high = joint_proxy.get("low"), joint_proxy.get("high")
            if low is not None and high is not None:
                ranges["initial_joints"] = [
                    [lower, upper] for lower, upper in zip(low, high)
                ]
    scenario_seed = int(settings.get("seed", 0) if seed is None else seed)
    rng = np.random.default_rng(scenario_seed)

    mesh = SockMesh(
        length_m=_uniform_scalar(
            rng, float(sock["length_m"]), ranges.get("sock_length_m"), "sock_length_m"
        ),
        radius_m=_uniform_scalar(
            rng, float(sock["radius_m"]), ranges.get("sock_radius_m"), "sock_radius_m"
        ),
        radial_segments=int(sock.get("radial_segments", 32)),
        length_segments=int(sock.get("length_segments", 24)),
    )
    mesh.validate()

    mapping = JointMap.from_config(config)
    initial_joints = np.asarray(settings.get("initial_joints", [0.0] * 18), dtype=float)
    if initial_joints.shape != (18,) or not np.all(np.isfinite(initial_joints)):
        raise ValueError("scenario.initial_joints must contain 18 finite values")
    randomized_joints = _uniform_vector(
        rng, initial_joints, ranges.get("initial_joints"), "initial_joints"
    )
    initial = np.asarray(randomized_joints, dtype=float)
    if ranges.get("initial_joints") is not None:
        for name, rule in mapping.mimic.items():
            target_index = mapping.index()[name]
            source_index = mapping.index()[rule["source"]]
            initial[target_index] = (
                initial[source_index] * float(rule["multiplier"])
                + float(rule["offset"])
            )
    if np.any(initial < mapping.lower) or np.any(initial > mapping.upper):
        raise ValueError("scenario.initial_joints violate configured joint limits")
    for name, rule in mapping.mimic.items():
        target = mapping.index()[name]
        source = mapping.index()[rule["source"]]
        expected = initial[source] * float(rule["multiplier"]) + float(rule["offset"])
        if not np.isclose(initial[target], expected):
            raise ValueError(f"scenario.initial_joints violates mimic rule for {name}")

    target = settings.get("initial_coverage_target")
    if target is not None and not 0.0 <= float(target) <= 1.0:
        raise ValueError("initial_coverage_target must be in [0, 1]")
    tolerance = float(settings.get("initial_coverage_tolerance", 0.15))
    if tolerance < 0:
        raise ValueError("initial_coverage_tolerance must be non-negative")
    foot_index = foot.get("ik_index")
    if foot_index is not None and int(foot_index) not in (2, 3):
        raise ValueError("foot.ik_index must be 2 (left) or 3 (right)")
    settle_steps = int(settings.get("settle_steps", 5))
    if settle_steps < 0:
        raise ValueError("settle_steps must be non-negative")

    return Scenario(
        seed=scenario_seed,
        sock_mesh=mesh,
        sock_position=_uniform_vector(
            rng, sock["position"], ranges.get("sock_position"), "sock_position"
        ),
        sock_rotation=_uniform_vector(
            rng, sock.get("rotation", [0, 0, 0]), ranges.get("sock_rotation"), "sock_rotation"
        ),
        foot_ik_index=None if foot_index is None else int(foot_index),
        foot_position=_uniform_vector(
            rng, foot.get("position", [0, 0, 0]), ranges.get("foot_position"), "foot_position"
        ),
        foot_rotation=_uniform_vector(
            rng, foot.get("rotation", [0, 0, 0]), ranges.get("foot_rotation"), "foot_rotation"
        ),
        initial_joints=tuple(float(item) for item in initial),
        settle_steps=settle_steps,
        initial_coverage_target=None if target is None else float(target),
        initial_coverage_tolerance=tolerance,
    )
