from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

PROTOCOL_VERSION = "sock-cloth-v1"
SIDES = ("left", "right")
REQUIRED_COLLIDER_REGIONS = (
    "calf",
    "ankle",
    "heel",
    "forefoot",
    "toes",
    "left_gripper",
    "right_gripper",
    "robot",
    "chair",
)


def _finite_vectors(value: Any, width: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != width:
        raise ValueError(f"{name} must have shape (n, {width})")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class GraspState:
    side: str
    attached: bool
    particle_indices: Tuple[int, ...]
    constraint_error: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GraspState":
        side = str(value["side"]).lower()
        if side not in SIDES:
            raise ValueError(f"invalid grasp side: {side}")
        indices = tuple(int(index) for index in value.get("particle_indices", ()))
        if any(index < 0 for index in indices):
            raise ValueError("particle indices must be non-negative")
        error = float(value.get("constraint_error", 0.0))
        if not np.isfinite(error) or error < 0:
            raise ValueError("constraint_error must be finite and non-negative")
        attached = bool(value.get("attached", False))
        if attached != bool(indices):
            raise ValueError("attached must agree with particle_indices")
        return cls(side, attached, indices, error)


@dataclass(frozen=True)
class ClothContact:
    particle_index: int
    collider_id: int
    position: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    penetration_m: float
    force_proxy: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ClothContact":
        position = tuple(float(item) for item in value["position"])
        normal = tuple(float(item) for item in value["normal"])
        if len(position) != 3 or len(normal) != 3:
            raise ValueError("contact position and normal must contain three values")
        scalars = np.asarray(
            [*position, *normal, value["penetration_m"], value["force_proxy"]],
            dtype=float,
        )
        if not np.all(np.isfinite(scalars)):
            raise ValueError("contact values must be finite")
        return cls(
            particle_index=int(value["particle_index"]),
            collider_id=int(value["collider_id"]),
            position=position,
            normal=normal,
            penetration_m=float(value["penetration_m"]),
            force_proxy=float(value["force_proxy"]),
        )


def validate_configuration(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> Dict[str, Any]:
    """Compare player-reported settings with the fail-closed Python contract."""
    required = (
        "protocol_version",
        "obi_version",
        "timestep_s",
        "substeps",
        "solver_iterations",
        "particle_radius_m",
        "stretch_compliance",
        "bend_compliance",
        "self_collision",
    )
    missing = [name for name in required if name not in actual]
    mismatches = {}
    for name, wanted in expected.items():
        if name not in actual:
            mismatches[name] = {"expected": wanted, "actual": None}
            continue
        got = actual[name]
        if isinstance(wanted, float):
            matches = bool(np.isclose(float(got), wanted, rtol=1e-6, atol=1e-9))
        else:
            matches = got == wanted
        if not matches:
            mismatches[name] = {"expected": wanted, "actual": got}
    return {
        "ok": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
        "actual": dict(actual),
    }


class SockClothAttr:
    """RFUniverse-style custom attribute without an eager pyrcareworld import."""

    def __init__(self, env: Any, object_id: int, data: Mapping[str, Any] = None):
        self.env = env
        self.id = int(object_id)
        self.data: Dict[str, Any] = dict(data or {})

    def parse_message(self, data: Mapping[str, Any]) -> None:
        self.data = dict(data)

    def _send_data(self, command: str, *args: Any) -> None:
        self.env._send_instance_data(self.id, command, *args)

    def request_particles(self) -> None:
        self._send_data("GetParticles")

    def particles(self) -> np.ndarray:
        return _finite_vectors(self.data.get("particles", []), 3, "particles")

    def request_particle_velocities(self) -> None:
        self._send_data("GetParticleVelocities")

    def particle_velocities(self) -> np.ndarray:
        return _finite_vectors(
            self.data.get("particle_velocities", []), 3, "particle_velocities"
        )

    def request_configuration(self) -> None:
        self._send_data("GetClothConfiguration")

    def configure(
        self,
        *,
        stretch_compliance: float,
        bend_compliance: float,
        particle_radius_m: float,
        particle_mass_kg: float,
        collision_margin_m: float,
        friction: float,
        self_collision: bool,
        substeps: int,
        solver_iterations: int,
    ) -> None:
        self._send_data(
            "ConfigureSock",
            float(stretch_compliance),
            float(bend_compliance),
            float(particle_radius_m),
            float(particle_mass_kg),
            float(collision_margin_m),
            float(friction),
            bool(self_collision),
            int(substeps),
            int(solver_iterations),
        )

    def set_grasp_targets(self, left_id: int, right_id: int) -> None:
        self._send_data("SetGraspTargets", int(left_id), int(right_id))

    def configure_right_leg_colliders(self, human_id: int) -> None:
        self._send_data("ConfigureRightLegColliders", int(human_id))

    def configure_mask_proxy_cameras(self, proxy_layer: int = 31) -> None:
        self._send_data("ConfigureMaskProxyCameras", int(proxy_layer))

    def configuration(self) -> Mapping[str, Any]:
        return dict(self.data.get("cloth_configuration", {}))

    def request_registered_colliders(self) -> None:
        self._send_data("GetRegisteredObiColliders")

    def registered_colliders(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self.data.get("registered_obi_colliders", ()))

    def grasp(self, side: str, max_distance_m: float) -> None:
        side = str(side).lower()
        if side not in SIDES:
            raise ValueError(f"invalid grasp side: {side}")
        distance = float(max_distance_m)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("max_distance_m must be finite and positive")
        self._send_data("GraspLeft" if side == "left" else "GraspRight", distance)

    def release(self, side: str) -> None:
        side = str(side).lower()
        if side not in SIDES:
            raise ValueError(f"invalid grasp side: {side}")
        self._send_data("ReleaseLeft" if side == "left" else "ReleaseRight")

    def request_grasp_state(self) -> None:
        self._send_data("GetGraspState")

    def request_attached_particle_indices(self, side: str) -> None:
        side = str(side).lower()
        if side not in SIDES:
            raise ValueError(f"invalid grasp side: {side}")
        self._send_data("GetAttachedParticleIndices", side)

    def request_grasp_constraint_error(self, side: str) -> None:
        side = str(side).lower()
        if side not in SIDES:
            raise ValueError(f"invalid grasp side: {side}")
        self._send_data("GetGraspForceOrConstraintError", side)

    def grasp_states(self) -> Tuple[GraspState, ...]:
        values: Iterable[Mapping[str, Any]] = self.data.get("grasp_state", ())
        return tuple(GraspState.from_mapping(value) for value in values)

    def request_contacts(self) -> None:
        self._send_data("GetClothContacts")

    def contacts(self) -> Tuple[ClothContact, ...]:
        return tuple(
            ClothContact.from_mapping(value)
            for value in self.data.get("cloth_contacts", ())
        )

    def request_coverage(self) -> None:
        self._send_data("GetCoverageObservations")

    def coverage(self) -> Mapping[str, Any]:
        value = dict(self.data.get("coverage_observations", {}))
        if value and not bool(value.get("valid", False)):
            value["coverage"] = None
        return value

    def reset(self) -> None:
        self._send_data("ResetSock")

    def SetTransform(self, position=None, rotation=None, **_kwargs: Any) -> None:
        self._send_data("SetTransform", position, rotation)

    # Compatibility with the distributed ClothAttr naming.
    def GetParticles(self) -> None:
        self.request_particles()
