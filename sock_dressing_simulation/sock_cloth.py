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
    peak_constraint_error: float = 0.0
    over_threshold_steps: int = 0
    release_reason: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GraspState":
        side = str(value["side"]).lower()
        if side not in SIDES:
            raise ValueError(f"invalid grasp side: {side}")
        indices = tuple(int(index) for index in value.get("particle_indices", ()))
        if any(index < 0 for index in indices):
            raise ValueError("particle indices must be non-negative")
        error = float(value.get("constraint_error", 0.0))
        peak = float(value.get("peak_constraint_error", error))
        threshold_steps = int(value.get("over_threshold_steps", 0))
        if not np.isfinite(error) or error < 0 or not np.isfinite(peak) or peak < 0:
            raise ValueError("constraint_error must be finite and non-negative")
        if threshold_steps < 0:
            raise ValueError("over_threshold_steps must be non-negative")
        attached = bool(value.get("attached", False))
        if attached != bool(indices):
            raise ValueError("attached must agree with particle_indices")
        return cls(
            side,
            attached,
            indices,
            error,
            peak,
            threshold_steps,
            str(value.get("release_reason", "")),
        )


@dataclass(frozen=True)
class SceneGeometry:
    opening_center: Tuple[float, float, float]
    opening_normal: Tuple[float, float, float]
    opening_outward_normal: Tuple[float, float, float]
    sock_body_direction: Tuple[float, float, float]
    sock_body_gravity_alignment: float
    right_toe_position: Tuple[float, float, float]
    foot_to_opening_plane_m: float
    foot_to_opening_lateral_m: float
    right_leg_raise_degrees: float
    right_knee_flexion_degrees: float
    left_grasp_position: Tuple[float, float, float]
    right_grasp_position: Tuple[float, float, float]
    left_opening_edge: Tuple[float, float, float]
    right_opening_edge: Tuple[float, float, float]
    opening_to_toe_alignment: float
    left_cuff_insertion_depth_m: float
    right_cuff_insertion_depth_m: float
    left_grasp_parent: str = ""
    right_grasp_parent: str = ""
    left_grasp_local_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_grasp_local_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    opening_target_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    left_grasp_thickness_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    right_grasp_thickness_axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    left_grasp_inward_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    right_grasp_inward_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    opening_span_m: float = 0.0
    left_grasp_corner_negative: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_grasp_corner_positive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_grasp_corner_negative: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_grasp_corner_positive: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    left_grasp_patch_span_m: float = 0.0
    right_grasp_patch_span_m: float = 0.0
    maximum_grasp_corner_error_m: float = 0.0
    grasp_thickness_axis_alignment: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SceneGeometry":
        if not bool(value.get("valid", False)):
            raise ValueError("Player did not return valid sock/right-leg geometry")
        vector_names = (
            "opening_center",
            "opening_normal",
            "opening_outward_normal",
            "sock_body_direction",
            "right_toe_position",
            "left_grasp_position",
            "right_grasp_position",
            "left_opening_edge",
            "right_opening_edge",
            "left_grasp_local_offset",
            "right_grasp_local_offset",
            "opening_target_normal",
            "left_grasp_thickness_axis",
            "right_grasp_thickness_axis",
            "left_grasp_inward_axis",
            "right_grasp_inward_axis",
            "left_grasp_corner_negative",
            "left_grasp_corner_positive",
            "right_grasp_corner_negative",
            "right_grasp_corner_positive",
        )
        vectors = {}
        for name in vector_names:
            fallback_name = {
                "opening_outward_normal": "opening_normal",
                "left_opening_edge": "left_grasp_position",
                "right_opening_edge": "right_grasp_position",
                "opening_target_normal": "opening_normal",
                "left_grasp_thickness_axis": None,
                "right_grasp_thickness_axis": None,
                "left_grasp_inward_axis": "opening_target_normal",
                "right_grasp_inward_axis": "opening_target_normal",
                "left_grasp_corner_negative": "left_grasp_position",
                "left_grasp_corner_positive": "left_grasp_position",
                "right_grasp_corner_negative": "right_grasp_position",
                "right_grasp_corner_positive": "right_grasp_position",
            }.get(name)
            fallback = (
                value.get(fallback_name, ())
                if fallback_name
                else (
                    (1.0, 0.0, 0.0)
                    if name.endswith("_thickness_axis")
                    else (
                        (0.0, 0.0, 0.0)
                        if name.endswith("_local_offset")
                        else ()
                    )
                )
            )
            vector = tuple(float(item) for item in value.get(name, fallback))
            if len(vector) != 3 or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must contain three finite values")
            vectors[name] = vector
        distance = float(value["foot_to_opening_plane_m"])
        lateral = float(value["foot_to_opening_lateral_m"])
        angle = float(value["right_leg_raise_degrees"])
        knee_flexion = float(value.get("right_knee_flexion_degrees", 0.0))
        gravity_alignment = float(value["sock_body_gravity_alignment"])
        toe_alignment = float(value.get("opening_to_toe_alignment", -1.0))
        left_insertion = float(value.get("left_cuff_insertion_depth_m", 0.0))
        right_insertion = float(value.get("right_cuff_insertion_depth_m", 0.0))
        opening_span = float(
            value.get(
                "opening_span_m",
                np.linalg.norm(
                    np.asarray(vectors["right_opening_edge"])
                    - np.asarray(vectors["left_opening_edge"])
                ),
            )
        )
        left_patch_span = float(value.get("left_grasp_patch_span_m", 0.0))
        right_patch_span = float(value.get("right_grasp_patch_span_m", 0.0))
        maximum_corner_error = float(
            value.get("maximum_grasp_corner_error_m", 0.0)
        )
        thickness_alignment = float(
            value.get("grasp_thickness_axis_alignment", 0.0)
        )
        if (
            not np.isfinite(distance)
            or distance < 0
            or not np.isfinite(lateral)
            or lateral < 0
            or not np.isfinite(angle)
            or not np.isfinite(knee_flexion)
            or knee_flexion < 0
            or knee_flexion > 180
            or not np.isfinite(gravity_alignment)
            or gravity_alignment < -1.0
            or gravity_alignment > 1.0
            or not np.isfinite(toe_alignment)
            or toe_alignment < -1.0
            or toe_alignment > 1.0
            or not np.isfinite(left_insertion)
            or left_insertion < 0
            or not np.isfinite(right_insertion)
            or right_insertion < 0
            or not np.isfinite(opening_span)
            or opening_span < 0
            or not np.isfinite(left_patch_span)
            or left_patch_span < 0
            or not np.isfinite(right_patch_span)
            or right_patch_span < 0
            or not np.isfinite(maximum_corner_error)
            or maximum_corner_error < 0
            or not np.isfinite(thickness_alignment)
            or thickness_alignment < 0
            or thickness_alignment > 1
        ):
            raise ValueError(
                "scene distances and angles are invalid: "
                f"distance={distance}, lateral={lateral}, angle={angle}, "
                f"knee_flexion={knee_flexion}, "
                f"gravity_alignment={gravity_alignment}, "
                f"toe_alignment={toe_alignment}, "
                f"left_insertion={left_insertion}, "
                f"right_insertion={right_insertion}, "
                f"opening_span={opening_span}, "
                f"left_patch_span={left_patch_span}, "
                f"right_patch_span={right_patch_span}, "
                f"maximum_corner_error={maximum_corner_error}, "
                f"thickness_alignment={thickness_alignment}"
            )
        return cls(
            foot_to_opening_plane_m=distance,
            foot_to_opening_lateral_m=lateral,
            right_leg_raise_degrees=angle,
            right_knee_flexion_degrees=knee_flexion,
            sock_body_gravity_alignment=gravity_alignment,
            opening_to_toe_alignment=toe_alignment,
            left_cuff_insertion_depth_m=left_insertion,
            right_cuff_insertion_depth_m=right_insertion,
            left_grasp_parent=str(value.get("left_grasp_parent", "")),
            right_grasp_parent=str(value.get("right_grasp_parent", "")),
            opening_span_m=opening_span,
            left_grasp_patch_span_m=left_patch_span,
            right_grasp_patch_span_m=right_patch_span,
            maximum_grasp_corner_error_m=maximum_corner_error,
            grasp_thickness_axis_alignment=thickness_alignment,
            **vectors,
        )


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


@dataclass(frozen=True)
class DressingQA:
    valid: bool
    simulation_frame: int
    surface_containment_ratio: float
    sections: Tuple[Mapping[str, Any], ...]
    cuff_progress_toward_ankle_m: float
    maximum_cuff_reverse_step_m: float
    cuff_beyond_distal_toe_m: float
    foot_contact_count: int
    maximum_cloth_foot_penetration_m: float
    maximum_cloth_foot_force_proxy: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DressingQA":
        required = (
            "surface_containment_ratio",
            "cuff_progress_toward_ankle_m",
            "maximum_cuff_reverse_step_m",
            "cuff_beyond_distal_toe_m",
            "maximum_cloth_foot_penetration_m",
            "maximum_cloth_foot_force_proxy",
        )
        scalars = np.asarray([value[name] for name in required], dtype=float)
        sections = tuple(dict(item) for item in value.get("sections", ()))
        section_values = np.asarray(
            [item.get("containment_ratio", np.nan) for item in sections],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(scalars))
            or not sections
            or not np.all(np.isfinite(section_values))
            or np.any(section_values < 0)
            or np.any(section_values > 1)
            or scalars[0] < 0
            or scalars[0] > 1
            or np.any(scalars[2:] < 0)
        ):
            raise ValueError("dressing QA values must be finite and in range")
        return cls(
            valid=bool(value.get("valid", False)),
            simulation_frame=int(value.get("simulation_frame", -1)),
            surface_containment_ratio=float(scalars[0]),
            sections=sections,
            cuff_progress_toward_ankle_m=float(scalars[1]),
            maximum_cuff_reverse_step_m=float(scalars[2]),
            cuff_beyond_distal_toe_m=float(scalars[3]),
            foot_contact_count=int(value.get("foot_contact_count", 0)),
            maximum_cloth_foot_penetration_m=float(scalars[4]),
            maximum_cloth_foot_force_proxy=float(scalars[5]),
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
        "damping",
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
        self.data.update(data)

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
        stretching_scale: float,
        particle_radius_m: float,
        particle_mass_kg: float,
        collision_margin_m: float,
        friction: float,
        self_collision: bool,
        damping: float,
        substeps: int,
        solver_iterations: int,
        tether_constraints: bool = True,
        tether_compliance: float = 0.0,
        tether_scale: float = 1.0,
        maximum_circumferential_stretch: float = 1.5,
        strain_limit_iterations: int = 8,
    ) -> None:
        self._send_data(
            "ConfigureSock",
            float(stretch_compliance),
            float(bend_compliance),
            float(stretching_scale),
            float(particle_radius_m),
            float(particle_mass_kg),
            float(collision_margin_m),
            float(friction),
            bool(self_collision),
            float(damping),
            int(substeps),
            int(solver_iterations),
            bool(tether_constraints),
            float(tether_compliance),
            float(tether_scale),
            float(maximum_circumferential_stretch),
            int(strain_limit_iterations),
        )

    def configure_grasp(
        self,
        *,
        linear_compliance: float,
        rotational_compliance: float,
        break_threshold: float,
        slip_constraint_error_m: float,
        slip_opening_span_m: float,
        slip_consecutive_steps: int,
        maximum_particles_per_side: int,
        cuff_insertion_depth_m: float,
        grasp_thickness_half_width_m: float,
    ) -> None:
        values = np.asarray(
            [
                linear_compliance,
                rotational_compliance,
                break_threshold,
                slip_constraint_error_m,
                slip_opening_span_m,
                cuff_insertion_depth_m,
                grasp_thickness_half_width_m,
            ],
            dtype=float,
        )
        if (
            not np.all(np.isfinite(values))
            or linear_compliance < 0
            or rotational_compliance <= 0
            or break_threshold <= 0
            or slip_constraint_error_m <= 0
            or slip_opening_span_m <= 0
            or cuff_insertion_depth_m <= 0
            or grasp_thickness_half_width_m <= 0
            or int(slip_consecutive_steps) < 1
            or int(maximum_particles_per_side) < 2
        ):
            raise ValueError("invalid grasp/slip configuration")
        self._send_data(
            "ConfigureGrasp",
            float(linear_compliance),
            float(rotational_compliance),
            float(break_threshold),
            float(slip_constraint_error_m),
            float(slip_opening_span_m),
            int(slip_consecutive_steps),
            int(maximum_particles_per_side),
            float(cuff_insertion_depth_m),
            float(grasp_thickness_half_width_m),
        )

    def set_grasp_targets(self, left_id: int, right_id: int) -> None:
        self._send_data("SetGraspTargets", int(left_id), int(right_id))

    def align_grasp_targets_to_opening(self) -> None:
        self._send_data("AlignGraspTargetsToOpening")

    def clamp_grasp_target_span(self, maximum_span_m: float) -> None:
        span = float(maximum_span_m)
        if not np.isfinite(span) or span <= 0:
            raise ValueError("maximum grasp target span must be finite and positive")
        self._send_data("ClampGraspTargetSpan", span)

    def align_sock_opening_to_grasp_targets(
        self, toe_target: Sequence[float]
    ) -> None:
        target = np.asarray(toe_target, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("right toe target must be a finite 3-vector")
        self._send_data("AlignSockOpeningToGraspTargets", *target.tolist())

    def align_sock_opening_to_grasp_targets_and_grasp(
        self, toe_target: Sequence[float], max_distance_m: float
    ) -> None:
        target = np.asarray(toe_target, dtype=float)
        distance = float(max_distance_m)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("right toe target must be a finite 3-vector")
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("max_distance_m must be finite and positive")
        self._send_data(
            "AlignSockOpeningToGraspTargetsAndGrasp",
            *target.tolist(),
            distance,
        )

    def align_sock_opening_to_grasp_plate_and_grasp(
        self, max_distance_m: float
    ) -> None:
        distance = float(max_distance_m)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("max_distance_m must be finite and positive")
        self._send_data("AlignSockOpeningToGraspPlateAndGrasp", distance)

    def set_grasp_target_position(
        self, side: str, position: Sequence[float]
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError(f"unsupported grasp side: {side}")
        value = np.asarray(position, dtype=float)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("grasp target position must be a finite 3-vector")
        self._send_data("SetGraspTargetPosition", side, *value.tolist())

    def configure_right_leg_colliders(
        self,
        human_id: int,
        foot_cross_section_scale: float = 1.0,
    ) -> None:
        scale = float(foot_cross_section_scale)
        if not np.isfinite(scale) or scale <= 0 or scale > 1:
            raise ValueError(
                "foot_cross_section_scale must be finite and in (0, 1]"
            )
        self._send_data(
            "ConfigureRightLegColliders",
            int(human_id),
            scale,
        )

    def configure_human_visual_pose(
        self, seat_position: Sequence[float], foot_position: Sequence[float]
    ) -> None:
        seat = np.asarray(seat_position, dtype=float)
        foot = np.asarray(foot_position, dtype=float)
        if (
            seat.shape != (3,)
            or foot.shape != (3,)
            or not np.all(np.isfinite(seat))
            or not np.all(np.isfinite(foot))
        ):
            raise ValueError("human visual pose points must be finite 3-vectors")
        self._send_data("ConfigureHumanVisualPose", *seat.tolist(), *foot.tolist())

    def configure_human_task_pose(
        self,
        seat_position: Sequence[float],
        foot_position: Sequence[float],
        seat_scale: Sequence[float],
        plantarflexion_degrees: float = 0.0,
        straight_right_leg: bool = False,
    ) -> None:
        seat = np.asarray(seat_position, dtype=float)
        foot = np.asarray(foot_position, dtype=float)
        scale = np.asarray(seat_scale, dtype=float)
        plantarflexion = float(plantarflexion_degrees)
        if (
            seat.shape != (3,)
            or foot.shape != (3,)
            or scale.shape != (3,)
            or not np.all(np.isfinite(seat))
            or not np.all(np.isfinite(foot))
            or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)
            or not np.isfinite(plantarflexion)
        ):
            raise ValueError(
                "human task pose points, plantarflexion, and seat scale are invalid"
            )
        self._send_data(
            "ConfigureHumanTaskPose",
            *seat.tolist(),
            *foot.tolist(),
            plantarflexion,
            *scale.tolist(),
            bool(straight_right_leg),
        )

    def set_task_right_toe_position(self, position: Sequence[float]) -> None:
        vector = np.asarray(position, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("task right toe position must be a finite 3-vector")
        self._send_data("SetTaskRightToePosition", *vector.tolist())

    def set_task_right_toe_position_articulated(
        self, position: Sequence[float]
    ) -> None:
        vector = np.asarray(position, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("task right toe position must be a finite 3-vector")
        self._send_data(
            "SetTaskRightToePositionArticulated", *vector.tolist()
        )

    def ignore_robot_human_rigid_collisions(self, robot_id: int) -> None:
        self._send_data("IgnoreRobotHumanRigidCollisions", int(robot_id))

    def ignore_non_gripper_robot_human_rigid_collisions(
        self, robot_id: int
    ) -> None:
        self._send_data(
            "IgnoreNonGripperRobotHumanRigidCollisions", int(robot_id)
        )

    def align_human_visual_foot_to_sock(self, distance_m: float) -> None:
        distance = float(distance_m)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("visual foot distance must be finite and positive")
        self._send_data("AlignHumanVisualFootToSock", distance)

    def translate_human_and_ik(self, delta: Sequence[float]) -> None:
        vector = np.asarray(delta, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("human translation must contain three finite values")
        self._send_data("TranslateHumanAndIK", *vector.tolist())

    def freeze_human_right_toe_at(self, position: Sequence[float]) -> None:
        vector = np.asarray(position, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("right toe target must contain three finite values")
        self._send_data("FreezeHumanRightToeAt", *vector.tolist())

    def set_foot_clearance_target(
        self, distance_m: float, chair_id: int = -1
    ) -> None:
        distance = float(distance_m)
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("foot clearance target must be finite and positive")
        self._send_data("SetFootClearanceTarget", distance, int(chair_id))

    def stop_foot_clearance_tracking(self) -> None:
        self._send_data("StopFootClearanceTracking")

    def restore_human_chair_world_pose(
        self,
        human_root_position: Sequence[float],
        chair_position: Sequence[float],
        human_anchor_position: Sequence[float],
        right_toe_position: Sequence[float],
        chair_id: int = -1,
    ) -> None:
        vectors = [
            np.asarray(value, dtype=float)
            for value in (
                human_root_position,
                chair_position,
                human_anchor_position,
                right_toe_position,
            )
        ]
        if any(
            value.shape != (3,) or not np.all(np.isfinite(value))
            for value in vectors
        ):
            raise ValueError(
                "restored human/chair pose requires four finite 3-vectors"
            )
        self._send_data(
            "RestoreHumanChairWorldPose",
            *(item for value in vectors for item in value.tolist()),
            int(chair_id),
        )

    def lock_human_and_chair(self, chair_id: int = -1) -> None:
        self._send_data("LockHumanAndChair", int(chair_id))

    def arm_slip_detection(self, armed: bool = True) -> None:
        self._send_data("ArmSlipDetection", bool(armed))

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

    def request_scene_geometry(self) -> None:
        self._send_data("GetSceneGeometry")

    def request_dressing_qa(self) -> None:
        self._send_data("GetDressingQA")

    def dressing_qa(self) -> DressingQA:
        return DressingQA.from_mapping(self.data.get("dressing_qa", {}))

    def request_visual_diagnostics(self, chair_id: int = -1) -> None:
        self._send_data("GetVisualDiagnostics", int(chair_id))

    def visual_diagnostics(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self.data.get("visual_diagnostics", ()))

    def scene_geometry(self) -> SceneGeometry:
        return SceneGeometry.from_mapping(self.data.get("scene_geometry", {}))

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

    def request_robot_human_rigid_collision_qa(self, robot_id: int) -> None:
        self._send_data("GetRobotHumanRigidCollisionQA", int(robot_id))

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

    def SetTransform(
        self,
        position=None,
        rotation=None,
        scale=None,
        is_world: bool = True,
        **_kwargs: Any,
    ) -> None:
        self._send_data(
            "SetTransform",
            position,
            rotation,
            scale if scale is not None else [1.0, 1.0, 1.0],
            bool(is_world),
        )

    # Compatibility with the distributed ClothAttr naming.
    def GetParticles(self) -> None:
        self.request_particles()
