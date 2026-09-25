from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from PIL import Image


class SockDressingEnv:
    """Thin lazy adapter over the API at the pinned RCareWorld commit."""

    def __init__(self, config: Mapping, backend: Optional[Any] = None):
        self.config = config
        self._env = backend
        self.robot = None
        self.human = None
        self.cloth = None
        self.camera = None
        self.recording_camera = None
        self.chair = []
        self.grasp_anchors = []
        self._grasp_attachment_report = []
        self._camera_mount_report: Dict[str, Any] = {}
        self.sock_cloth = None
        self._rollout_started = False
        self._right_toe_offset_report: Optional[Dict[str, Any]] = None
        self._human_chair_translation_report: Optional[Dict[str, Any]] = None
        self._effective_locked_pose_baseline: Optional[Dict[str, Any]] = None

    def connect(self) -> "SockDressingEnv":
        if self._env is None:
            try:
                from pyrcareworld.envs.base_env import RCareWorld
            except ImportError as error:
                raise RuntimeError(
                    "pyrcareworld is not installed; run scripts/setup_rcareworld.sh"
                ) from error
            if self.config["rcareworld"].get("profile") == "custom_player":
                import pyrcareworld.attributes as attributes

                from .sock_cloth import SockClothAttr

                attributes.attrs["SockClothAttr"] = SockClothAttr
            settings = self.config["rcareworld"]
            scene = self.config["scene"]
            runtime_assets = ["Camera", "Empty"]
            chair = scene.get("chair", {})
            if chair.get("enabled", False):
                runtime_assets.append(str(chair.get("asset_name", "Collider_Box")))
            kwargs = {
                "port": int(settings.get("port", 5005)),
                "graphics": bool(settings.get("graphics", False)),
                "assets": list(dict.fromkeys(runtime_assets)),
                "log_level": int(settings.get("log_level", 1)),
            }
            if settings.get("executable"):
                from .config import resolve_package_path

                kwargs["executable_file"] = str(
                    resolve_package_path(settings["executable"]).resolve()
                )
            if self.config["scene"].get("scene_file"):
                kwargs["scene_file"] = self.config["scene"]["scene_file"]
            assimp_dir = settings.get("assimp_library_dir")
            previous_library_path = os.environ.get("LD_LIBRARY_PATH")
            if assimp_dir:
                from .config import resolve_package_path

                resolved = str(resolve_package_path(assimp_dir).resolve())
                os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                    [resolved] + ([previous_library_path] if previous_library_path else [])
                )
            try:
                self._env = RCareWorld(**kwargs)
            finally:
                if previous_library_path is None:
                    os.environ.pop("LD_LIBRARY_PATH", None)
                else:
                    os.environ["LD_LIBRARY_PATH"] = previous_library_path
        return self

    def load(
        self,
        urdf: Path,
        sock_obj: Path,
        initial_joints: Optional[Any] = None,
    ) -> None:
        self.connect()
        if self.config["rcareworld"].get("profile") == "custom_player":
            if self.config["rcareworld"].get("native_rcareworld", False):
                self._load_native_custom_player(urdf, sock_obj, initial_joints)
                return
            self._load_custom_player()
            return
        assets = self.config["assets"]
        self.robot = self._env.LoadURDF(
            str(Path(urdf).resolve()), id=int(assets["robot_id"]), native_ik=False, axis="z"
        )
        self._env.step()
        self._validate_simulator_joint_mapping()
        if initial_joints is None:
            initial_joints = self.config.get("scenario", {}).get("initial_joints")
        if initial_joints is not None:
            target = np.asarray(initial_joints, dtype=float)
            current = self.robot_signals()["angle"]
            command_steps = 0
            while not np.allclose(current, target, atol=1e-6, rtol=0):
                current = self.command(target, current)
                command_steps += 1
                if command_steps > 1000:
                    raise RuntimeError(
                        "initial robot pose did not converge before cloth loading"
                    )
        self.cloth = self._env.LoadCloth(
            str(Path(sock_obj).resolve()), id=int(assets["sock_id"])
        )
        self._env.EnabledGroundObiCollider(True)
        self.robot.SetImmovable(bool(self.config["scene"].get("robot_immovable", True)))
        self.robot.SetTransform(
            position=list(self.config["scene"]["robot_position"]),
            rotation=list(self.config["scene"].get("robot_rotation", [0, 0, 0])),
        )
        self.cloth.SetTransform(position=list(self.config["scene"]["sock_position"]))
        self.robot.AddObiCollider()
        self._env.step()
        scene = self.config["scene"]
        if scene.get("human_id") is not None:
            self.human = self._env.GetAttr(int(scene["human_id"]))
            if scene.get("human_position") is not None:
                self.human.SetTransform(
                    position=list(scene["human_position"]),
                    rotation=list(scene.get("human_rotation", [0, 0, 0])),
                )
            if scene.get("enable_human_obi_collider", False):
                self.human.AddObiCollider()
        self._create_chair()
        self._create_native_cameras()

    def _load_native_custom_player(
        self,
        urdf: Path,
        sock_obj: Path,
        initial_joints: Optional[Any],
    ) -> None:
        """Load the sock task through RCareCommon's native PlayerMain APIs."""
        from .sock_cloth import SockClothAttr, validate_configuration

        assets = self.config["assets"]
        scene = self.config["scene"]
        # Freeze the preloaded human articulation before the robot's long
        # initialization loop can pull the rig away from its animated bind pose.
        self._env._send_env_data(
            "EnsureSockHuman",
            str(scene.get("human_asset_name", "male1_c6-c7")),
            int(scene["human_id"]),
        )
        self._env.step()
        self.human = self._env.GetAttr(int(scene["human_id"]))
        if scene.get("human_position") is not None:
            self.human.SetTransform(
                position=list(scene["human_position"]),
                rotation=list(scene.get("human_rotation", [0, 0, 0])),
                scale=[1.0, 1.0, 1.0],
            )
        self.robot = self._env.LoadURDF(
            str(Path(urdf).resolve()),
            id=int(assets["robot_id"]),
            native_ik=False,
            axis="z",
        )
        self._env.step()
        self._validate_simulator_joint_mapping()
        target = (
            np.asarray(initial_joints, dtype=float)
            if initial_joints is not None
            else np.asarray(self.config["scenario"]["initial_joints"], dtype=float)
        )
        current = self.robot_signals()["angle"]
        for _ in range(1001):
            if np.allclose(current, target, atol=1e-6, rtol=0):
                break
            current = self.command(target, current)
        else:
            raise RuntimeError("initial robot pose did not converge before sock loading")

        self._env._send_env_data(
            "LoadSockCloth",
            str(Path(sock_obj).resolve()),
            int(assets["sock_id"]),
        )
        self._env.step()
        loaded = self._env.GetAttr(int(assets["sock_id"]))
        self.sock_cloth = (
            loaded
            if isinstance(loaded, SockClothAttr)
            else SockClothAttr(
                self._env,
                int(assets["sock_id"]),
                getattr(loaded, "data", {}),
            )
        )
        self._env.attrs[int(assets["sock_id"])] = self.sock_cloth
        self.cloth = self.sock_cloth

        self._env.EnabledGroundObiCollider(True)
        self.robot.SetImmovable(bool(scene.get("robot_immovable", True)))
        self.robot.SetJointUseGravity(
            bool(scene.get("robot_joint_gravity", False))
        )
        self.robot.SetTransform(
            position=list(scene["robot_position"]),
            rotation=list(scene.get("robot_rotation", [0, 0, 0])),
            scale=[1.0, 1.0, 1.0],
        )
        self._env._send_env_data(
            "EnsureRobotObiColliders",
            int(assets["robot_id"]),
        )
        self.cloth.SetTransform(
            position=list(scene["sock_position"]),
            rotation=list(scene.get("sock_rotation", [0, 0, 0])),
            scale=[1.0, 1.0, 1.0],
        )
        self.sock_cloth.configure_right_leg_colliders(int(scene["human_id"]))
        frozen_toe = scene.get("visuals", {}).get("frozen_right_toe_position")
        if frozen_toe is not None:
            self.sock_cloth.freeze_human_right_toe_at(frozen_toe)
            self._env.step()
        self._create_chair()
        chair_parts = scene.get("chair", {}).get("parts", ())
        pose_contract = scene.get("initial_pose_contract", {})
        visuals = scene.get("visuals", {})
        if chair_parts and pose_contract:
            foot_visual = np.asarray(scene["sock_position"], dtype=float)
            foot_visual[2] -= float(pose_contract.get("foot_to_sock_m", 0.10))
            foot_visual = np.asarray(
                visuals.get("task_right_toe_position", foot_visual),
                dtype=float,
            )
            if visuals.get("human_proxy", False):
                self.sock_cloth.configure_human_visual_pose(
                    chair_parts[0]["position"],
                    foot_visual.tolist(),
                )
            else:
                self.sock_cloth.configure_human_task_pose(
                    chair_parts[0]["position"],
                    foot_visual.tolist(),
                    chair_parts[0]["scale"],
                    float(
                        self.config.get("scenario", {})
                        .get("foot", {})
                        .get("plantarflexion_degrees", 0.0)
                    ),
                    bool(pose_contract.get("straight_right_leg", False)),
                )
        if scene.get(
            "ignore_non_gripper_robot_human_rigid_collisions", False
        ):
            self.sock_cloth.ignore_non_gripper_robot_human_rigid_collisions(
                int(assets["robot_id"])
            )
        elif scene.get("ignore_robot_human_rigid_collisions", True):
            self.sock_cloth.ignore_robot_human_rigid_collisions(
                int(assets["robot_id"])
            )
        self._create_native_cameras()
        self.sock_cloth.configure_mask_proxy_cameras()
        self._create_grasp_anchors()
        anchors = {
            str(item["side"]): int(item["id"])
            for item in scene.get("grasp_anchors", {}).get("anchors", ())
        }
        if set(anchors) != {"left", "right"}:
            raise RuntimeError("native custom Player requires left/right grasp anchors")
        self.sock_cloth.set_grasp_targets(anchors["left"], anchors["right"])

        expected = self.config.get("obi", {}).get("expected", {})
        self.sock_cloth.configure(
            stretch_compliance=float(expected["stretch_compliance"]),
            bend_compliance=float(expected["bend_compliance"]),
            stretching_scale=float(expected["stretching_scale"]),
            particle_radius_m=float(expected["particle_radius_m"]),
            particle_mass_kg=float(expected["particle_mass_kg"]),
            collision_margin_m=float(expected["collision_margin_m"]),
            friction=float(expected["friction"]),
            self_collision=bool(expected["self_collision"]),
            damping=float(expected["damping"]),
            substeps=int(expected["substeps"]),
            solver_iterations=int(expected["solver_iterations"]),
            tether_constraints=bool(expected["tether_constraints"]),
            tether_compliance=float(expected["tether_compliance"]),
            tether_scale=float(expected["tether_scale"]),
            maximum_circumferential_stretch=float(
                expected["maximum_circumferential_stretch"]
            ),
            strain_limit_iterations=int(expected["strain_limit_iterations"]),
        )
        self.sock_cloth.configure_grasp(
            linear_compliance=float(expected["grasp_linear_compliance"]),
            rotational_compliance=float(expected["grasp_rotational_compliance"]),
            break_threshold=float(expected["grasp_break_threshold"]),
            slip_constraint_error_m=float(expected["slip_constraint_error_m"]),
            slip_opening_span_m=float(expected["slip_opening_span_m"]),
            slip_consecutive_steps=int(expected["slip_consecutive_steps"]),
            maximum_particles_per_side=int(
                expected["maximum_grasp_particles_per_side"]
            ),
            cuff_insertion_depth_m=float(expected["cuff_insertion_depth_m"]),
            grasp_thickness_half_width_m=float(
                expected["grasp_thickness_half_width_m"]
            ),
        )
        self.sock_cloth.request_configuration()
        self.sock_cloth.request_registered_colliders()
        self._env.step()
        report = validate_configuration(self.sock_cloth.configuration(), expected)
        if not report["ok"]:
            raise RuntimeError(f"native custom Player cloth contract mismatch: {report}")
        enabled_ids = {
            int(item["object_id"])
            for item in self.sock_cloth.registered_colliders()
            if item.get("enabled", False)
        }
        required = set(int(value) for value in scene.get("contact_proxy_ids", ()))
        if not required.issubset(enabled_ids):
            raise RuntimeError(
                "native custom Player is missing enabled Obi colliders: "
                f"{sorted(required - enabled_ids)}"
            )

    def _create_native_cameras(self) -> None:
        import pyrcareworld.attributes as attr

        scene = self.config["scene"]
        if scene.get("camera_id") is not None:
            self.camera = self._env.InstanceObject(
                name="Camera",
                id=int(scene["camera_id"]),
                attr_type=attr.CameraAttr,
            )
            parent_link = scene.get("camera_parent_link")
            local_position = scene.get("camera_local_position", [0.0, 0.0, 0.0])
            local_rotation = scene.get("camera_local_rotation", [0.0, 0.0, 0.0])
            if parent_link:
                if self.robot is None:
                    raise RuntimeError(
                        "camera_parent_link requires a loaded Dry-AIREC robot"
                    )
                self.camera.SetParent(self.robot.id, str(parent_link))
                self.camera.SetTransform(
                    position=list(local_position),
                    rotation=list(local_rotation),
                    scale=[1.0, 1.0, 1.0],
                    is_world=False,
                )
                self._camera_mount_report = {
                    "mode": "robot_link",
                    "parent_id": int(self.robot.id),
                    "parent_link": str(parent_link),
                    "local_position": list(local_position),
                    "local_rotation": list(local_rotation),
                }
            else:
                self.camera.SetTransform(
                    position=list(scene["camera_position"]),
                    rotation=list(scene["camera_rotation"]),
                    scale=[1.0, 1.0, 1.0],
                )
                self._camera_mount_report = {
                    "mode": "world",
                    "position": list(scene["camera_position"]),
                    "rotation": list(scene["camera_rotation"]),
                }
        if scene.get("recording_camera_id") is not None:
            self.recording_camera = self._env.InstanceObject(
                name="Camera",
                id=int(scene["recording_camera_id"]),
                attr_type=attr.CameraAttr,
            )
            self.recording_camera.SetTransform(
                position=list(scene["recording_camera_position"]),
                rotation=list(scene["recording_camera_rotation"]),
                scale=[1.0, 1.0, 1.0],
            )
        self._env.step()
        camera_data = getattr(self.camera, "data", {}) if self.camera is not None else {}
        if self._camera_mount_report and camera_data:
            self._camera_mount_report["world_position"] = list(
                camera_data.get("position", ())
            )
            self._camera_mount_report["world_rotation"] = list(
                camera_data.get("rotation", ())
            )

    def _load_custom_player(self) -> None:
        """Bind to objects authored into the greenfield Player scene."""
        from .sock_cloth import SockClothAttr, validate_configuration

        self._env.step()
        assets = self.config["assets"]
        scene = self.config["scene"]
        self.robot = self._env.GetAttr(int(assets["robot_id"]))
        self.human = self._env.GetAttr(int(scene["human_id"]))
        distributed_cloth = self._env.GetAttr(int(assets["sock_id"]))
        self.sock_cloth = (
            distributed_cloth
            if isinstance(distributed_cloth, SockClothAttr)
            else SockClothAttr(
                self._env,
                int(assets["sock_id"]),
                getattr(distributed_cloth, "data", {}),
            )
        )
        self._env.attrs[int(assets["sock_id"])] = self.sock_cloth
        self.cloth = self.sock_cloth
        self.sock_cloth.request_configuration()
        self.sock_cloth.request_registered_colliders()
        self._env.step()
        report = validate_configuration(
            self.sock_cloth.configuration(),
            self.config.get("obi", {}).get("expected", {}),
        )
        if not report["ok"]:
            raise RuntimeError(f"custom Player cloth contract mismatch: {report}")
        colliders = self.sock_cloth.registered_colliders()
        enabled_ids = {
            int(item["object_id"]) for item in colliders if item.get("enabled", False)
        }
        required = set(int(value) for value in scene.get("contact_proxy_ids", ()))
        if not required.issubset(enabled_ids):
            raise RuntimeError(
                "custom Player is missing enabled Obi colliders: "
                f"{sorted(required - enabled_ids)}"
            )
        self._validate_simulator_joint_mapping()

    def _create_chair(self) -> None:
        chair = self.config["scene"].get("chair", {})
        if not chair.get("enabled", False):
            return
        try:
            import pyrcareworld.attributes as attr
        except ImportError as error:
            raise RuntimeError("pyrcareworld attributes are unavailable") from error
        asset_name = str(chair.get("asset_name", "Collider_Box"))
        for part in chair.get("parts", []):
            collider = self._env.InstanceObject(
                name=asset_name,
                id=int(part["id"]),
                attr_type=attr.ColliderAttr,
            )
            collider.SetTransform(
                position=list(part["position"]),
                rotation=list(part.get("rotation", [0, 0, 0])),
                scale=list(part["scale"]),
            )
            collider.AddObiCollider()
            self.chair.append(collider)
        self._env.step()

    def _create_grasp_anchors(self) -> None:
        settings = self.config["scene"].get("grasp_anchors", {})
        if not settings.get("enabled", False) or self.grasp_anchors:
            return
        max_distance = float(settings.get("max_distance_m", 0.03))
        for item in settings.get("anchors", []):
            anchor = self._env.InstanceObject("Empty", id=int(item["id"]))
            parent_link = item.get("parent_link")
            if parent_link:
                anchor.SetParent(self.robot.id, str(parent_link))
            local_position = item.get("local_position")
            anchor.SetTransform(
                position=list(
                    local_position
                    if local_position is not None
                    else item.get("position", [0.0, 0.0, 0.0])
                ),
                rotation=[0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0],
                is_world=local_position is None,
            )
            attached = bool(settings.get("attach", False))
            if attached:
                self.cloth.AddAttach(anchor.id, max_dis=max_distance)
            self.grasp_anchors.append(anchor)
            self._grasp_attachment_report.append(
                {
                    "id": int(item["id"]),
                    "side": str(item["side"]),
                    "parent_link": parent_link,
                    "attach_requested": attached,
                    "verified": False,
                }
            )
        self._env.step()

    def _apply_foot_target(
        self,
        index: Optional[int],
        position: Any,
        rotation: Any,
    ) -> None:
        if index is None:
            return
        if self.human is None:
            raise RuntimeError("foot IK was requested but no human is configured")
        self.human.HumanIKTargetDoMove(
            index=index,
            position=list(position),
            duration=1,
            speed_based=False,
            relative=False,
        )
        self.human.HumanIKTargetDoRotate(
            index=index,
            rotation=list(rotation),
            duration=1,
            speed_based=False,
            relative=True,
        )
        self.human.HumanIKTargetDoComplete(index)
        self._env.step()

    def _request_scene_geometry(self):
        self.sock_cloth.request_scene_geometry()
        self._env.step()
        return self.sock_cloth.scene_geometry()

    def _request_grasp_target_positions(self) -> tuple[np.ndarray, np.ndarray]:
        self.sock_cloth.request_scene_geometry()
        self._env.step()
        geometry = self.sock_cloth.data.get("scene_geometry", {})
        left = np.asarray(geometry.get("left_grasp_position", ()), dtype=float)
        right = np.asarray(geometry.get("right_grasp_position", ()), dtype=float)
        if (
            left.shape != (3,)
            or right.shape != (3,)
            or not np.all(np.isfinite(left))
            or not np.all(np.isfinite(right))
        ):
            raise RuntimeError("Player did not return finite grasp target positions")
        return left, right

    def frame_recording_camera_on_opening(
        self,
        distance_m: float,
        lateral_m: float = 0.0,
        reverse: bool = False,
    ) -> Dict[str, Any]:
        """Aim the overview camera along the measured physical opening normal."""
        if self.recording_camera is None:
            raise RuntimeError("recording camera is unavailable")
        distance = float(distance_m)
        lateral = float(lateral_m)
        if (
            not np.isfinite(distance)
            or distance <= 0
            or not np.isfinite(lateral)
        ):
            raise ValueError("recording camera offset must be finite and distance positive")
        geometry = self._request_scene_geometry()
        center = np.asarray(geometry.opening_center, dtype=float)
        opening_normal = np.asarray(geometry.opening_normal, dtype=float)
        opening_normal /= np.linalg.norm(opening_normal)
        opening_axis = (
            np.asarray(geometry.right_grasp_position, dtype=float)
            - np.asarray(geometry.left_grasp_position, dtype=float)
        )
        opening_axis /= np.linalg.norm(opening_axis)
        normal_offset = opening_normal * distance * (1.0 if reverse else -1.0)
        position = center + normal_offset + opening_axis * lateral
        forward = center - position
        forward /= np.linalg.norm(forward)
        pitch = -np.degrees(np.arcsin(np.clip(forward[1], -1.0, 1.0)))
        yaw = np.degrees(np.arctan2(forward[0], forward[2]))
        rotation = np.asarray([pitch, yaw, 0.0], dtype=float)
        self.recording_camera.SetTransform(
            position=position.tolist(),
            rotation=rotation.tolist(),
        )
        self._env.step()
        return {
            "position": position.tolist(),
            "rotation": rotation.tolist(),
            "opening_center": center.tolist(),
            "opening_normal": opening_normal.tolist(),
            "distance_m": distance,
            "lateral_m": lateral,
            "reverse": bool(reverse),
        }

    def _drive_joint_target(
        self, target: np.ndarray, *, maximum_steps: int = 64
    ) -> np.ndarray:
        from .joints import JointMap

        mapping = JointMap.from_config(self.config)
        target = mapping.bound(np.asarray(target, dtype=float))
        current = np.asarray(self.robot.data.get("joint_positions", []), dtype=float)
        if current.ndim != 1 or current.size <= int(mapping.simulator_indices.max()):
            raise RuntimeError("RCareWorld full joint state is unavailable")
        simulator_target = mapping.merge_for_simulator(target, current)
        self.robot.SetJointPositionDirectly(simulator_target.tolist())
        self._env.step()
        reached = self.robot_signals()["angle"]
        if not np.all(np.isfinite(reached)):
            raise RuntimeError("robot joint target produced non-finite feedback")
        return reached

    def align_grippers_to_opening(self) -> Dict[str, Any]:
        """Numerically align the real gripper frames with opposite opening edges."""
        geometry = self._request_scene_geometry()
        return self.move_grippers_to_targets(
            geometry.left_opening_edge, geometry.right_opening_edge
        )

    def move_grippers_to_targets(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
    ) -> Dict[str, Any]:
        """Move both arm chains to Cartesian targets using live finite differences."""
        settings = self.config["scene"].get("grasp_alignment", {})
        tolerance = float(settings.get("tolerance_m", 0.015))
        epsilon = float(settings.get("finite_difference_rad", 0.02))
        damping = float(settings.get("damping", 0.002))
        maximum_joint_step = float(settings.get("maximum_joint_step_rad", 0.06))
        maximum_iterations = int(settings.get("maximum_iterations", 8))
        if min(tolerance, epsilon, damping, maximum_joint_step) <= 0:
            raise ValueError("grasp alignment parameters must be positive")
        if maximum_iterations < 1:
            raise ValueError("grasp alignment maximum_iterations must be positive")

        side_indices = {
            "left": np.arange(0, 7, dtype=int),
            "right": np.arange(9, 16, dtype=int),
        }
        from .joints import JointMap

        joint_map = JointMap.from_config(self.config)
        targets = {
            "left": np.asarray(left_target, dtype=float),
            "right": np.asarray(right_target, dtype=float),
        }
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in targets.values()):
            raise ValueError("gripper targets must be finite world 3-vectors")
        trace = []
        for iteration in range(maximum_iterations + 1):
            left_position, right_position = self._request_grasp_target_positions()
            actual = {
                "left": left_position,
                "right": right_position,
            }
            errors = {
                side: float(np.linalg.norm(targets[side] - actual[side]))
                for side in side_indices
            }
            trace.append({"iteration": iteration, "errors_m": errors})
            if max(errors.values()) <= tolerance:
                return {
                    "ok": True,
                    "tolerance_m": tolerance,
                    "iterations": iteration,
                    "errors_m": errors,
                    "trace": trace,
                }
            if iteration == maximum_iterations:
                break

            baseline = self.robot_signals()["angle"].copy()
            deltas = {}
            for side, indices in side_indices.items():
                jacobian = np.zeros((3, len(indices)), dtype=float)
                for column, joint_index in enumerate(indices):
                    restored = self._drive_joint_target(baseline)
                    baseline_positions = self._request_grasp_target_positions()
                    baseline_position = baseline_positions[0 if side == "left" else 1]
                    probe = baseline.copy()
                    direction = (
                        1.0
                        if baseline[joint_index] + epsilon
                        <= joint_map.upper[joint_index]
                        else -1.0
                    )
                    probe[joint_index] += direction * epsilon
                    reached = self._drive_joint_target(probe)
                    moved_positions = self._request_grasp_target_positions()
                    moved_position = moved_positions[0 if side == "left" else 1]
                    joint_delta = reached[joint_index] - restored[joint_index]
                    if abs(joint_delta) < 1e-6:
                        self._drive_joint_target(baseline)
                        continue
                    jacobian[:, column] = (
                        moved_position - baseline_position
                    ) / joint_delta
                self._drive_joint_target(baseline)
                side_positions = self._request_grasp_target_positions()
                side_actual = side_positions[0 if side == "left" else 1]
                error = targets[side] - side_actual
                normal = jacobian @ jacobian.T + damping * np.eye(3)
                delta = jacobian.T @ np.linalg.solve(normal, error)
                deltas[side] = np.clip(
                    delta, -maximum_joint_step, maximum_joint_step
                )

            best = baseline
            best_score = (max(errors.values()), sum(errors.values()))
            for scale in (1.0, 0.5, 0.25, -0.25):
                candidate = baseline.copy()
                for side, indices in side_indices.items():
                    candidate[indices] += deltas[side] * scale
                reached = self._drive_joint_target(candidate)
                measured = self._request_grasp_target_positions()
                candidate_errors = {
                    side: float(
                        np.linalg.norm(
                            targets[side]
                            - measured[0 if side == "left" else 1]
                        )
                    )
                    for side in side_indices
                }
                candidate_score = (
                    max(candidate_errors.values()),
                    sum(candidate_errors.values()),
                )
                if candidate_score < best_score:
                    best_score = candidate_score
                    best = reached.copy()
                self._drive_joint_target(baseline)
            self._drive_joint_target(best)

        geometry = self._request_scene_geometry()
        actual = {
            "left": np.asarray(geometry.left_grasp_position),
            "right": np.asarray(geometry.right_grasp_position),
        }
        errors = {
            side: float(np.linalg.norm(targets[side] - actual[side]))
            for side in side_indices
        }
        return {
            "ok": False,
            "tolerance_m": tolerance,
            "iterations": maximum_iterations,
            "errors_m": errors,
            "trace": trace,
        }

    def apply_scenario(self, scenario: Any) -> Dict[str, Any]:
        """Apply one validated scenario before the first recorded observation."""
        if self.robot is None or self.cloth is None:
            raise RuntimeError("robot and cloth must be loaded before applying a scenario")
        if self.config["rcareworld"].get("profile") == "custom_player":
            self.cloth.reset()
            self.cloth.SetTransform(
                position=list(scenario.sock_position),
                rotation=list(scenario.sock_rotation),
            )
            self._env.step()
            native = bool(
                self.config["rcareworld"].get("native_rcareworld", False)
            )
            if native:
                self._apply_foot_target(
                    scenario.support_foot_ik_index,
                    scenario.support_foot_position,
                    scenario.support_foot_rotation,
                )
                self._apply_foot_target(
                    scenario.foot_ik_index,
                    scenario.foot_position,
                    scenario.foot_rotation,
                )
            current = self.robot_signals()["angle"]
            if self.config["scene"].get("robot_direct_joint_control", False):
                applied = self._drive_joint_target(
                    np.asarray(scenario.initial_joints, dtype=float)
                )
                for _ in range(
                    int(
                        self.config["scene"]
                        .get("grasp_alignment", {})
                        .get("joint_settle_steps", 3)
                    )
                ):
                    applied = self._drive_joint_target(applied)
            else:
                applied = self.command(
                    np.asarray(scenario.initial_joints, dtype=float), current
                )
            pose_settings = self.config["scene"].get("initial_pose_contract", {})
            initial_grasp = []
            grasp_alignment = None
            grasp_settings = self.config["scene"].get("grasp_anchors", {})
            grasp_distance = float(grasp_settings.get("max_distance_m", 0.03))
            if grasp_settings.get("auto_grasp", False):
                alignment_settings = self.config["scene"].get(
                    "grasp_alignment", {}
                )
                maximum_anchor_span = grasp_settings.get(
                    "maximum_anchor_span_m"
                )
                if maximum_anchor_span is not None:
                    self.sock_cloth.clamp_grasp_target_span(
                        float(maximum_anchor_span)
                    )
                    self._env.step()
                if alignment_settings.get("enabled", False):
                    target_span = float(
                        alignment_settings.get("target_span_m", 0.09)
                    )
                    left, right = self._request_grasp_target_positions()
                    center = 0.5 * (left + right)
                    axis = right - left
                    span = float(np.linalg.norm(axis))
                    if span <= 1e-8 or target_span <= 0:
                        raise RuntimeError("invalid initial gripper span")
                    axis /= span
                    grasp_alignment = self.move_grippers_to_targets(
                        center - axis * target_span * 0.5,
                        center + axis * target_span * 0.5,
                    )
                    if not grasp_alignment["ok"]:
                        raise RuntimeError(
                            "real gripper frames could not reach safe opening span: "
                            f"{grasp_alignment}"
                        )
                    # Settling re-applies `applied`; retain the aligned arm pose
                    # instead of restoring the original wide bimanual span.
                    applied = self.robot_signals()["angle"].copy()
                if (
                    not grasp_settings.get("align_sock_to_grippers", False)
                    and not alignment_settings.get("enabled", False)
                ):
                    self.sock_cloth.align_grasp_targets_to_opening()
                    self._env.step()
                if not grasp_settings.get("align_sock_to_grippers", False):
                    initial_grasp = [
                        self.grasp("left", grasp_distance),
                        self.grasp("right", grasp_distance),
                    ]
            for _ in range(scenario.settle_steps):
                if self.config["scene"].get(
                    "robot_direct_joint_control", False
                ):
                    self.command(applied, applied)
                else:
                    self._env.step()
            if (
                native
                and pose_settings
                and self.config["scene"].get("visuals", {}).get(
                    "human_proxy", False
                )
            ):
                self.sock_cloth.align_human_visual_foot_to_sock(
                    float(pose_settings["foot_to_sock_m"])
                )
                for _ in range(min(scenario.settle_steps, 30)):
                    self._env.step()
                self.sock_cloth.align_human_visual_foot_to_sock(
                    float(pose_settings["foot_to_sock_m"])
                )
                self._env.step()
            if (
                native
                and pose_settings.get("calibrate_foot_to_sock", False)
            ):
                self._calibrate_foot_to_sock(
                    scenario.foot_ik_index,
                    float(pose_settings["foot_to_sock_m"]),
                )
                if scenario.foot_ik_index is None:
                    self.sock_cloth.stop_foot_clearance_tracking()
                    self._env.step()
            self._right_toe_offset_report = (
                self._apply_post_calibration_right_toe_offset()
            )
            if (
                grasp_settings.get("auto_grasp", False)
                and grasp_settings.get("align_sock_to_grippers", False)
            ):
                if grasp_settings.get("align_sock_to_gripper_plate", False):
                    self.sock_cloth.align_sock_opening_to_grasp_plate_and_grasp(
                        grasp_distance
                    )
                else:
                    final_geometry = self._request_scene_geometry()
                    self.sock_cloth.align_sock_opening_to_grasp_targets_and_grasp(
                        list(final_geometry.right_toe_position),
                        grasp_distance,
                    )
                self.sock_cloth.request_grasp_state()
                self._env.step()
                states = {
                    state.side: state
                    for state in self.sock_cloth.grasp_states()
                }
                initial_grasp = [
                    {
                        "side": state.side,
                        "attached": state.attached,
                        "particle_indices": list(state.particle_indices),
                        "constraint_error": state.constraint_error,
                        "peak_constraint_error": state.peak_constraint_error,
                        "over_threshold_steps": state.over_threshold_steps,
                        "release_reason": state.release_reason,
                    }
                    for side in ("left", "right")
                    for state in (states[side],)
                ]
            # Rotating the opening frame changes the plane used by foot
            # clearance. Recalibrate once against that final plane, then freeze
            # the human/chair pose for the rollout. Translation along the new
            # normal preserves the toe-facing grasp orientation.
            if (
                native
                and pose_settings.get("calibrate_foot_to_sock", False)
                and grasp_settings.get("align_sock_to_grippers", False)
                and pose_settings.get("vertical_toe_drop_m") is None
            ):
                if scenario.foot_ik_index is None:
                    self._calibrate_foot_to_sock(
                        None,
                        float(pose_settings["foot_to_sock_m"]),
                        3,
                    )
                    self.sock_cloth.stop_foot_clearance_tracking()
                    self._env.step()
                else:
                    self._calibrate_foot_to_sock(
                        scenario.foot_ik_index,
                        float(pose_settings["foot_to_sock_m"]),
                    )
            locked_pose_baseline = pose_settings.get("locked_pose_baseline")
            if native and locked_pose_baseline:
                effective_baseline = {
                    name: list(value)
                    for name, value in locked_pose_baseline.items()
                }
                if pose_settings.get("vertical_toe_drop_m") is not None:
                    final_geometry = self._request_scene_geometry()
                    (
                        effective_baseline,
                        self._human_chair_translation_report,
                    ) = self._translate_locked_pose_to_opening_normal(
                        locked_pose_baseline,
                        final_geometry,
                    )
                self._effective_locked_pose_baseline = effective_baseline
                chair_id = int(
                    self.config["scene"]
                    .get("stable_ids", {})
                    .get("chair", -1)
                )
                self.sock_cloth.restore_human_chair_world_pose(
                    effective_baseline["human_root_position"],
                    effective_baseline["chair_position"],
                    effective_baseline["human_anchor_position"],
                    effective_baseline["right_toe_position"],
                    chair_id,
                )
                if not pose_settings.get("lock_human_and_chair", False):
                    self._env.step()
            if (
                native
                and pose_settings.get("lock_human_and_chair", False)
            ):
                self.sock_cloth.lock_human_and_chair(
                    int(
                        self.config["scene"]
                        .get("stable_ids", {})
                        .get("chair", -1)
                    )
                )
                self._env.step()
            if self._human_chair_translation_report is not None:
                geometry = self._request_scene_geometry()
                actual = np.asarray(geometry.right_toe_position, dtype=float)
                target = np.asarray(
                    self._human_chair_translation_report[
                        "target_right_toe_position"
                    ],
                    dtype=float,
                )
                tolerance = float(
                    pose_settings.get("vertical_toe_drop_tolerance_m", 0.005)
                )
                target_error = float(np.linalg.norm(actual - target))
                self._human_chair_translation_report.update(
                    {
                        "actual_right_toe_position": actual.tolist(),
                        "target_error_m": target_error,
                        "opening_to_toe_alignment": (
                            geometry.opening_to_toe_alignment
                        ),
                        "ok": (
                            target_error <= tolerance
                            and geometry.opening_to_toe_alignment
                            >= float(
                                pose_settings.get(
                                    "opening_to_toe_alignment_min", 0.9
                                )
                            )
                        ),
                    }
                )
            if initial_grasp and not all(
                item["attached"] for item in initial_grasp
            ):
                raise RuntimeError(
                    f"initial bimanual sock grasp failed: {initial_grasp}"
                )
            if initial_grasp:
                by_side = {item["side"]: item for item in initial_grasp}
                for item in self._grasp_attachment_report:
                    state = by_side.get(item["side"], {})
                    item["verified"] = bool(state.get("attached", False))
                    item["particle_indices"] = list(
                        state.get("particle_indices", ())
                    )
            pose_contract = self._request_initial_pose_contract() if native else {}
            if (
                pose_contract
                and not pose_contract["ok"]
                and pose_settings.get("enforce", True)
            ):
                raise RuntimeError(
                    f"initial human/sock pose contract failed: {pose_contract}"
                )
            if initial_grasp:
                self.sock_cloth.arm_slip_detection(True)
                self.sock_cloth.request_configuration()
                self._env.step()
            self._rollout_started = True
            return {
                "initial_command_steps": int(not np.allclose(current, applied)),
                "settle_steps": scenario.settle_steps,
                "grasp_alignment": grasp_alignment,
                "foot_ik_applied": native and scenario.foot_ik_index is not None,
                "support_foot_ik_applied": (
                    native and scenario.support_foot_ik_index is not None
                ),
                "foot_pose_source": (
                    "RCareCommon HumanbodyAttr"
                    if native
                    else "authored greenfield scene"
                ),
                "chair_part_ids": [int(part.id) for part in self.chair],
                "grasp_attachments": list(self._grasp_attachment_report),
                "initial_grasp": initial_grasp,
                "right_toe_offset": self._right_toe_offset_report,
                "human_chair_translation": (
                    self._human_chair_translation_report
                ),
                "initial_pose_contract": pose_contract,
            }
        self.cloth.SetTransform(
            position=list(scenario.sock_position),
            rotation=list(scenario.sock_rotation),
        )
        self._env.step()
        self._create_grasp_anchors()
        self._apply_foot_target(
            scenario.support_foot_ik_index,
            scenario.support_foot_position,
            scenario.support_foot_rotation,
        )
        self._apply_foot_target(
            scenario.foot_ik_index,
            scenario.foot_position,
            scenario.foot_rotation,
        )

        target = np.asarray(scenario.initial_joints, dtype=float)
        current = self.robot_signals()["angle"]
        command_steps = 0
        while not np.allclose(current, target, atol=1e-6, rtol=0):
            applied = self.command(target, current)
            command_steps += 1
            current = applied
            if command_steps > 1000:
                raise RuntimeError("initial joint pose did not converge within 1000 commands")
        for _ in range(scenario.settle_steps):
            self._env.step()
        return {
            "initial_command_steps": command_steps,
            "settle_steps": scenario.settle_steps,
            "foot_ik_applied": scenario.foot_ik_index is not None,
            "support_foot_ik_applied": scenario.support_foot_ik_index is not None,
            "chair_part_ids": [int(part.id) for part in self.chair],
            "grasp_attachments": list(self._grasp_attachment_report),
        }

    def _validate_simulator_joint_mapping(self) -> None:
        from .joints import JointMap

        mapping = JointMap.from_config(self.config)
        data = getattr(self.robot, "data", {}) or {}
        names = data.get("names", [])
        types = data.get("types", [])
        simulator_names = [
            name
            for name, joint_type in zip(names, types)
            if joint_type != "FixedJoint" and name not in mapping.mimic
        ]
        joint_positions = data.get("joint_positions", [])
        if len(simulator_names) != len(joint_positions):
            raise RuntimeError(
                "cannot align RCareWorld joint names with its movable joint vector: "
                f"names={len(simulator_names)}, positions={len(joint_positions)}"
            )
        configured_indices = mapping.simulator_indices.tolist()
        configured_indices.extend(mapping.fixed_simulator_indices.tolist())
        max_index = max(configured_indices)
        if not simulator_names or max_index >= len(simulator_names):
            raise RuntimeError(
                "configured simulator joint index exceeds RCareWorld movable joints: "
                f"max_index={max_index}, "
                f"joint_count={len(simulator_names)}"
            )
        expected_names = []
        for name in mapping.names:
            rule = mapping.mimic.get(name)
            expected_names.append(rule["source"] if rule else name)
        selected_names = [
            simulator_names[int(index)] for index in mapping.simulator_indices
        ]
        if selected_names != expected_names:
            mismatches = [
                {
                    "action": mapping.names[index],
                    "expected": expected_names[index],
                    "actual": selected_names[index],
                    "simulator_index": int(mapping.simulator_indices[index]),
                }
                for index in range(len(mapping.names))
                if selected_names[index] != expected_names[index]
            ]
            raise RuntimeError(f"RCareWorld joint mapping mismatch: {mismatches}")
        selected_fixed_names = [
            simulator_names[int(index)] for index in mapping.fixed_simulator_indices
        ]
        if selected_fixed_names != list(mapping.fixed_names):
            mismatches = [
                {
                    "fixed": mapping.fixed_names[index],
                    "actual": selected_fixed_names[index],
                    "simulator_index": int(mapping.fixed_simulator_indices[index]),
                }
                for index in range(len(mapping.fixed_names))
                if selected_fixed_names[index] != mapping.fixed_names[index]
            ]
            raise RuntimeError(
                f"RCareWorld fixed joint mapping mismatch: {mismatches}"
            )

    def diagnostics(self) -> Dict[str, Any]:
        scene = self.config["scene"]
        if self.config["rcareworld"].get("profile") == "custom_player":
            from .sock_cloth import validate_configuration

            configuration = self.sock_cloth.configuration() if self.sock_cloth else {}
            colliders = (
                list(self.sock_cloth.registered_colliders()) if self.sock_cloth else []
            )
            grasp_state = list(
                getattr(self.cloth, "data", {}).get("grasp_state", [])
            )
            return {
                "human_configured": self.human is not None,
                "camera_configured": (
                    self.camera is not None
                    or bool(getattr(self.cloth, "data", {}).get("rgb_png"))
                ),
                "camera_mount": dict(self._camera_mount_report),
                "exact_foot_colliders_configured": bool(
                    scene.get("human_foot_collider_ids")
                ),
                "contact_force_available": (
                    "cloth_contacts" in getattr(self.cloth, "data", {})
                ),
                "robot_obi_collider_verified": any(
                    int(item.get("object_id", -1))
                    == int(self.config["assets"]["robot_id"])
                    and item.get("enabled")
                    for item in colliders
                ),
                "contact_proxy_ids": list(scene.get("contact_proxy_ids", [])),
                "registered_obi_colliders": colliders,
                "grasp_state": grasp_state,
                "grasp_attachments": [
                    {
                        "side": str(item.get("side", "")),
                        "verified": bool(item.get("attached", False)),
                        "particle_indices": list(item.get("particle_indices", ())),
                        "release_reason": str(item.get("release_reason", "")),
                    }
                    for item in grasp_state
                ],
                "scene_geometry": dict(
                    getattr(self.cloth, "data", {}).get("scene_geometry", {})
                ),
                "robot_human_rigid_collision_qa": {
                    "enabled_pair_count": int(
                        configuration.get(
                            "robot_human_rigid_enabled_pair_count", 0
                        )
                    ),
                    "ignored_pair_count": int(
                        configuration.get(
                            "robot_human_rigid_ignored_pair_count", 0
                        )
                    ),
                    "maximum_penetration_m": float(
                        configuration.get(
                            "robot_human_maximum_penetration_m", float("inf")
                        )
                    ),
                },
                "visual_diagnostics": list(
                    getattr(self.cloth, "data", {}).get("visual_diagnostics", [])
                ),
                "external_torque_source": "joint_force residual proxy",
                "obi_contract": validate_configuration(
                    configuration, self.config.get("obi", {}).get("expected", {})
                ),
                "limitations": [] if configuration.get("obi_available") else [
                    "Licensed Obi runtime is unavailable; physical results are invalid."
                ],
            }
        return {
            "human_configured": scene.get("human_id") is not None,
            "camera_configured": scene.get("camera_id") is not None,
            "camera_mount": dict(self._camera_mount_report),
            "exact_foot_colliders_configured": bool(scene.get("human_foot_collider_ids")),
            "contact_force_available": False,
            "robot_obi_collider_verified": False,
            "contact_proxy_ids": list(scene.get("contact_proxy_ids", [])),
            "chair_obi_collider_ids": [int(part.id) for part in self.chair],
            "grasp_attachments": list(self._grasp_attachment_report),
            "external_torque_source": "joint_force residual proxy",
            "obi_contract": dict(self.config.get("obi", {})),
            "limitations": [
                "HumanBodyIK exposes target indices (feet are 2 and 3), not exact foot collider IDs.",
                "The pinned Python API exposes collision pairs/robot efforts but no direct contact-force vectors.",
                "Robot AddObiCollider is requested but cannot be verified without the Unity project/player registration.",
                "Requested stretch/bend compliance cannot be applied or verified through this Player's Python API.",
            ],
        }

    def observe(self) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("environment is not connected")
        if self.config["rcareworld"].get("profile") == "custom_player":
            if self.config["rcareworld"].get("native_rcareworld", False):
                return self._observe_native_custom_player()
            return self._observe_custom_player()
        observation = {
            "robot": dict(getattr(self.robot, "data", {}) or {}),
            "human": dict(getattr(self.human, "data", {}) or {}),
            "cloth": dict(getattr(self.cloth, "data", {}) or {}),
            "contact_force": None,
            "diagnostics": self.diagnostics(),
        }
        if self.cloth is not None:
            self.cloth.GetParticles()
            self._env.step()
            observation["cloth"] = dict(getattr(self.cloth, "data", {}) or {})
        if self.camera is not None:
            observation["camera"] = self._capture_camera()
        else:
            observation["camera"] = None
        if self.recording_camera is not None:
            observation["recording_camera"] = self._capture_recording_camera()
        else:
            observation["recording_camera"] = None
        self._env.GetCurrentCollisionPairs()
        self._env.step()
        observation["collision_pairs"] = list(
            self._env.data.get("CurrentCollisionPairs", [])
        )
        observation.update(self.robot_signals())
        return observation

    def _observe_native_custom_player(self) -> Dict[str, Any]:
        self.sock_cloth.request_particles()
        self.sock_cloth.request_particle_velocities()
        self.sock_cloth.request_configuration()
        self.sock_cloth.request_registered_colliders()
        self.sock_cloth.request_grasp_state()
        self.sock_cloth.request_scene_geometry()
        self.sock_cloth.request_dressing_qa()
        self.sock_cloth.request_visual_diagnostics(
            int(self.config["scene"].get("stable_ids", {}).get("chair", -1))
        )
        self.sock_cloth.request_contacts()
        if hasattr(self.robot, "GetJointInverseDynamicsForce"):
            self.robot.GetJointInverseDynamicsForce()
        self._env.step()
        cloth = dict(self.sock_cloth.data)
        robot_data = dict(getattr(self.robot, "data", {}) or {})
        contacts = list(cloth.get("cloth_contacts", []))
        camera = self._capture_camera() if self.camera is not None else None
        recording = (
            self._capture_recording_camera()
            if self.recording_camera is not None
            else None
        )
        collision_pairs = sorted(
            {
                (int(self.sock_cloth.id), int(item["collider_id"]))
                for item in contacts
                if int(item.get("collider_id", -1)) >= 0
            }
        )
        visual_diagnostics = list(cloth.get("visual_diagnostics", []))
        lock_report = next(
            (
                dict(item)
                for item in visual_diagnostics
                if item.get("role") == "human_chair_lock"
            ),
            None,
        )
        pose_settings = self.config["scene"].get("initial_pose_contract", {})
        if pose_settings.get("lock_human_and_chair", False):
            maximum_drift = float(
                pose_settings.get("maximum_lock_drift_m", 0.002)
            )
            measured = (
                "right_toe_drift_m",
                "chair_drift_m",
                "human_root_drift_m",
                "human_anchor_drift_m",
            )
            if (
                lock_report is None
                or not bool(lock_report.get("valid", False))
                or any(
                    not np.isfinite(float(lock_report.get(name, float("inf"))))
                    or float(lock_report.get(name, float("inf"))) > maximum_drift
                    for name in measured
                )
            ):
                raise RuntimeError(
                    "human/chair rollout lock drift exceeded contract: "
                    f"{lock_report}"
                )
        diagnostics = self.diagnostics()
        diagnostics["human_chair_lock_qa"] = lock_report
        observation = {
            "robot": robot_data,
            "human": dict(getattr(self.human, "data", {}) or {}),
            "cloth": cloth,
            "camera": camera,
            "recording_camera": recording,
            "contact_force": contacts,
            "dressing_qa": dict(cloth.get("dressing_qa", {})),
            "collision_pairs": collision_pairs,
            "diagnostics": diagnostics,
        }
        observation.update(self.robot_signals(robot_data))
        return observation

    def _request_initial_pose_contract(self) -> Dict[str, Any]:
        settings = self.config["scene"].get("initial_pose_contract", {})
        if not settings:
            return {}
        self.sock_cloth.request_scene_geometry()
        self.sock_cloth.request_grasp_state()
        self.sock_cloth.request_particles()
        self.sock_cloth.request_configuration()
        self.sock_cloth.request_visual_diagnostics(
            int(self.config["scene"].get("stable_ids", {}).get("chair", -1))
        )
        self._env.step()
        try:
            geometry = self.sock_cloth.scene_geometry()
        except (KeyError, TypeError, ValueError) as error:
            return {"ok": False, "error": str(error)}
        wanted_distance = float(settings["foot_to_sock_m"])
        distance_tolerance = float(settings["foot_to_sock_tolerance_m"])
        lateral_tolerance = float(settings["foot_lateral_tolerance_m"])
        wanted_angle = float(settings["right_leg_raise_degrees"])
        angle_tolerance = float(settings["right_leg_raise_tolerance_degrees"])
        maximum_knee_flexion = float(
            settings.get("right_knee_flexion_max_degrees", 180.0)
        )
        minimum_gravity_alignment = float(
            settings.get("sock_body_gravity_alignment_min", 0.8)
        )
        minimum_toe_alignment = float(
            settings.get("opening_to_toe_alignment_min", 0.9)
        )
        minimum_cuff_insertion = float(
            settings.get("minimum_cuff_insertion_depth_m", 0.0)
        )
        maximum_opening_span = float(
            settings.get("maximum_opening_span_m", 0.12)
        )
        maximum_cloth_bounds_span = float(
            settings.get("maximum_cloth_bounds_span_m", 0.35)
        )
        opening_span = float(
            getattr(
                geometry,
                "opening_span_m",
                np.linalg.norm(
                    np.asarray(geometry.right_opening_edge, dtype=float)
                    - np.asarray(geometry.left_opening_edge, dtype=float)
                ),
            )
        )
        maximum_stretch = float(
            self.config["obi"]["expected"].get(
                "maximum_circumferential_stretch", 1.5
            )
        )
        cloth_qa = self.cloth_radius_qa(
            dict(self.sock_cloth.data),
            radial_segments=int(
                self.config["scenario"]["sock"]["radial_segments"]
            ),
            maximum_circumferential_stretch=maximum_stretch,
        )
        bounds = cloth_qa.get("bounds_world", {})
        bounds_minimum = np.asarray(bounds.get("minimum", ()), dtype=float)
        bounds_maximum = np.asarray(bounds.get("maximum", ()), dtype=float)
        bounds_span = (
            bounds_maximum - bounds_minimum
            if bounds_minimum.shape == (3,)
            and bounds_maximum.shape == (3,)
            else np.full(3, np.inf)
        )
        initial_cloth_ok = (
            bool(cloth_qa.get("passes", False))
            and opening_span <= maximum_opening_span
            and bool(np.all(bounds_span <= maximum_cloth_bounds_span))
        )
        configured_opening_edge_error = settings.get(
            "maximum_opening_edge_error_m"
        )
        maximum_opening_edge_error = (
            float(configured_opening_edge_error)
            if configured_opening_edge_error is not None
            else float("inf")
        )
        if maximum_opening_edge_error <= 0:
            raise ValueError(
                "maximum_opening_edge_error_m must be positive when configured"
            )
        left_opening_edge_error = float(
            np.linalg.norm(
                np.asarray(geometry.left_opening_edge, dtype=float)
                - np.asarray(geometry.left_grasp_position, dtype=float)
            )
        )
        right_opening_edge_error = float(
            np.linalg.norm(
                np.asarray(geometry.right_opening_edge, dtype=float)
                - np.asarray(geometry.right_grasp_position, dtype=float)
            )
        )
        opening_edges_at_grippers = (
            left_opening_edge_error <= maximum_opening_edge_error
            and right_opening_edge_error <= maximum_opening_edge_error
        )
        grasp_states = {
            state.side: state for state in self.sock_cloth.grasp_states()
        }
        bimanual_grasp_attached = all(
            side in grasp_states and grasp_states[side].attached
            for side in ("left", "right")
        )
        required_particles_per_side = int(
            settings.get("required_grasp_particles_per_side", 2)
        )
        grasp_particle_counts = {
            side: len(grasp_states[side].particle_indices)
            if side in grasp_states
            else 0
            for side in ("left", "right")
        }
        grasp_particle_indices = {
            side: list(grasp_states[side].particle_indices)
            if side in grasp_states
            else []
            for side in ("left", "right")
        }
        four_point_grasp_attached = all(
            grasp_particle_counts[side] == required_particles_per_side
            for side in ("left", "right")
        )
        target_patch_span = 2.0 * float(
            self.config["obi"]["expected"]["grasp_thickness_half_width_m"]
        )
        patch_span_tolerance = float(
            settings.get("grasp_patch_span_tolerance_m", 0.006)
        )
        maximum_grasp_corner_error = float(
            settings.get("maximum_grasp_corner_error_m", 0.006)
        )
        minimum_thickness_alignment = float(
            settings.get("minimum_grasp_thickness_axis_alignment", 0.95)
        )
        rectangular_opening_ok = (
            abs(geometry.left_grasp_patch_span_m - target_patch_span)
            <= patch_span_tolerance
            and abs(geometry.right_grasp_patch_span_m - target_patch_span)
            <= patch_span_tolerance
            and geometry.maximum_grasp_corner_error_m
            <= maximum_grasp_corner_error
            and geometry.grasp_thickness_axis_alignment
            >= minimum_thickness_alignment
        )
        target_opening_rectangle_area = (
            float(
                self.config["scene"]
                .get("grasp_alignment", {})
                .get("target_span_m", opening_span)
            )
            * target_patch_span
        )
        measured_opening_rectangle_area = opening_span * (
            geometry.left_grasp_patch_span_m
            + geometry.right_grasp_patch_span_m
        ) * 0.5
        expected_human_position = settings.get("expected_human_position")
        expected_chair_position = settings.get("expected_chair_position")
        configured_human_position = self.config["scene"].get("human_position")
        chair_parts = self.config["scene"].get("chair", {}).get("parts", ())
        configured_chair_position = (
            chair_parts[0].get("position") if chair_parts else None
        )
        configured_pose_baseline_ok = (
            expected_human_position is None
            or np.allclose(
                np.asarray(configured_human_position, dtype=float),
                np.asarray(expected_human_position, dtype=float),
                rtol=0,
                atol=1e-9,
            )
        ) and (
            expected_chair_position is None
            or np.allclose(
                np.asarray(configured_chair_position, dtype=float),
                np.asarray(expected_chair_position, dtype=float),
                rtol=0,
                atol=1e-9,
            )
        )
        distance_error = abs(geometry.foot_to_opening_plane_m - wanted_distance)
        angle_error = abs(geometry.right_leg_raise_degrees - wanted_angle)
        offset_report = self._right_toe_offset_report
        translation_report = self._human_chair_translation_report
        offset_ok = (
            offset_report is None or bool(offset_report.get("ok", False))
        ) and (
            translation_report is None
            or bool(translation_report.get("ok", False))
        )
        sock_alignment_ok = (
            geometry.opening_to_toe_alignment >= minimum_toe_alignment
            and (
                offset_report is not None
                or translation_report is not None
                or (
                    distance_error <= distance_tolerance
                    and geometry.foot_to_opening_lateral_m <= lateral_tolerance
                )
            )
        )
        visual_entries = self.sock_cloth.visual_diagnostics()
        task_pose_visual = next(
            (
                dict(item)
                for item in visual_entries
                if item.get("role") == "human_task_pose"
            ),
            {},
        )
        visual_ok = bool(task_pose_visual.get("valid", False))
        require_lock = bool(settings.get("lock_human_and_chair", False))
        lock_visual = next(
            (
                dict(item)
                for item in visual_entries
                if item.get("role") == "human_chair_lock"
            ),
            {},
        )
        maximum_lock_drift = float(settings.get("maximum_lock_drift_m", 0.002))
        lock_ok = (
            not require_lock
            or (
                bool(lock_visual.get("valid", False))
                and bool(lock_visual.get("locked", False))
                and float(lock_visual.get("right_toe_drift_m", float("inf")))
                <= maximum_lock_drift
                and float(lock_visual.get("chair_drift_m", float("inf")))
                <= maximum_lock_drift
                and float(lock_visual.get("human_root_drift_m", float("inf")))
                <= maximum_lock_drift
                and float(lock_visual.get("human_anchor_drift_m", float("inf")))
                <= maximum_lock_drift
            )
        )
        locked_pose_baseline = (
            self._effective_locked_pose_baseline
            or settings.get("locked_pose_baseline")
        )
        locked_pose_tolerance = float(
            settings.get("locked_pose_tolerance_m", 0.002)
        )
        locked_pose_errors = {}
        if locked_pose_baseline:
            for expected_name, measured_name in (
                ("human_root_position", "human_root_position"),
                ("chair_position", "chair_position"),
                ("human_anchor_position", "human_anchor_position"),
                ("right_toe_position", "right_toe_position"),
            ):
                expected_value = np.asarray(
                    locked_pose_baseline.get(expected_name, ()), dtype=float
                )
                measured_value = np.asarray(
                    lock_visual.get(measured_name, ()), dtype=float
                )
                locked_pose_errors[expected_name] = (
                    float(np.linalg.norm(measured_value - expected_value))
                    if expected_value.shape == (3,)
                    and measured_value.shape == (3,)
                    else float("inf")
                )
        locked_pose_baseline_ok = (
            not locked_pose_baseline
            or (
                bool(lock_visual.get("valid", False))
                and all(
                    np.isfinite(value) and value <= locked_pose_tolerance
                    for value in locked_pose_errors.values()
                )
            )
        )
        return {
            "ok": (
                sock_alignment_ok
                and offset_ok
                and angle_error <= angle_tolerance
                and geometry.right_knee_flexion_degrees <= maximum_knee_flexion
                and geometry.sock_body_gravity_alignment
                >= minimum_gravity_alignment
                and geometry.left_cuff_insertion_depth_m
                >= minimum_cuff_insertion
                and geometry.right_cuff_insertion_depth_m
                >= minimum_cuff_insertion
                and opening_edges_at_grippers
                and bimanual_grasp_attached
                and four_point_grasp_attached
                and rectangular_opening_ok
                and initial_cloth_ok
                and visual_ok
                and lock_ok
                and configured_pose_baseline_ok
                and locked_pose_baseline_ok
            ),
            "foot_to_sock_m": geometry.foot_to_opening_plane_m,
            "foot_to_sock_target_m": wanted_distance,
            "foot_to_sock_tolerance_m": distance_tolerance,
            "foot_lateral_offset_m": geometry.foot_to_opening_lateral_m,
            "foot_lateral_tolerance_m": lateral_tolerance,
            "right_leg_raise_degrees": geometry.right_leg_raise_degrees,
            "right_leg_raise_target_degrees": wanted_angle,
            "right_leg_raise_tolerance_degrees": angle_tolerance,
            "right_knee_flexion_degrees": geometry.right_knee_flexion_degrees,
            "right_knee_flexion_max_degrees": maximum_knee_flexion,
            "opening_center": list(geometry.opening_center),
            "opening_normal": list(geometry.opening_normal),
            "opening_outward_normal": list(geometry.opening_outward_normal),
            "opening_target_normal": list(geometry.opening_target_normal),
            "opening_to_toe_alignment": geometry.opening_to_toe_alignment,
            "opening_to_toe_alignment_min": minimum_toe_alignment,
            "opening_span_m": opening_span,
            "maximum_opening_span_m": maximum_opening_span,
            "cloth_bounds_span_m": bounds_span.tolist(),
            "maximum_cloth_bounds_span_m": maximum_cloth_bounds_span,
            "initial_cloth_qa": cloth_qa,
            "initial_cloth_ok": initial_cloth_ok,
            "left_cuff_insertion_depth_m": geometry.left_cuff_insertion_depth_m,
            "right_cuff_insertion_depth_m": geometry.right_cuff_insertion_depth_m,
            "minimum_cuff_insertion_depth_m": minimum_cuff_insertion,
            "sock_body_direction": list(geometry.sock_body_direction),
            "sock_body_gravity_alignment": geometry.sock_body_gravity_alignment,
            "sock_body_gravity_alignment_min": minimum_gravity_alignment,
            "right_toe_position": list(geometry.right_toe_position),
            "left_grasp_position": list(geometry.left_grasp_position),
            "right_grasp_position": list(geometry.right_grasp_position),
            "left_opening_edge": list(geometry.left_opening_edge),
            "right_opening_edge": list(geometry.right_opening_edge),
            "left_opening_edge_error_m": left_opening_edge_error,
            "right_opening_edge_error_m": right_opening_edge_error,
            "maximum_opening_edge_error_m": (
                maximum_opening_edge_error
                if configured_opening_edge_error is not None
                else None
            ),
            "opening_edges_at_grippers": opening_edges_at_grippers,
            "bimanual_grasp_attached": bimanual_grasp_attached,
            "grasp_particle_counts": grasp_particle_counts,
            "grasp_particle_indices": grasp_particle_indices,
            "required_grasp_particles_per_side": required_particles_per_side,
            "four_point_grasp_attached": four_point_grasp_attached,
            "left_grasp_thickness_axis": list(
                geometry.left_grasp_thickness_axis
            ),
            "right_grasp_thickness_axis": list(
                geometry.right_grasp_thickness_axis
            ),
            "left_grasp_inward_axis": list(geometry.left_grasp_inward_axis),
            "right_grasp_inward_axis": list(geometry.right_grasp_inward_axis),
            "left_grasp_corner_negative": list(
                geometry.left_grasp_corner_negative
            ),
            "left_grasp_corner_positive": list(
                geometry.left_grasp_corner_positive
            ),
            "right_grasp_corner_negative": list(
                geometry.right_grasp_corner_negative
            ),
            "right_grasp_corner_positive": list(
                geometry.right_grasp_corner_positive
            ),
            "left_grasp_patch_span_m": geometry.left_grasp_patch_span_m,
            "right_grasp_patch_span_m": geometry.right_grasp_patch_span_m,
            "target_grasp_patch_span_m": target_patch_span,
            "grasp_patch_span_tolerance_m": patch_span_tolerance,
            "maximum_grasp_corner_error_m": (
                geometry.maximum_grasp_corner_error_m
            ),
            "maximum_grasp_corner_error_limit_m": maximum_grasp_corner_error,
            "grasp_thickness_axis_alignment": (
                geometry.grasp_thickness_axis_alignment
            ),
            "minimum_grasp_thickness_axis_alignment": (
                minimum_thickness_alignment
            ),
            "rectangular_opening_ok": rectangular_opening_ok,
            "target_opening_rectangle_area_m2": target_opening_rectangle_area,
            "measured_opening_rectangle_area_m2": (
                measured_opening_rectangle_area
            ),
            "configured_pose_baseline_ok": configured_pose_baseline_ok,
            "expected_human_position": expected_human_position,
            "configured_human_position": configured_human_position,
            "expected_chair_position": expected_chair_position,
            "configured_chair_position": configured_chair_position,
            "right_toe_offset": offset_report,
            "human_chair_translation": translation_report,
            "human_visual_ok": visual_ok,
            "human_visual_diagnostics": task_pose_visual,
            "human_chair_lock_ok": lock_ok,
            "human_chair_lock_diagnostics": lock_visual,
            "maximum_lock_drift_m": maximum_lock_drift,
            "locked_pose_baseline": locked_pose_baseline,
            "locked_pose_tolerance_m": locked_pose_tolerance,
            "locked_pose_errors_m": locked_pose_errors,
            "locked_pose_baseline_ok": locked_pose_baseline_ok,
        }

    def _calibrate_foot_to_sock(
        self,
        foot_ik_index: Optional[int],
        target_distance_m: float,
        iterations: int = 10,
    ) -> None:
        if target_distance_m <= 0:
            raise ValueError("right foot calibration requires a positive target distance")
        if foot_ik_index is None and self.sock_cloth is None:
            raise ValueError("right foot calibration requires IK or the custom player")
        if foot_ik_index is None:
            chair_id = int(
                self.config["scene"].get("stable_ids", {}).get("chair", -1)
            )
            self.sock_cloth.set_foot_clearance_target(
                target_distance_m, chair_id
            )
        for _ in range(iterations):
            self.sock_cloth.request_scene_geometry()
            self._env.step()
            geometry = self.sock_cloth.scene_geometry()
            if (
                abs(geometry.foot_to_opening_plane_m - target_distance_m) <= 0.002
                and geometry.foot_to_opening_lateral_m <= 0.002
            ):
                break
            opening = (
                np.asarray(geometry.left_grasp_position, dtype=float)
                + np.asarray(geometry.right_grasp_position, dtype=float)
            ) * 0.5
            normal = np.asarray(geometry.opening_target_normal, dtype=float)
            toe = np.asarray(geometry.right_toe_position, dtype=float)
            desired = opening - normal * target_distance_m
            if foot_ik_index is None:
                continue
            else:
                self.sock_cloth.translate_human_and_ik((desired - toe).tolist())
            self._env.step()
    def _apply_post_calibration_right_toe_offset(
        self,
    ) -> Optional[Dict[str, Any]]:
        settings = self.config["scene"].get("initial_pose_contract", {})
        configured = settings.get("right_toe_offset_world_m")
        if configured is None:
            return None
        offset = np.asarray(configured, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError(
                "right_toe_offset_world_m must contain three finite values"
            )
        tolerance = float(settings.get("right_toe_offset_tolerance_m", 0.005))
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(
                "right_toe_offset_tolerance_m must be finite and positive"
            )
        before = self._request_scene_geometry()
        baseline = np.asarray(before.right_toe_position, dtype=float)
        target = baseline + offset
        self.sock_cloth.set_task_right_toe_position_articulated(target.tolist())
        self._env.step()
        after = self._request_scene_geometry()
        actual = np.asarray(after.right_toe_position, dtype=float)
        achieved = actual - baseline
        target_error = float(np.linalg.norm(actual - target))
        offset_error = float(np.linalg.norm(achieved - offset))
        return {
            "ok": target_error <= tolerance and offset_error <= tolerance,
            "frame": "unity_world",
            "baseline_position": baseline.tolist(),
            "requested_offset_m": offset.tolist(),
            "target_position": target.tolist(),
            "actual_position": actual.tolist(),
            "achieved_offset_m": achieved.tolist(),
            "target_error_m": target_error,
            "offset_error_m": offset_error,
            "tolerance_m": tolerance,
        }

    def _translate_locked_pose_to_opening_normal(
        self,
        locked_pose_baseline: Mapping[str, Any],
        geometry: Any,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        settings = self.config["scene"].get("initial_pose_contract", {})
        drop = float(settings["vertical_toe_drop_m"])
        tolerance = float(
            settings.get("vertical_toe_drop_tolerance_m", 0.005)
        )
        if not np.isfinite(drop) or drop <= 0:
            raise ValueError(
                "vertical_toe_drop_m must be finite and positive"
            )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(
                "vertical_toe_drop_tolerance_m must be finite and positive"
            )
        center = 0.5 * (
            np.asarray(geometry.left_grasp_position, dtype=float)
            + np.asarray(geometry.right_grasp_position, dtype=float)
        )
        inward = np.asarray(geometry.opening_target_normal, dtype=float)
        baseline_toe = np.asarray(
            locked_pose_baseline["right_toe_position"], dtype=float
        )
        if (
            center.shape != (3,)
            or inward.shape != (3,)
            or baseline_toe.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(inward))
            or not np.all(np.isfinite(baseline_toe))
        ):
            raise ValueError(
                "opening geometry and locked right toe must be finite 3-vectors"
            )
        normal_length = float(np.linalg.norm(inward))
        if normal_length <= 1e-8:
            raise ValueError("opening target normal is degenerate")
        outward = -inward / normal_length
        if outward[1] >= -1e-3:
            raise RuntimeError(
                "gripper plate outward normal does not point world-down"
            )
        target_y = float(baseline_toe[1] - drop)
        distance = (target_y - center[1]) / outward[1]
        if not np.isfinite(distance) or distance <= 0:
            raise RuntimeError(
                "lowered right toe is not on the downward opening ray: "
                f"center={center.tolist()}, outward={outward.tolist()}, "
                f"target_y={target_y}, distance={distance}"
            )
        target_toe = center + outward * distance
        delta = target_toe - baseline_toe
        translated: Dict[str, Any] = {}
        for name in (
            "human_root_position",
            "chair_position",
            "human_anchor_position",
            "right_toe_position",
        ):
            value = np.asarray(locked_pose_baseline[name], dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"locked_pose_baseline.{name} must be a finite 3-vector"
                )
            translated[name] = (value + delta).tolist()
        return translated, {
            "ok": False,
            "frame": "unity_world",
            "baseline_right_toe_position": baseline_toe.tolist(),
            "opening_target_center": center.tolist(),
            "opening_inward_normal": (inward / normal_length).tolist(),
            "opening_outward_normal": outward.tolist(),
            "requested_vertical_drop_m": drop,
            "target_right_toe_position": target_toe.tolist(),
            "translation_m": delta.tolist(),
            "distance_along_outward_normal_m": float(distance),
            "tolerance_m": tolerance,
            "translated_locked_pose_baseline": translated,
        }

    def _observe_custom_player(self) -> Dict[str, Any]:
        self.sock_cloth.request_particles()
        self.sock_cloth.request_particle_velocities()
        self.sock_cloth.request_configuration()
        self.sock_cloth.request_registered_colliders()
        self.sock_cloth.request_grasp_state()
        self.sock_cloth.request_contacts()
        self.sock_cloth.request_coverage()
        self.sock_cloth.request_dressing_qa()
        if hasattr(self.robot, "GetJointInverseDynamicsForce"):
            self.robot.GetJointInverseDynamicsForce()
        self._env.step()
        cloth = dict(self.sock_cloth.data)
        robot_data = dict(getattr(self.robot, "data", {}) or {})
        contacts = list(cloth.get("cloth_contacts", []))
        collision_pairs = sorted(
            {
                (int(self.sock_cloth.id), int(item["collider_id"]))
                for item in contacts
            }
        )
        observation = {
            "robot": robot_data,
            "human": dict(getattr(self.human, "data", {}) or {}),
            "cloth": cloth,
            "camera": self._capture_custom_camera(cloth),
            "recording_camera": None,
            "contact_force": contacts,
            "dressing_qa": dict(cloth.get("dressing_qa", {})),
            "collision_pairs": collision_pairs,
            "diagnostics": self.diagnostics(),
        }
        observation.update(self.robot_signals(robot_data))
        return observation

    def _capture_custom_camera(
        self, cloth: Mapping[str, Any]
    ) -> Optional[Dict[str, np.ndarray]]:
        required = ("rgb_png", "depth_png", "sock_mask_png", "leg_mask_png")
        if any(not cloth.get(name) for name in required):
            return None
        instance = cloth.get("instance_mask_png")
        return {
            "rgb": self._decode(cloth["rgb_png"], "RGB"),
            "camera_depth": self._decode(cloth["depth_png"], "L").astype(np.uint8),
            "sock_mask": self._decode(cloth["sock_mask_png"], "L") > 0,
            "leg_mask": self._decode(cloth["leg_mask_png"], "L") > 0,
            "instance_mask": (
                self._decode(instance, "RGB") if instance else None
            ),
            "simulation_frame": cloth.get("camera_frame"),
            "raw": {
                "coverage": cloth.get("coverage_observations", {}),
                "protocol_version": cloth.get("protocol_version"),
            },
        }

    def measured_coverage(
        self, camera: Optional[Mapping[str, np.ndarray]] = None
    ) -> Optional[float]:
        # Preserve the historical class-call form used by callers/tests:
        # SockDressingEnv.measured_coverage(camera).
        if camera is None:
            camera = self
            environment = None
        else:
            environment = self
        if (
            environment is not None
            and environment.config["rcareworld"].get("profile") == "custom_player"
        ):
            geometry = getattr(environment.cloth, "data", {}).get(
                "scene_geometry", {}
            )
            distance = geometry.get("foot_to_opening_plane_m")
            if distance is not None and np.isfinite(float(distance)):
                current = float(distance)
                if not hasattr(environment, "_coverage_initial_distance"):
                    environment._coverage_initial_distance = current
                sock_length = float(
                    environment.config["scenario"]["sock"]["length_m"]
                )
                if sock_length > 0:
                    return float(
                        np.clip(
                            (
                                environment._coverage_initial_distance - current
                            )
                            / sock_length,
                            0.0,
                            1.0,
                        )
                    )
        sock = np.asarray(camera["sock_mask"], dtype=bool)
        leg = np.asarray(camera["leg_mask"], dtype=bool)
        if (
            sock.shape != leg.shape
            or sock.size == 0
            or np.count_nonzero(sock) in (0, sock.size)
            or np.count_nonzero(leg) in (0, leg.size)
            or np.array_equal(sock, leg)
        ):
            return None
        leg_pixels = np.count_nonzero(leg)
        return float(np.count_nonzero(sock & leg) / leg_pixels) if leg_pixels else None

    @staticmethod
    def cloth_radius_qa(
        cloth: Mapping[str, Any],
        radial_segments: Optional[int] = None,
        maximum_circumferential_stretch: Optional[float] = None,
    ) -> Dict[str, Any]:
        if maximum_circumferential_stretch is None:
            from .scenario import SOCK_MAX_CIRCUMFERENTIAL_STRETCH

            maximum_circumferential_stretch = (
                SOCK_MAX_CIRCUMFERENTIAL_STRETCH
            )
        maximum_circumferential_stretch = float(
            maximum_circumferential_stretch
        )
        if (
            not np.isfinite(maximum_circumferential_stretch)
            or maximum_circumferential_stretch <= 0
        ):
            raise ValueError(
                "maximum_circumferential_stretch must be finite and positive"
            )
        particles = np.asarray(cloth.get("particles", []), dtype=float)
        if particles.ndim != 2 or particles.shape[0] < 3 or particles.shape[1] != 3:
            return {"available": False, "passes": False}
        edges = np.asarray(cloth.get("particle_edges", ()), dtype=int)
        rest_lengths = np.asarray(
            cloth.get("particle_rest_edge_lengths", ()), dtype=float
        )
        diagnostics: Dict[str, Any] = {}
        if (
            edges.ndim == 2
            and edges.shape[1:] == (2,)
            and edges.shape[0] == rest_lengths.size
            and edges.size
            and edges.min() >= 0
            and edges.max() < particles.shape[0]
            and np.all(np.isfinite(rest_lengths))
            and np.all(rest_lengths > 0)
        ):
            pinned_particles = {
                int(index)
                for state in cloth.get("grasp_state", ())
                if bool(state.get("attached", False))
                for index in state.get("particle_indices", ())
            }
            opening_particles = {
                int(index)
                for index in cloth.get("opening_particle_indices", ())
            }
            edge_lengths = np.linalg.norm(
                particles[edges[:, 0]] - particles[edges[:, 1]],
                axis=1,
            )
            edge_stretches = edge_lengths / rest_lengths
            maximum_stretch_index = int(np.argmax(edge_stretches))
            stretch = float(edge_stretches[maximum_stretch_index])
            maximum = float(edge_lengths.max())
            method = "Obi topology structural edge stretch"
            pinned_counts = np.asarray(
                [
                    int(int(edge[0]) in pinned_particles)
                    + int(int(edge[1]) in pinned_particles)
                    for edge in edges
                ],
                dtype=int,
            )
            edge_classes = {}
            for name, count in (
                ("body_body", 0),
                ("pin_body", 1),
                ("pin_pin", 2),
            ):
                mask = pinned_counts == count
                if not np.any(mask):
                    edge_classes[name] = {
                        "edge_count": 0,
                        "maximum_stretch": None,
                        "maximum_stretch_edge": None,
                        "maximum_stretch_current_edge_m": None,
                        "maximum_stretch_rest_edge_m": None,
                    }
                    continue
                class_indices = np.flatnonzero(mask)
                local_index = int(np.argmax(edge_stretches[mask]))
                edge_index = int(class_indices[local_index])
                edge_classes[name] = {
                    "edge_count": int(mask.sum()),
                    "maximum_stretch": float(edge_stretches[edge_index]),
                    "maximum_stretch_edge": edges[edge_index].tolist(),
                    "maximum_stretch_current_edge_m": float(
                        edge_lengths[edge_index]
                    ),
                    "maximum_stretch_rest_edge_m": float(
                        rest_lengths[edge_index]
                    ),
                }
            diagnostics = {
                "minimum_rest_edge_m": float(rest_lengths.min()),
                "maximum_rest_edge_m": float(rest_lengths.max()),
                "minimum_current_edge_m": float(edge_lengths.min()),
                "maximum_current_edge_m": maximum,
                "maximum_stretch_edge": edges[maximum_stretch_index].tolist(),
                "pinned_particle_count": len(pinned_particles),
                "opening_particle_count": len(opening_particles),
                "edge_classes": edge_classes,
            }
        elif (
            radial_segments is not None
            and radial_segments >= 3
            and particles.shape[0] % radial_segments == 0
        ):
            rings = particles.reshape((-1, radial_segments, 3))
            edge_lengths = np.linalg.norm(
                np.roll(rings, -1, axis=1) - rings,
                axis=2,
            )
            rest_edge_length = 2.0 * 0.04 * np.sin(np.pi / radial_segments)
            stretch = float(edge_lengths.max() / rest_edge_length)
            maximum = float(
                edge_lengths.max()
                / (2.0 * np.sin(np.pi / radial_segments))
            )
            method = "mesh-ring maximum edge stretch"
        else:
            centered = particles - particles.mean(axis=0)
            _, _, axes = np.linalg.svd(centered, full_matrices=False)
            axis = axes[0]
            axial = np.outer(centered @ axis, axis)
            radius = np.linalg.norm(centered - axial, axis=1)
            method = "PCA axis radial-distance proxy"
            maximum = float(radius.max())
            stretch = maximum / 0.04
        return {
            "available": True,
            "centroid_world": particles.mean(axis=0).tolist(),
            "bounds_world": {
                "minimum": particles.min(axis=0).tolist(),
                "maximum": particles.max(axis=0).tolist(),
            },
            "maximum_radius_m": maximum,
            "circumferential_stretch_proxy": stretch,
            "passes": (
                (
                    method == "Obi topology structural edge stretch"
                    or maximum <= 0.06
                )
                and stretch <= maximum_circumferential_stretch
            ),
            "maximum_circumferential_stretch": (
                maximum_circumferential_stretch
            ),
            "method": method,
            **diagnostics,
        }

    def _capture_camera(self) -> Dict[str, np.ndarray]:
        camera = self.config["camera"]
        scene = self.config["scene"]
        width, height = int(camera["width"]), int(camera["height"])
        fov = float(camera["fov"])
        near, far = float(camera["depth_near_m"]), float(camera["depth_far_m"])
        self.camera.GetRGB(width, height, fov)
        self._env.step()
        rgb = self._decode(self.camera.data["rgb"], "RGB")
        self.camera.GetDepth(near, far, width, height, fov)
        self._env.step()
        depth = self._decode(self.camera.data["depth"], "L").astype(np.uint8)
        self.camera.GetID(width, height, fov)
        self._env.step()
        instance_mask = self._decode(self.camera.data["id_map"], "RGB")
        sock_mask = self._amodal_mask(
            int(self.config["assets"]["sock_id"]), width, height, fov
        )
        foot_ids = [
            int(value) for value in scene.get("human_foot_collider_ids", [])
        ]
        if not foot_ids and scene.get("human_id") is not None:
            foot_ids = [int(scene["human_id"])]
        leg_mask = np.zeros((height, width), dtype=bool)
        for target_id in foot_ids:
            leg_mask |= self._amodal_mask(target_id, width, height, fov)
        return {
            "rgb": rgb,
            "camera_depth": depth,
            "sock_mask": sock_mask,
            "leg_mask": leg_mask,
            "instance_mask": instance_mask,
            "simulation_frame": dict(self._env.data).get("frame"),
            "leg_mask_is_whole_human_proxy": not bool(scene.get("human_foot_collider_ids")),
            "raw": dict(self.camera.data),
        }

    def _capture_recording_camera(self) -> Dict[str, np.ndarray]:
        camera = self.config["camera"]
        width, height = int(camera["width"]), int(camera["height"])
        fov = float(camera["fov"])
        self.recording_camera.GetRGB(width, height, fov)
        self._env.step()
        return {
            "rgb": self._decode(self.recording_camera.data["rgb"], "RGB"),
            "raw": dict(self.recording_camera.data),
        }

    def robot_signals(
        self, robot_data: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract the configured 18 joints and an explicitly labelled force proxy."""
        from .joints import JointMap

        if self.robot is None:
            raise RuntimeError("robot is not loaded")
        data = robot_data if robot_data is not None else (
            getattr(self.robot, "data", {}) or {}
        )
        mapping = JointMap.from_config(self.config)
        indices = mapping.simulator_indices

        def extract(key: str) -> Optional[np.ndarray]:
            values = data.get(key)
            if values is None:
                return None
            array = np.asarray(values, dtype=float)
            if array.ndim != 1 or int(indices.max()) >= array.size:
                return None
            return array[indices]

        full_angle = data.get("joint_positions")
        angle = mapping.from_simulator(full_angle) if full_angle is not None else None
        joint_force = extract("joint_force")
        drive = extract("drive_forces")
        gravity = extract("gravity_forces")
        coriolis = extract("coriolis_centrifugal_forces")
        if angle is None or joint_force is None:
            raise RuntimeError(
                "RCareWorld did not return joint_positions/joint_force for all configured joints"
            )
        drive_available = drive is not None and bool(np.any(np.abs(drive) > 1e-9))
        joint_force_available = bool(np.any(np.abs(joint_force) > 1e-9))
        if drive_available:
            torque = drive
            torque_source = "drive_forces"
        elif joint_force_available:
            torque = joint_force
            torque_source = "joint_force"
        else:
            torque = np.zeros_like(joint_force)
            torque_source = "unavailable"
        if drive is not None and gravity is not None and coriolis is not None:
            external = joint_force - drive - gravity - coriolis
            source = "joint_force-drive_forces-gravity_forces-coriolis_centrifugal_forces"
        else:
            external = joint_force.copy()
            source = "joint_force proxy (inverse-dynamics components unavailable)"
        return {
            "angle": angle,
            "torque": torque,
            "external_torque": external,
            "external_torque_source": source,
            "torque_source": torque_source,
            "torque_available": torque_source != "unavailable",
        }

    def command(self, command: np.ndarray, previous: Optional[np.ndarray] = None) -> np.ndarray:
        """Bound an 18-D command and merge it into RCareWorld's full articulation vector."""
        from .joints import JointMap

        if self.robot is None:
            raise RuntimeError("robot is not loaded")
        mapping = JointMap.from_config(self.config)
        bounded = mapping.bound(command, previous)
        current = np.asarray(self.robot.data.get("joint_positions", []), dtype=float)
        if current.ndim != 1 or current.size <= int(mapping.simulator_indices.max()):
            raise RuntimeError("RCareWorld full joint state is unavailable")
        target = mapping.merge_for_simulator(bounded, current)
        direct_control = bool(
            self.config["scene"].get("robot_direct_joint_control", False)
        )
        if self._rollout_started:
            direct_control = bool(
                self.config["scene"].get(
                    "robot_rollout_direct_joint_control", direct_control
                )
            )
        if direct_control:
            self.robot.SetJointPositionDirectly(target.tolist())
        else:
            self.robot.SetJointPosition(target.tolist())
        self._env.step()
        return bounded

    def advance_physics(self, steps: int) -> None:
        """Advance additional fixed steps after a command without changing its target."""
        count = int(steps)
        if count < 0:
            raise ValueError("physics step count must be non-negative")
        if self._env is None:
            raise RuntimeError("environment is not connected")
        for _ in range(count):
            self._env.step()

    def grasp(self, side: str, max_distance_m: float = 0.03) -> Dict[str, Any]:
        if self.sock_cloth is None:
            raise RuntimeError("verified grasp API requires the custom Player profile")
        self.sock_cloth.grasp(side, max_distance_m)
        self.sock_cloth.request_grasp_state()
        self._env.step()
        states = {state.side: state for state in self.sock_cloth.grasp_states()}
        state = states[str(side).lower()]
        return {
            "side": state.side,
            "attached": state.attached,
            "particle_indices": list(state.particle_indices),
            "constraint_error": state.constraint_error,
            "peak_constraint_error": state.peak_constraint_error,
            "over_threshold_steps": state.over_threshold_steps,
            "release_reason": state.release_reason,
        }

    def release(self, side: str) -> Dict[str, Any]:
        if self.sock_cloth is None:
            raise RuntimeError("verified release API requires the custom Player profile")
        self.sock_cloth.release(side)
        self.sock_cloth.request_grasp_state()
        self._env.step()
        states = {state.side: state for state in self.sock_cloth.grasp_states()}
        state = states[str(side).lower()]
        return {
            "side": state.side,
            "attached": state.attached,
            "particle_indices": list(state.particle_indices),
            "constraint_error": state.constraint_error,
            "peak_constraint_error": state.peak_constraint_error,
            "over_threshold_steps": state.over_threshold_steps,
            "release_reason": state.release_reason,
        }

    def _amodal_mask(self, target_id: int, width: int, height: int, fov: float) -> np.ndarray:
        self.camera.GetAmodalMask(target_id, width, height, fov)
        self._env.step()
        image = self._decode(self.camera.data["amodal_mask"], "L")
        return image > 0

    @staticmethod
    def _decode(payload: bytes, mode: Optional[str]) -> np.ndarray:
        with Image.open(BytesIO(payload)) as image:
            if mode:
                image = image.convert(mode)
            return np.asarray(image).copy()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self) -> "SockDressingEnv":
        return self.connect()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
