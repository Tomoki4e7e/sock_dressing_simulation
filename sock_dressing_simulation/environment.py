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
        self.chair = []
        self.grasp_anchors = []
        self._grasp_attachment_report = []

    def connect(self) -> "SockDressingEnv":
        if self._env is None:
            try:
                from pyrcareworld.envs.base_env import RCareWorld
            except ImportError as error:
                raise RuntimeError(
                    "pyrcareworld is not installed; run scripts/setup_rcareworld.sh"
                ) from error
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
        assets = self.config["assets"]
        self.robot = self._env.LoadURDF(
            str(Path(urdf).resolve()), id=int(assets["robot_id"]), native_ik=False, axis="z"
        )
        self._env.step()
        self._validate_simulator_joint_mapping()
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
        self.cloth = self._env.LoadCloth(str(Path(sock_obj).resolve()), id=int(assets["sock_id"]))
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
        if scene.get("camera_id") is not None:
            try:
                import pyrcareworld.attributes as attr
            except ImportError as error:
                raise RuntimeError("pyrcareworld attributes are unavailable") from error
            self.camera = self._env.InstanceObject(
                name="Camera", id=int(scene["camera_id"]), attr_type=attr.CameraAttr
            )
            self.camera.SetTransform(
                position=list(scene["camera_position"]),
                rotation=list(scene["camera_rotation"]),
            )
        self._env.step()

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
            anchor.SetTransform(position=list(item["position"]), is_world=True)
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
            relative=True,
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

    def apply_scenario(self, scenario: Any) -> Dict[str, Any]:
        """Apply one validated scenario before the first recorded observation."""
        if self.robot is None or self.cloth is None:
            raise RuntimeError("robot and cloth must be loaded before applying a scenario")
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
        if not simulator_names or int(mapping.simulator_indices.max()) >= len(
            simulator_names
        ):
            raise RuntimeError(
                "configured simulator joint index exceeds RCareWorld movable joints: "
                f"max_index={int(mapping.simulator_indices.max())}, "
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

    def diagnostics(self) -> Dict[str, Any]:
        scene = self.config["scene"]
        return {
            "human_configured": scene.get("human_id") is not None,
            "camera_configured": scene.get("camera_id") is not None,
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
        self._env.GetCurrentCollisionPairs()
        self._env.step()
        observation["collision_pairs"] = list(
            self._env.data.get("CurrentCollisionPairs", [])
        )
        observation.update(self.robot_signals())
        return observation

    @staticmethod
    def measured_coverage(camera: Mapping[str, np.ndarray]) -> Optional[float]:
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
    ) -> Dict[str, Any]:
        particles = np.asarray(cloth.get("particles", []), dtype=float)
        if particles.ndim != 2 or particles.shape[0] < 3 or particles.shape[1] != 3:
            return {"available": False, "passes": False}
        if (
            radial_segments is not None
            and radial_segments >= 3
            and particles.shape[0] % radial_segments == 0
        ):
            rings = particles.reshape((-1, radial_segments, 3))
            radius = np.linalg.norm(rings - rings.mean(axis=1, keepdims=True), axis=2)
            method = "mesh-ring centroid radial distance"
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
            "passes": maximum <= 0.06 and stretch <= 1.5,
            "method": method,
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
        sock_mask = self._amodal_mask(int(self.config["assets"]["sock_id"]), width, height, fov)
        foot_ids = [int(value) for value in scene.get("human_foot_collider_ids", [])]
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
            "leg_mask_is_whole_human_proxy": not bool(scene.get("human_foot_collider_ids")),
            "raw": dict(self.camera.data),
        }

    def robot_signals(self) -> Dict[str, np.ndarray]:
        """Extract the configured 18 joints and an explicitly labelled force proxy."""
        from .joints import JointMap

        if self.robot is None:
            raise RuntimeError("robot is not loaded")
        data = getattr(self.robot, "data", {}) or {}
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
        torque = drive if drive is not None else joint_force
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
        self.robot.SetJointPosition(target.tolist())
        self._env.step()
        return bounded

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
