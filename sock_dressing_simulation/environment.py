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
        self.sock_cloth = None

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
        self.robot.SetTransform(
            position=list(scene["robot_position"]),
            rotation=list(scene.get("robot_rotation", [0, 0, 0])),
        )
        self.robot.AddObiCollider()
        self.cloth.SetTransform(position=list(scene["sock_position"]))
        self.human = self._env.GetAttr(int(scene["human_id"]))
        if scene.get("human_position") is not None:
            self.human.SetTransform(
                position=list(scene["human_position"]),
                rotation=list(scene.get("human_rotation", [0, 0, 0])),
            )
        self.sock_cloth.configure_right_leg_colliders(int(scene["human_id"]))
        self._create_chair()
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
            particle_radius_m=float(expected["particle_radius_m"]),
            particle_mass_kg=float(expected["particle_mass_kg"]),
            collision_margin_m=float(expected["collision_margin_m"]),
            friction=float(expected["friction"]),
            self_collision=bool(expected["self_collision"]),
            substeps=int(expected["substeps"]),
            solver_iterations=int(expected["solver_iterations"]),
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
            self.camera.SetTransform(
                position=list(scene["camera_position"]),
                rotation=list(scene["camera_rotation"]),
            )
        if scene.get("recording_camera_id") is not None:
            self.recording_camera = self._env.InstanceObject(
                name="Camera",
                id=int(scene["recording_camera_id"]),
                attr_type=attr.CameraAttr,
            )
            self.recording_camera.SetTransform(
                position=list(scene["recording_camera_position"]),
                rotation=list(scene["recording_camera_rotation"]),
            )
        self._env.step()

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
            applied = self.command(np.asarray(scenario.initial_joints, dtype=float), current)
            for _ in range(scenario.settle_steps):
                self._env.step()
            return {
                "initial_command_steps": int(not np.allclose(current, applied)),
                "settle_steps": scenario.settle_steps,
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
            return {
                "human_configured": self.human is not None,
                "camera_configured": (
                    self.camera is not None
                    or bool(getattr(self.cloth, "data", {}).get("rgb_png"))
                ),
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
                "grasp_state": list(
                    getattr(self.cloth, "data", {}).get("grasp_state", [])
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
        self.sock_cloth.request_contacts()
        self._env.step()
        cloth = dict(self.sock_cloth.data)
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
        observation = {
            "robot": dict(getattr(self.robot, "data", {}) or {}),
            "human": dict(getattr(self.human, "data", {}) or {}),
            "cloth": cloth,
            "camera": camera,
            "recording_camera": recording,
            "contact_force": contacts,
            "collision_pairs": collision_pairs,
            "diagnostics": self.diagnostics(),
        }
        observation.update(self.robot_signals())
        return observation

    def _observe_custom_player(self) -> Dict[str, Any]:
        self.sock_cloth.request_particles()
        self.sock_cloth.request_particle_velocities()
        self.sock_cloth.request_configuration()
        self.sock_cloth.request_registered_colliders()
        self.sock_cloth.request_grasp_state()
        self.sock_cloth.request_contacts()
        self.sock_cloth.request_coverage()
        self._env.step()
        cloth = dict(self.sock_cloth.data)
        contacts = list(cloth.get("cloth_contacts", []))
        collision_pairs = sorted(
            {
                (int(self.sock_cloth.id), int(item["collider_id"]))
                for item in contacts
            }
        )
        observation = {
            "robot": dict(getattr(self.robot, "data", {}) or {}),
            "human": dict(getattr(self.human, "data", {}) or {}),
            "cloth": cloth,
            "camera": self._capture_custom_camera(cloth),
            "recording_camera": None,
            "contact_force": contacts,
            "collision_pairs": collision_pairs,
            "diagnostics": self.diagnostics(),
        }
        observation.update(self.robot_signals())
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
        self._env.SetTimeScale(0)
        self._env.step()
        try:
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
        finally:
            self._env.SetTimeScale(1)
            self._env.step()
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
