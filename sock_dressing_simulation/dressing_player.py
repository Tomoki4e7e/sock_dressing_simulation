from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
from PIL import Image

from .config import DRESSING_PLAYER_COMMIT, resolve_package_path


class DressingPlayerEnv:
    """Isolated adapter for the binary-only phy-robo-care DressingPlayer."""

    def __init__(self, config: Mapping, backend: Optional[Any] = None):
        self.config = config
        self._env = backend
        self.robot = self.gripper = self.camera = self.cloth = self.grasper = None

    def connect(self) -> "DressingPlayerEnv":
        if self._env is None:
            source = resolve_package_path(self.config["rcareworld"]["python_source"])
            sys.path.insert(0, str(source))
            try:
                from pyrcareworld.envs.dressing_env import DressingEnv
            except ImportError as error:
                raise RuntimeError(
                    f"alternate pyrcareworld is unavailable at {source}"
                ) from error
            finally:
                sys.path.pop(0)
            executable = resolve_package_path(self.config["rcareworld"]["executable"])
            self._env = DressingEnv(
                executable_file=str(executable.resolve()),
                port=int(self.config["rcareworld"].get("port", 5005)),
                graphics=bool(self.config["rcareworld"].get("graphics", True)),
                log_level=int(self.config["rcareworld"].get("log_level", 1)),
            )
        self.robot = self._env.get_robot()
        self.gripper = self._env.get_gripper()
        self.camera = self._env.get_camera()
        self.cloth = self._env.get_cloth()
        self._env.step()
        return self

    def capabilities(self) -> dict:
        settings = self.config["dressing_player"]
        attrs = getattr(self._env, "attrs", {}) or {}
        ids = {
            "robot": int(settings["robot_id"]),
            "gripper": int(settings["gripper_id"]),
            "camera": int(settings["camera_id"]),
            "cloth": int(settings["cloth_id"]),
        }
        present = {name: value in attrs for name, value in ids.items()}
        joint_positions = list((getattr(self.robot, "data", {}) or {}).get("joint_positions", []))
        available_grippers = 1 if present["gripper"] else 0
        required_grippers = int(settings.get("required_grippers", 2))
        return {
            "profile": "dressing_player",
            "commit": self.config["rcareworld"]["commit"],
            "expected_commit": DRESSING_PLAYER_COMMIT,
            "ids": ids,
            "objects_present": present,
            "joint_count": len(joint_positions),
            "available_grippers": available_grippers,
            "required_grippers": required_grippers,
            "bimanual_ready": available_grippers >= required_grippers,
            "cloth_particles_available": bool(
                (getattr(self.cloth, "data", {}) or {}).get("particles")
            ),
            "graphics": bool(self.config["rcareworld"].get("graphics", False)),
            "binary_only": True,
            "scene_editable": False,
        }

    def configure_grasper(self) -> None:
        try:
            from pyrcareworld.attributes import ClothGrasperAttr
        except ImportError as error:
            raise RuntimeError("ClothGrasperAttr is absent from alternate runtime") from error
        settings = self.config["dressing_player"]
        self.grasper = self.gripper.SetType(ClothGrasperAttr)
        self.grasper.set_cloth_and_robot(
            int(settings["cloth_id"]),
            str(settings["cloth_name"]),
            int(settings["robot_id"]),
            str(settings["gripper_name"]),
            float(settings["grasp_radius_m"]),
        )
        self._env.step()

    def garment_held(self) -> bool:
        if self.grasper is None:
            return False
        return bool(self.grasper.is_garment_being_held())

    def particles(self) -> np.ndarray:
        self.cloth.GetParticles()
        self._env.step()
        values = np.asarray(
            (getattr(self.cloth, "data", {}) or {}).get("particles", []), dtype=float
        )
        return values if values.ndim == 2 and values.shape[1:] == (3,) else np.empty((0, 3))

    def capture(self) -> dict:
        camera = self.config["camera"]
        width, height = int(camera["width"]), int(camera["height"])
        fov = float(camera["fov"])
        self.camera.GetRGB(width, height, fov)
        self._env.step()
        rgb = _decode(self.camera.data["rgb"], "RGB")
        masks = {}
        for name, target in (
            ("sock_mask", self.config["dressing_player"]["cloth_id"]),
            ("leg_mask", self.config["dressing_player"].get("leg_id")),
        ):
            if target is None:
                masks[name] = None
                continue
            self.camera.GetAmodalMask(int(target), width, height, fov)
            self._env.step()
            masks[name] = _decode(self.camera.data["amodal_mask"], "L") > 0
        return {"rgb": rgb, **masks}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self) -> "DressingPlayerEnv":
        return self.connect()

    def __exit__(self, *_args) -> None:
        self.close()


def run_dressing_probe(
    config: Mapping,
    *,
    output: Path,
    environment_factory: Callable = DressingPlayerEnv,
) -> dict:
    """Exercise real cloth grasp-follow without claiming bimanual sock success."""
    if not config["rcareworld"].get("graphics", False):
        raise ValueError("DressingPlayer cloth/SAM probes require graphics")
    settings = config["dressing_player"]
    trajectory = settings["trajectory"]
    output.mkdir(parents=True, exist_ok=True)
    with environment_factory(config) as environment:
        capabilities = environment.capabilities()
        before = environment.particles()
        environment.gripper.GripperOpen()
        environment._env.step(int(settings.get("settle_steps", 60)))
        environment.robot.IKTargetDoMove(
            position=list(trajectory["approach_position"]),
            duration=float(trajectory["duration_s"]),
            speed_based=False,
        )
        environment.robot.IKTargetDoRotate(
            rotation=list(trajectory["approach_rotation"]),
            duration=float(trajectory["duration_s"]),
            speed_based=False,
        )
        environment.robot.WaitDo()
        environment.configure_grasper()
        environment.gripper.GripperClose()
        environment.grasper.toggle_grasp_and_gripper()
        environment._env.step(int(settings.get("hold_steps", 30)))
        held_before_pull = environment.garment_held()
        grasped = environment.particles()
        environment.robot.IKTargetDoMove(
            position=list(trajectory["pull_offset"]),
            duration=float(trajectory["duration_s"]),
            speed_based=False,
            relative=True,
        )
        environment.robot.WaitDo()
        environment._env.step(int(settings.get("hold_steps", 30)))
        held_after_pull = environment.garment_held()
        after = environment.particles()
        frame = environment.capture()
        Image.fromarray(frame["rgb"].astype(np.uint8), "RGB").save(output / "probe.png")

    displacement = _centroid_displacement(grasped, after)
    initial_displacement = _centroid_displacement(before, grasped)
    tracking_limit = float(settings.get("maximum_tracking_error_m", 0.06))
    tracking_verified = bool(
        held_before_pull
        and held_after_pull
        and displacement is not None
        and displacement > 0
        and displacement <= np.linalg.norm(trajectory["pull_offset"]) + tracking_limit
    )
    physical_success = bool(tracking_verified and capabilities["bimanual_ready"])
    report = {
        "ok": tracking_verified,
        "physical_sock_dressing_success": physical_success,
        "capabilities": capabilities,
        "held_before_pull": held_before_pull,
        "held_after_pull": held_after_pull,
        "cloth_centroid_displacement_before_grasp_m": initial_displacement,
        "cloth_centroid_displacement_during_pull_m": displacement,
        "grasp_follow_verified": tracking_verified,
        "image": str(output / "probe.png"),
        "blockers": [] if physical_success else [
            "distributed DressingPlayer exposes one gripper; bimanual sock opening control requires two",
            "binary scene exposes no editable foot collider or sock replacement contract",
        ],
    }
    (output / "probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _centroid_displacement(first: np.ndarray, second: np.ndarray) -> Optional[float]:
    if not len(first) or not len(second) or first.shape != second.shape:
        return None
    return float(np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)))


def _decode(payload: bytes, mode: str) -> np.ndarray:
    with Image.open(BytesIO(payload)) as image:
        return np.asarray(image.convert(mode)).copy()
