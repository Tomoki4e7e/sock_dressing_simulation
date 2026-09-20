from __future__ import annotations

import ctypes
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

from .assets import discover_ros_packages
from .config import SUPPORTED_RCAREWORLD_PROFILES, resolve_package_path
from .joints import JointMap


def run_doctor(config: Mapping) -> Dict[str, Any]:
    profile = str(config["rcareworld"].get("profile", "canonical"))
    expected_commit = SUPPORTED_RCAREWORLD_PROFILES[profile]
    dressing_profile = profile == "dressing_player"
    torobo = Path(config["assets"]["torobo_ros"]).expanduser()
    packages = discover_ros_packages(torobo) if torobo.is_dir() else {}
    product = torobo / config["assets"]["product_config"]
    pyrcare_spec = importlib.util.find_spec("pyrcareworld")
    executable_value = config["rcareworld"].get("executable")
    executable = resolve_package_path(executable_value) if executable_value else None
    checks = {
        "python_at_least_3_8": sys.version_info >= (3, 8),
        "python_package_importable": (
            resolve_package_path(config["rcareworld"].get("python_source", "")).is_dir()
            if dressing_profile
            else pyrcare_spec is not None
        ),
        "xacro_executable": dressing_profile or shutil.which("xacro") is not None,
        "torobo_ros_exists": dressing_profile or torobo.is_dir(),
        "torobo_description_found": dressing_profile or "torobo_description" in packages,
        "torobo_resources_found": dressing_profile or "torobo_resources" in packages,
        "product_config_exists": dressing_profile or product.is_file(),
        "joint_mapping_valid": _mapping_valid(config),
        "unity_executable_exists": bool(
            executable
            and executable.is_file()
            and executable.stat().st_mode & 0o111
        ),
    }
    missing_libraries = _missing_libraries(executable)
    checks["unity_runtime_libraries_resolved"] = not missing_libraries
    assimp_dir_value = config["rcareworld"].get("assimp_library_dir")
    assimp_dir = resolve_package_path(assimp_dir_value) if assimp_dir_value else None
    assimp_version = _assimp_version(assimp_dir)
    checks["assimp_4_1_runtime_available"] = (
        True if dressing_profile else assimp_version[:2] == (4, 1)
    )
    scene_file = config["scene"].get("scene_file")
    checks["scene_file_available"] = _scene_available(executable, scene_file)
    checkout_value = config["rcareworld"].get("checkout")
    installed_commit = (
        _checkout_commit(resolve_package_path(checkout_value))
        if checkout_value
        else _installed_commit(pyrcare_spec)
    )
    checks["rcareworld_commit_verified"] = installed_commit == expected_commit
    scene = config["scene"]
    diagnostics = {
        "profile": profile,
        "expected_rcareworld_commit": expected_commit,
        "installed_rcareworld_commit": installed_commit,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_recommendation": "RCareWorld README recommends Python 3.10",
        "unity_executable": str(executable) if executable else "package default",
        "missing_unity_libraries": missing_libraries,
        "assimp_library_dir": str(assimp_dir) if assimp_dir else "not configured",
        "assimp_runtime_version": (
            ".".join(map(str, assimp_version)) if assimp_version else "unavailable"
        ),
        "human_foot_collision": (
            "configured scene IDs"
            if scene.get("human_foot_collider_ids")
            else "UNVERIFIED: HumanBodyIK API does not identify exact foot colliders"
        ),
        "robot_obi_collider": (
            "UNVERIFIED: request exists in Python, but player registration "
            "cannot be validated without the Unity project"
        ),
        "contact_force": (
            "proxy IDs configured; still not direct force"
            if scene.get("contact_proxy_ids")
            else "UNAVAILABLE: configure collision/effort proxy IDs if needed"
        ),
    }
    inference_checks, inference_diagnostics = _inference_checks(config)
    diagnostics["inference"] = inference_diagnostics
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "inference_ready": all(inference_checks.values()),
        "inference_checks": inference_checks,
        "diagnostics": diagnostics,
    }


def _checkout_commit(path: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "not available"


def _mapping_valid(config: Mapping) -> bool:
    try:
        JointMap.from_config(config)
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _inference_checks(config: Mapping) -> tuple:
    settings = config.get("inference", {})

    def resolved(key: str) -> Path:
        value = settings.get(key)
        return resolve_package_path(value) if value else Path()

    sam = settings.get("sam2", {})
    depth = settings.get("depth_anything", {})
    data_root = resolved("training_data_root")
    dataset_name = settings.get("training_dataset_name", "")
    dataset = data_root / dataset_name
    sample_rgb = next(dataset.glob("train/*/camera_right/*.png"), None) if dataset.is_dir() else None
    sample_sock = next(dataset.glob("train/*/depth_mask/sock_depth/*.png"), None) if dataset.is_dir() else None
    sample_foot = next(dataset.glob("train/*/depth_mask/foot_depth/*.png"), None) if dataset.is_dir() else None
    if sample_foot is None and dataset.is_dir():
        sample_foot = next(dataset.glob("train/*/depth_mask/leg_depth/*.png"), None)
    paths = {
        "shareset_model_source": resolved("shareset_src"),
        "training_manifest": resolved("training_manifest"),
        "training_dataset": dataset,
        "training_rgb": sample_rgb,
        "training_sock_depth": sample_sock,
        "training_foot_depth": sample_foot,
        "policy_checkpoint": resolved("checkpoint"),
        "policy_stats": resolved("stats"),
        "sam2_source": resolve_package_path(sam.get("source", "")),
        "sam2_config": resolve_package_path(sam.get("config", "")),
        "sam2_checkpoint": resolve_package_path(sam.get("checkpoint", "")),
        "depth_anything_source": resolve_package_path(depth.get("source", "")),
        "depth_anything_checkpoint": resolve_package_path(depth.get("checkpoint", "")),
        "depth_calibration": resolve_package_path(depth.get("calibration", "")),
    }
    directory_keys = {
        "shareset_model_source",
        "training_dataset",
        "sam2_source",
        "depth_anything_source",
    }
    checks = {
        name: bool(path and (path.is_dir() if name in directory_keys else path.is_file()))
        for name, path in paths.items()
    }
    for module_name, check_name in (
        ("torch", "torch_importable"),
        ("scipy", "scipy_importable"),
        ("cv2", "opencv_importable"),
        ("hydra", "hydra_importable"),
    ):
        checks[check_name] = importlib.util.find_spec(module_name) is not None
    configured_device = str(settings.get("device", "cuda"))
    cuda_available = False
    if checks["torch_importable"]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except (ImportError, RuntimeError):
            pass
    checks["configured_device_available"] = (
        cuda_available if configured_device.startswith("cuda") else True
    )
    diagnostics = {
        "paths": {name: str(path) if path else "not found" for name, path in paths.items()},
        "configured_device": configured_device,
        "cuda_available": cuda_available,
        "note": (
            "Core RCareWorld readiness is reported separately. Missing inference "
            "weights do not disable smoke/collection commands."
        ),
    }
    return checks, diagnostics


def _installed_commit(spec) -> str:
    if spec is None or spec.origin is None:
        return "not installed"
    path = Path(spec.origin).resolve()
    for parent in (path,) + tuple(path.parents):
        if (parent / ".git").exists():
            process = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if process.returncode == 0:
                return process.stdout.strip()
    return "unknown (installed without git metadata)"


def _missing_libraries(executable: Path) -> list:
    if executable is None or not executable.is_file() or shutil.which("ldd") is None:
        return ["executable unavailable"]
    process = subprocess.run(
        ["ldd", str(executable)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return [
        line.strip().split(" => ", 1)[0]
        for line in process.stdout.splitlines()
        if "not found" in line
    ]


def _assimp_version(directory: Path) -> tuple:
    if directory is None:
        return ()
    library = directory / "libassimp.so"
    if not library.is_file():
        return ()
    try:
        assimp = ctypes.CDLL(str(library.resolve()))
        return tuple(
            int(getattr(assimp, function)())
            for function in (
                "aiGetVersionMajor",
                "aiGetVersionMinor",
            )
        )
    except (AttributeError, OSError):
        return ()


def _scene_available(executable: Path, scene_file: str) -> bool:
    if not scene_file:
        return True
    scene = Path(scene_file).expanduser()
    if scene.is_absolute():
        return scene.is_file()
    if executable is None:
        return False
    data_dir = executable.with_name(executable.stem + "_Data")
    return (data_dir / "StreamingAssets" / "SceneData" / scene).is_file()


def format_report(report: Mapping) -> str:
    return json.dumps(report, indent=2, sort_keys=True)
