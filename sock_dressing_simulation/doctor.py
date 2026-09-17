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
from .config import RCAREWORLD_COMMIT, resolve_package_path
from .joints import JointMap


def run_doctor(config: Mapping) -> Dict[str, Any]:
    torobo = Path(config["assets"]["torobo_ros"]).expanduser()
    packages = discover_ros_packages(torobo) if torobo.is_dir() else {}
    product = torobo / config["assets"]["product_config"]
    pyrcare_spec = importlib.util.find_spec("pyrcareworld")
    executable_value = config["rcareworld"].get("executable")
    executable = resolve_package_path(executable_value) if executable_value else None
    checks = {
        "python_at_least_3_8": sys.version_info >= (3, 8),
        "python_package_importable": pyrcare_spec is not None,
        "xacro_executable": shutil.which("xacro") is not None,
        "torobo_ros_exists": torobo.is_dir(),
        "torobo_description_found": "torobo_description" in packages,
        "torobo_resources_found": "torobo_resources" in packages,
        "product_config_exists": product.is_file(),
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
    checks["assimp_4_1_runtime_available"] = assimp_version[:2] == (4, 1)
    scene_file = config["scene"].get("scene_file")
    checks["scene_file_available"] = _scene_available(executable, scene_file)
    installed_commit = _installed_commit(pyrcare_spec)
    checks["rcareworld_commit_verified"] = installed_commit == RCAREWORLD_COMMIT
    scene = config["scene"]
    diagnostics = {
        "expected_rcareworld_commit": RCAREWORLD_COMMIT,
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
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "diagnostics": diagnostics,
    }


def _mapping_valid(config: Mapping) -> bool:
    try:
        JointMap.from_config(config)
        return True
    except (KeyError, TypeError, ValueError):
        return False


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
