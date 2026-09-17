from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from .assets import (
    generate_dry_airec_urdf,
    generate_rcareworld_runtime_urdf,
    generate_sock_obj,
    movable_joint_names,
)
from .calibration import calibrate_randomization, save_calibration
from .config import DEFAULT_CONFIG, load_config, resolve_package_path
from .doctor import format_report, run_doctor
from .environment import SockDressingEnv
from .episode import EpisodeWriter
from .joints import JointMap
from .quality import assess_observation_quality
from .scenario import scenario_from_config
from .pipeline import run_phase1_pipeline


def _prepare(config, scenario=None):
    scenario = scenario or scenario_from_config(config)
    assets = config["assets"]
    torobo = Path(assets["torobo_ros"]).expanduser()
    output = resolve_package_path(assets["output_dir"])
    urdf, meshes = generate_dry_airec_urdf(
        torobo, torobo / assets["product_config"], output
    )
    runtime_urdf = output / "dry_airec3_gripper_v2_rcareworld.urdf"
    compatibility = generate_rcareworld_runtime_urdf(urdf, runtime_urdf)
    joints = JointMap.from_config(config)
    movable = movable_joint_names(urdf)
    runtime_movable = movable_joint_names(runtime_urdf)
    if runtime_movable != movable:
        raise RuntimeError("RCareWorld mesh compatibility changed the URDF joint order")
    missing_controlled = set(joints.names) - set(movable)
    if missing_controlled:
        raise RuntimeError(
            "configured controlled joints are absent from generated URDF: "
            f"{sorted(missing_controlled)}"
        )
    sock = output / "sock.obj"
    vertices, triangles = generate_sock_obj(
        sock,
        length=scenario.sock_mesh.length_m,
        radius=scenario.sock_mesh.radius_m,
        radial_segments=scenario.sock_mesh.radial_segments,
        length_segments=scenario.sock_mesh.length_segments,
    )
    return {
        "urdf": str(urdf),
        "runtime_urdf": str(runtime_urdf),
        "mesh_compatibility": compatibility,
        "vendored_mesh_references": meshes,
        "sock_obj": str(sock),
        "sock_vertices": vertices,
        "sock_triangles": triangles,
        "sock_mesh": scenario.to_metadata()["sock_mesh"],
        "movable_joint_count": len(movable),
        "controlled_joint_names": list(joints.names),
    }


def _new_episode_path(config, root: Path, prefix: str = "smoke"):
    dataset_name = config["dataset"]["name"]
    episode_name = prefix + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    episode = root / dataset_name / "train" / episode_name
    return dataset_name, episode_name, episode


def _write_manifest(
    root: Path, dataset_name: str, episode_name: str, frames: int, name: str = "dataset_smoke.yaml"
) -> Path:
    manifest = root / name
    payload = {
        dataset_name: {
            "train": {episode_name: {"start": 0, "end": frames}},
            "test": {},
        }
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return manifest


def _offline_smoke(config, output_root: Path, frames: int):
    joints = JointMap.from_config(config)
    bounded = joints.bound(np.full(18, 999.0), np.zeros(18))
    dataset_name, episode_name, episode = _new_episode_path(config, output_root)
    with EpisodeWriter(
        episode, joints.names, {"smoke": True, "mode": "offline-contract"}
    ) as writer:
        for _ in range(frames):
            writer.append(
                rgb=np.zeros((4, 5, 3), dtype=np.uint8),
                sock_mask=np.zeros((4, 5), dtype=bool),
                leg_mask=np.zeros((4, 5), dtype=bool),
                camera_depth=np.ones((4, 5), dtype=np.uint8),
                angle=bounded,
                torque=np.zeros(18),
                external_torque=np.zeros(18),
            )
    manifest = _write_manifest(output_root, dataset_name, episode_name, frames)
    return {
        "ok": True,
        "mode": "offline-contract",
        "frames_written": frames,
        "episode": str(episode),
        "manifest": str(manifest),
        "warning": "No Unity physics was exercised.",
    }


def _unity_smoke(config, output_root: Path, frames: int):
    report = run_doctor(config)
    if not report["ok"]:
        failed = [name for name, ok in report["checks"].items() if not ok]
        raise RuntimeError(
            "live smoke prerequisites failed: "
            + ", ".join(failed)
            + "; run `python3 -m sock_dressing_simulation.cli doctor` for details"
        )
    prepared = _prepare(config)
    joints = JointMap.from_config(config)
    dataset_name, episode_name, episode = _new_episode_path(config, output_root)
    output = resolve_package_path(config["assets"]["output_dir"])
    with SockDressingEnv(config) as environment:
        environment.load(
            Path(prepared["runtime_urdf"]), output / "sock.obj"
        )
        first = environment.observe()
        if first["camera"] is None:
            raise RuntimeError("camera observation is unavailable")
        metadata = {
            "smoke": True,
            "mode": "unity-headless"
            if not config["rcareworld"].get("graphics", False)
            else "unity-graphics",
            "diagnostics": first["diagnostics"],
            "external_torque_source": first["external_torque_source"],
            "rcareworld_commit": config["rcareworld"]["commit"],
            "mesh_compatibility": prepared["mesh_compatibility"],
        }
        with EpisodeWriter(episode, joints.names, metadata) as writer:
            observation = first
            for frame in range(frames):
                camera = observation["camera"]
                writer.append(
                    rgb=camera["rgb"],
                    sock_mask=camera["sock_mask"],
                    leg_mask=camera["leg_mask"],
                    camera_depth=camera["camera_depth"],
                    angle=observation["angle"],
                    torque=observation["torque"],
                    external_torque=observation["external_torque"],
                )
                if frame + 1 < frames:
                    command = observation["angle"].copy()
                    command[0] += 0.005
                    environment.command(command, observation["angle"])
                    observation = environment.observe()
    manifest = _write_manifest(output_root, dataset_name, episode_name, frames)
    return {
        "ok": True,
        "mode": metadata["mode"],
        "frames_written": frames,
        "episode": str(episode),
        "manifest": str(manifest),
        "prepared_assets": prepared,
    }


def _unity_collect(config, output_root: Path, frames: int, seed: int):
    report = run_doctor(config)
    if not report["ok"]:
        failed = [name for name, ok in report["checks"].items() if not ok]
        raise RuntimeError("live collect prerequisites failed: " + ", ".join(failed))
    scenario = scenario_from_config(config, seed=seed)
    prepared = _prepare(config, scenario)
    joints = JointMap.from_config(config)
    dataset_name, episode_name, episode = _new_episode_path(
        config, output_root, prefix="phase1"
    )
    output = resolve_package_path(config["assets"]["output_dir"])
    cameras = []
    collisions = []
    with SockDressingEnv(config) as environment:
        environment.load(Path(prepared["runtime_urdf"]), output / "sock.obj")
        application = environment.apply_scenario(scenario)
        observation = environment.observe()
        metadata = {
            "phase": 1,
            "mode": "unity-graphics"
            if config["rcareworld"].get("graphics", False)
            else "unity-headless",
            "scenario": scenario.to_metadata(),
            "randomization_provenance": {
                "enabled": bool(
                    config.get("scenario", {})
                    .get("randomization", {})
                    .get("enabled", False)
                ),
                "calibration_file": config.get("scenario", {})
                .get("randomization", {})
                .get("calibration_file"),
                "physical_unmeasured_values_remain_nominal": True,
            },
            "scenario_application": application,
            "diagnostics": observation["diagnostics"],
            "external_torque_source": observation["external_torque_source"],
            "rcareworld_commit": config["rcareworld"]["commit"],
            "mesh_compatibility": prepared["mesh_compatibility"],
            "timing_contract": "command, simulator step, observe, append same state frame",
            "camera": dict(config["camera"]),
        }
        with EpisodeWriter(episode, joints.names, metadata) as writer:
            for frame in range(frames):
                camera = observation["camera"]
                if camera is None:
                    raise RuntimeError("camera observation is unavailable")
                cameras.append(camera)
                collisions.append(observation["collision_pairs"])
                writer.append(
                    rgb=camera["rgb"],
                    sock_mask=camera["sock_mask"],
                    leg_mask=camera["leg_mask"],
                    camera_depth=camera["camera_depth"],
                    angle=observation["angle"],
                    torque=observation["torque"],
                    external_torque=observation["external_torque"],
                )
                if frame + 1 < frames:
                    command = observation["angle"].copy()
                    command[0] += 0.005
                    environment.command(command, observation["angle"])
                    observation = environment.observe()
            quality = assess_observation_quality(
                cameras,
                expected_width=config["camera"]["width"],
                expected_height=config["camera"]["height"],
                exact_foot_colliders=bool(
                    config["scene"].get("human_foot_collider_ids")
                ),
            )
            writer.update_metadata(
                {
                    "observation_quality": quality,
                    "collision_pairs_by_frame": collisions,
                }
            )
    manifest = _write_manifest(
        output_root,
        dataset_name,
        episode_name,
        frames,
        name="dataset_phase1.yaml",
    )
    return {
        "ok": quality["learning_ready"],
        "episode_written": True,
        "mode": metadata["mode"],
        "frames_written": frames,
        "episode": str(episode),
        "manifest": str(manifest),
        "observation_quality": quality,
        "prepared_assets": prepared,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sock-sim")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    commands.add_parser("prepare-assets")
    smoke = commands.add_parser("smoke")
    smoke.add_argument(
        "--headless",
        action="store_true",
        help="launch the configured Unity player without graphics (the default)",
    )
    smoke.add_argument(
        "--offline",
        action="store_true",
        help="only validate the writer contract; do not launch Unity",
    )
    smoke.add_argument("--frames", type=int, default=3)
    smoke.add_argument(
        "--output-root",
        type=Path,
        default=resolve_package_path("artifacts/data"),
    )
    collect = commands.add_parser("collect")
    collect.add_argument("--frames", type=int, default=30)
    collect.add_argument("--seed", type=int, default=None)
    collect.add_argument(
        "--graphics",
        action="store_true",
        help="launch on the configured DISPLAY and require rendered observations",
    )
    collect.add_argument(
        "--output-root",
        type=Path,
        default=resolve_package_path("artifacts/phase1/data"),
    )
    calibrate = commands.add_parser("calibrate")
    calibrate.add_argument(
        "--data-root",
        type=Path,
        default=resolve_package_path(
            "../ShareSet/share_folder/data/data_center_general_depth_n5_sample"
        ),
    )
    calibrate.add_argument(
        "--output",
        type=Path,
        default=resolve_package_path("artifacts/phase1/randomization_calibration.json"),
    )
    calibrate.add_argument("--max-episodes", type=int, default=0)
    features = commands.add_parser("features")
    features.add_argument("--episode", type=Path, required=True)
    features.add_argument("--data-root", type=Path, required=True)
    features.add_argument("--manifest", type=Path, required=True)
    features.add_argument(
        "--audit-output",
        type=Path,
        default=resolve_package_path("artifacts/phase1/audit.json"),
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "doctor":
        report = run_doctor(config)
        print(format_report(report))
        return 0 if report["ok"] else 1
    if args.command == "prepare-assets":
        print(json.dumps(_prepare(config), indent=2))
        return 0
    if args.command == "calibrate":
        try:
            calibration = calibrate_randomization(
                args.data_root.resolve(), max_episodes=args.max_episodes
            )
            save_calibration(calibration, args.output.resolve())
        except (OSError, RuntimeError, ValueError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(
            json.dumps(
                {"ok": True, "output": str(args.output.resolve()), **calibration},
                indent=2,
            )
        )
        return 0
    if args.command == "features":
        try:
            result = run_phase1_pipeline(
                args.episode,
                data_root=args.data_root.resolve(),
                manifest=args.manifest.resolve(),
                dataset_name=config["dataset"]["name"],
                audit_output=args.audit_output.resolve(),
            )
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 3
    if args.command == "collect":
        if args.frames < 2:
            parser.error("collect --frames must be at least 2 for temporal QA")
        if args.graphics:
            config["rcareworld"]["graphics"] = True
        seed = (
            int(config.get("scenario", {}).get("seed", 0))
            if args.seed is None
            else args.seed
        )
        try:
            result = _unity_collect(config, args.output_root.resolve(), args.frames, seed)
        except (OSError, RuntimeError, ValueError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 3
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.headless:
        config["rcareworld"]["graphics"] = False
    try:
        result = (
            _offline_smoke(config, args.output_root.resolve(), args.frames)
            if args.offline
            else _unity_smoke(config, args.output_root.resolve(), args.frames)
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
