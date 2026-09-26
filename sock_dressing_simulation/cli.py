from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

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
from .quality import assess_observation_quality, compare_learning_domains
from .scenario import scenario_from_config
from .pipeline import run_phase1_pipeline
from .training import audit_training_data, load_training_specs, train_policy


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
        closed_toe=scenario.sock_mesh.closed_toe,
        bend_start=scenario.sock_mesh.rest_bend_start_m,
        bend_length=scenario.sock_mesh.rest_bend_length_m,
        bend_degrees=scenario.sock_mesh.rest_bend_degrees,
        bend_azimuth_degrees=scenario.sock_mesh.rest_bend_azimuth_degrees,
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
            "torque_source": first["torque_source"],
            "torque_available": first["torque_available"],
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
    if scenario.plantarflexion_degrees is None:
        raise ValueError(
            "foot.plantarflexion_degrees is unset; run `scene-preview` and "
            "record the selected angle before collection"
        )
    prepared = _prepare(config, scenario)
    joints = JointMap.from_config(config)
    dataset_name, episode_name, episode = _new_episode_path(
        config, output_root, prefix="phase1"
    )
    output = resolve_package_path(config["assets"]["output_dir"])
    cameras = []
    collisions = []
    with SockDressingEnv(config) as environment:
        environment.load(
            Path(prepared["runtime_urdf"]),
            output / "sock.obj",
            initial_joints=scenario.initial_joints,
        )
        application = environment.apply_scenario(scenario)
        observation = environment.observe()
        measured_coverage = environment.measured_coverage(observation["camera"])
        cloth_radius_qa = environment.cloth_radius_qa(
            observation["cloth"], scenario.sock_mesh.radial_segments
        )
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
            "torque_source": observation["torque_source"],
            "torque_available": observation["torque_available"],
            "external_torque_source": observation["external_torque_source"],
            "rcareworld_commit": config["rcareworld"]["commit"],
            "mesh_compatibility": prepared["mesh_compatibility"],
            "timing_contract": "command, simulator step, observe, append same state frame",
            "camera": dict(config["camera"]),
            "initial_coverage_measurement": {
                "value": measured_coverage,
                "target": scenario.initial_coverage_target,
                "source": "non-degenerate sock/leg amodal masks",
                "available": measured_coverage is not None,
            },
            "cloth_radius_qa": cloth_radius_qa,
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
                player_diagnostics=(
                    observation["diagnostics"]
                    if config["rcareworld"].get("profile") == "custom_player"
                    else None
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


def _scene_preview(config, output_dir: Path, angles) -> dict:
    report = run_doctor(config)
    if not report["ok"]:
        failed = [name for name, ok in report["checks"].items() if not ok]
        raise RuntimeError("scene preview prerequisites failed: " + ", ".join(failed))
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for angle in angles:
        scenario = scenario_from_config(
            config,
            seed=int(config.get("scenario", {}).get("seed", 0)),
            plantarflexion_degrees=float(angle),
        )
        prepared = _prepare(config, scenario)
        output = resolve_package_path(config["assets"]["output_dir"])
        with SockDressingEnv(config) as environment:
            environment.load(
                Path(prepared["runtime_urdf"]),
                output / "sock.obj",
                initial_joints=scenario.initial_joints,
            )
            application = environment.apply_scenario(scenario)
            observation = environment.observe()
            camera = observation["camera"]
            if camera is None:
                raise RuntimeError("camera observation is unavailable")
            image_path = output_dir / f"plantarflexion_{float(angle):+06.1f}.png"
            Image.fromarray(camera["rgb"]).save(image_path)
            coverage = environment.measured_coverage(camera)
            records.append(
                {
                    "angle_degrees": float(angle),
                    "image": str(image_path),
                    "scenario": scenario.to_metadata(),
                    "application": application,
                    "coverage": coverage,
                    "coverage_available": coverage is not None,
                    "cloth_radius_qa": environment.cloth_radius_qa(
                        observation["cloth"], scenario.sock_mesh.radial_segments
                    ),
                    "collision_pairs": observation["collision_pairs"],
                    "diagnostics": observation["diagnostics"],
                }
            )
    payload = {
        "ok": True,
        "selection_required": True,
        "angle_axis": config["scenario"]["foot"].get(
            "plantarflexion_axis", "x"
        ),
        "warning": (
            "Preview angles are labelled probes, not measured physical angles. "
            "Set foot.plantarflexion_degrees only after visual selection."
        ),
        "records": records,
    }
    report_path = output_dir / "preview.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report"] = str(report_path)
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sock-sim")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument(
        "--inference",
        action="store_true",
        help="also require training data and inference model assets",
    )
    commands.add_parser("prepare-assets")
    smoke = commands.add_parser("smoke")
    smoke_mode = smoke.add_mutually_exclusive_group()
    smoke_mode.add_argument(
        "--headless",
        action="store_true",
        help="launch the configured Unity player without graphics (the default)",
    )
    smoke_mode.add_argument(
        "--graphics",
        action="store_true",
        help="launch the configured Unity player with graphics",
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
    preview = commands.add_parser("scene-preview")
    preview.add_argument(
        "--angles",
        type=float,
        nargs="+",
        required=True,
        help="labelled plantarflexion probes in degrees; no value is selected automatically",
    )
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=resolve_package_path("artifacts/phase1/preview"),
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
    train = commands.add_parser("train")
    train.add_argument("--epochs", type=int)
    train.add_argument("--device")
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--audit-only", action="store_true", help="validate teacher data without training"
    )
    domain_audit = commands.add_parser("domain-audit")
    domain_audit.add_argument(
        "--simulation-episode", type=Path, action="append", required=True
    )
    domain_audit.add_argument(
        "--output",
        type=Path,
        default=resolve_package_path("artifacts/domain-audit.json"),
    )
    demo = commands.add_parser("demo")
    demo.add_argument("--checkpoint", type=Path)
    demo.add_argument("--device")
    demo.add_argument("--max-steps", type=int)
    demo.add_argument("--seed", type=int)
    demo.add_argument("--graphics", action="store_true")
    demo.add_argument("--headless", action="store_true")
    demo.add_argument("--sock-point", type=float, nargs=2, action="append")
    demo.add_argument("--leg-point", type=float, nargs=2, action="append")
    demo.add_argument("--sock-negative-point", type=float, nargs=2, action="append")
    demo.add_argument("--leg-negative-point", type=float, nargs=2, action="append")
    demo.add_argument(
        "--output-root",
        type=Path,
        default=resolve_package_path("artifacts/phase4/data"),
    )
    dressing_probe = commands.add_parser("dressing-probe")
    dressing_probe.add_argument(
        "--output",
        type=Path,
        default=resolve_package_path("artifacts/dressing_player/probe"),
    )
    dressing_trial = commands.add_parser("dressing-trial")
    dressing_trial.add_argument("--seed", type=int)
    dressing_trial.add_argument(
        "--output-root",
        type=Path,
        default=resolve_package_path("artifacts/dressing-trial"),
    )
    dressing_tune = commands.add_parser("dressing-tune")
    dressing_tune.add_argument("--max-trials", type=int, default=0)
    dressing_tune.add_argument(
        "--output-root",
        type=Path,
        default=resolve_package_path("artifacts/dressing-tuning"),
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "doctor":
        report = run_doctor(config)
        print(format_report(report))
        return 0 if report["ok"] and (not args.inference or report["inference_ready"]) else 1
    if args.command == "dressing-probe":
        if config["rcareworld"].get("profile") != "dressing_player":
            parser.error("dressing-probe requires config/dressing_player.yaml")
        if not config["rcareworld"].get("graphics", False):
            parser.error("dressing-probe requires graphics")
        try:
            from .dressing_player import run_dressing_probe

            result = run_dressing_probe(config, output=args.output.resolve())
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["physical_sock_dressing_success"] else 3
    if args.command in {"dressing-trial", "dressing-tune"}:
        if config["rcareworld"].get("profile") != "custom_player":
            parser.error(f"{args.command} requires config/custom_player.yaml")
        config["rcareworld"]["graphics"] = True
        try:
            from .dressing_trial import run_dressing_trial, run_staged_tuning

            prepared = _prepare(config)
            if args.command == "dressing-trial":
                result = run_dressing_trial(
                    config,
                    prepared=prepared,
                    output_root=args.output_root.resolve(),
                    seed=args.seed,
                )
            else:
                result = run_staged_tuning(
                    config,
                    prepared=prepared,
                    output_root=args.output_root.resolve(),
                    max_trials=args.max_trials,
                )
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["physical_sock_dressing_success"] else 3
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
    if args.command == "train":
        try:
            result = (
                audit_training_data(config)
                if args.audit_only
                else train_policy(
                    config,
                    epochs=args.epochs,
                    device=args.device,
                    resume=args.resume.resolve() if args.resume else None,
                )
            )
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 3
    if args.command == "domain-audit":
        try:
            real_episodes = [spec.directory for spec in load_training_specs(config)]
            result = compare_learning_domains(
                real_episodes,
                [path.resolve() for path in args.simulation_episode],
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["report"] = str(args.output.resolve())
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 3
    if args.command == "demo":
        if config["rcareworld"].get("profile") == "dressing_player":
            parser.error(
                "DressingPlayer has no verified 18-D Dry-AIREC action contract; "
                "use dressing-probe"
            )
        if args.graphics and args.headless:
            parser.error("demo accepts only one of --graphics and --headless")
        if config["inference"].get("require_graphics", False) and not args.graphics:
            parser.error("this inference profile requires --graphics")
        config["rcareworld"]["graphics"] = bool(args.graphics)
        if not args.graphics and (not args.sock_point or not args.leg_point):
            parser.error("headless demo requires --sock-point X Y and --leg-point X Y")
        report = run_doctor(config)
        if not report["ok"]:
            failed = [name for name, ok in report["checks"].items() if not ok]
            print(json.dumps({"ok": False, "error": "core doctor failed", "failed": failed}, indent=2))
            return 2
        try:
            from .demo import run_demo

            result = run_demo(
                config,
                prepared=_prepare(config),
                output_root=args.output_root.resolve(),
                sock_points=args.sock_point,
                leg_points=args.leg_point,
                sock_negative_points=args.sock_negative_point,
                leg_negative_points=args.leg_negative_point,
                max_steps=args.max_steps,
                checkpoint=args.checkpoint.resolve() if args.checkpoint else None,
                device=args.device,
                seed=args.seed,
            )
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as error:
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
    if args.command == "scene-preview":
        config["rcareworld"]["graphics"] = True
        try:
            result = _scene_preview(config, args.output_dir.resolve(), args.angles)
        except (OSError, RuntimeError, ValueError) as error:
            print(json.dumps({"ok": False, "error": str(error)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.headless:
        config["rcareworld"]["graphics"] = False
    elif args.graphics:
        config["rcareworld"]["graphics"] = True
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
