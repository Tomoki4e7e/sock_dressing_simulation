from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from PIL import Image
import yaml

from .config import resolve_package_path
from .environment import SockDressingEnv
from .episode import EpisodeWriter
from .joints import JointMap
from .perception import SAMDepthPerception, select_prompt_points
from .policy import SAMDAMSARNNPolicy
from .scenario import scenario_from_config


def run_demo(
    config: Mapping,
    *,
    prepared: Mapping,
    output_root: Path,
    sock_points: Optional[Sequence[Sequence[float]]],
    leg_points: Optional[Sequence[Sequence[float]]],
    sock_negative_points: Optional[Sequence[Sequence[float]]] = None,
    leg_negative_points: Optional[Sequence[Sequence[float]]] = None,
    max_steps: Optional[int] = None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    environment_factory: Callable = SockDressingEnv,
    perception_factory: Callable = SAMDepthPerception,
    policy_factory: Callable = SAMDAMSARNNPolicy,
) -> dict:
    settings = config["inference"]
    steps = int(max_steps if max_steps is not None else settings.get("max_steps", 250))
    if steps < 1:
        raise ValueError("max_steps must be positive")
    joints = JointMap.from_config(config)
    scenario = scenario_from_config(config, seed=seed)
    episode = output_root / config["dataset"]["name"] / "train" / _episode_name()
    policy = policy_factory(config, checkpoint=checkpoint, device=device)
    perception = perception_factory(config)
    reference_actions = None
    reference_path = settings.get("reference_actions")
    if reference_path:
        reference_actions = np.loadtxt(
            resolve_package_path(reference_path),
            delimiter=",",
            dtype=float,
            ndmin=2,
        )
        if (
            reference_actions.ndim != 2
            or reference_actions.shape[1] != 18
            or not np.all(np.isfinite(reference_actions))
        ):
            raise ValueError("reference_actions must be a finite 18-column CSV")
    reference_blend = float(settings.get("reference_action_blend", 0.0))
    reference_hold = int(settings.get("reference_action_hold_steps", 1))
    reference_interpolation = str(
        settings.get("reference_action_interpolation", "hold")
    )
    reference_pull = float(settings.get("reference_cartesian_pull_m", 0.0))
    physics_steps_per_action = int(settings.get("physics_steps_per_action", 1))
    if (
        not 0 <= reference_blend <= 1
        or reference_hold < 1
        or physics_steps_per_action < 1
        or reference_interpolation not in {"hold", "linear"}
    ):
        raise ValueError("invalid reference action projection settings")
    output = resolve_package_path(config["assets"]["output_dir"])
    metadata = {
        "phase": 4,
        "mode": "airec-closed-loop",
        "checkpoint": str(policy.checkpoint),
        "checkpoint_sha256": policy.checkpoint_sha256,
        "perception": "SAM2+Depth-Anything-V2",
        "seed": scenario.seed,
        "scenario": scenario.to_metadata(),
        "command_contract": "model 36D output -> first 18D -> JointMap.bound -> RCareWorld",
        "reference_action_projection": {
            "path": reference_path,
            "blend": reference_blend,
            "hold_steps": reference_hold,
            "interpolation": reference_interpolation,
            "cartesian_pull_m": reference_pull,
        },
    }
    stop_reason = "max_steps"
    frames = 0
    quality_by_frame = []
    coverage_by_frame = []
    grasp_quality_by_frame = []
    cloth_quality_by_frame = []
    dressing_quality_by_frame = []
    foot_contact_ids_by_frame = []
    rigid_collision_qa_by_frame = []
    human_chair_lock_qa_by_frame = []
    final_task_success = {}
    video_path = episode / "demo.mp4"
    configured_prompts = settings.get("prompts", {})
    if not sock_points:
        sock_points = configured_prompts.get("sock", {}).get("positive")
        sock_negative_points = configured_prompts.get("sock", {}).get("negative")
    if not leg_points:
        leg_points = configured_prompts.get("leg", {}).get("positive")
        leg_negative_points = configured_prompts.get("leg", {}).get("negative")
    with environment_factory(config) as environment:
        environment.load(
            Path(prepared["runtime_urdf"]),
            output / "sock.obj",
            initial_joints=scenario.initial_joints,
        )
        application = environment.apply_scenario(scenario)
        recording_camera_frame = None
        if config["scene"].get("recording_camera_frame_opening", False):
            recording_camera_frame = environment.frame_recording_camera_on_opening(
                float(config["scene"].get("recording_camera_opening_distance_m", 0.65)),
                float(config["scene"].get("recording_camera_opening_lateral_m", 0.0)),
                bool(config["scene"].get("recording_camera_opening_reverse", False)),
            )
        cartesian_reference = None
        if reference_pull > 0:
            geometry = environment._request_scene_geometry()
            direction = np.asarray(geometry.right_toe_position, dtype=float) - np.asarray(
                geometry.opening_center, dtype=float
            )
            direction /= np.linalg.norm(direction)
            cartesian_reference = {
                "left": np.asarray(geometry.left_grasp_position, dtype=float),
                "right": np.asarray(geometry.right_grasp_position, dtype=float),
                "direction": direction,
            }
        observation = environment.observe()
        if observation["camera"] is None:
            raise RuntimeError("camera observation is unavailable")
        cloth_following_baseline = _cloth_following_state(observation, config)
        initial_renderer_masks = _renderer_masks(observation["camera"])
        if (
            config["rcareworld"].get("profile") == "custom_player"
            and initial_renderer_masks is not None
        ):
            sock_center = _mask_centroid_point(initial_renderer_masks["sock"])
            leg_center = _mask_centroid_point(initial_renderer_masks["leg"])
            sock_points = [sock_center]
            leg_points = [leg_center]
            sock_negative_points = [leg_center]
            leg_negative_points = [sock_center]
        sock_labels = leg_labels = None
        if not sock_points or not leg_points:
            if not config["rcareworld"].get("graphics", False):
                raise ValueError(
                    "headless demo requires --sock-point and --leg-point"
                )
            if not sock_points:
                sock_points, sock_labels = select_prompt_points(
                    observation["camera"]["rgb"], "sock"
                )
            if not leg_points:
                leg_points, leg_labels = select_prompt_points(
                    observation["camera"]["rgb"], "leg"
                )
        if sock_negative_points:
            positive_labels = (
                list(sock_labels) if sock_labels is not None else [1] * len(sock_points)
            )
            sock_points = list(sock_points) + list(sock_negative_points)
            sock_labels = positive_labels + [0] * len(sock_negative_points)
        if leg_negative_points:
            positive_labels = (
                list(leg_labels) if leg_labels is not None else [1] * len(leg_points)
            )
            leg_points = list(leg_points) + list(leg_negative_points)
            leg_labels = positive_labels + [0] * len(leg_negative_points)
        metadata["prompt_points"] = {
            "sock": list(map(list, sock_points)),
            "leg": list(map(list, leg_points)),
            "sock_labels": None if sock_labels is None else list(sock_labels),
            "leg_labels": None if leg_labels is None else list(leg_labels),
        }
        metadata["scenario_application"] = application
        metadata["diagnostics"] = observation["diagnostics"]
        metadata["torque_source"] = observation.get("torque_source", "unavailable")
        metadata["torque_available"] = bool(
            observation.get("torque_available", False)
        )
        metadata["external_torque_source"] = observation.get(
            "external_torque_source", "unavailable"
        )
        video_camera = observation.get("recording_camera")
        video_camera_source = (
            "recording_camera" if video_camera is not None else "inference_camera"
        )
        if video_camera is None:
            video_camera = observation["camera"]
        scene = config["scene"]
        metadata["video_camera"] = {
            "source": video_camera_source,
            "position": (
                recording_camera_frame["position"]
                if recording_camera_frame is not None
                else scene.get("recording_camera_position")
            )
            if video_camera_source == "recording_camera"
            else scene.get("camera_position"),
            "rotation": (
                recording_camera_frame["rotation"]
                if recording_camera_frame is not None
                else scene.get("recording_camera_rotation")
            )
            if video_camera_source == "recording_camera"
            else scene.get("camera_rotation"),
            "crop_xywh": settings.get("recording_crop_xywh"),
            "opening_frame": recording_camera_frame,
        }
        with EpisodeWriter(episode, joints.names, metadata) as writer:
            Image.fromarray(observation["camera"]["rgb"].astype("uint8"), "RGB").save(
                episode / "prompt_frame.png"
            )
            video = _open_video(
                video_path,
                video_camera["rgb"].shape,
                float(settings.get("rate_hz", 5.0)),
            )
            predicted_file = (episode / "predicted_action.csv").open(
                "w", newline="", encoding="utf-8"
            )
            applied_file = (episode / "applied_action.csv").open(
                "w", newline="", encoding="utf-8"
            )
            predicted_writer = csv.writer(predicted_file)
            applied_writer = csv.writer(applied_file)
            try:
                renderer_masks = _renderer_masks(observation["camera"])
                last_renderer_masks = renderer_masks
                initialize_kwargs = {
                    "sock_points": sock_points,
                    "leg_points": leg_points,
                    "sock_labels": sock_labels,
                    "leg_labels": leg_labels,
                }
                if renderer_masks is not None:
                    initialize_kwargs["renderer_masks"] = renderer_masks
                perceived = perception.initialize(
                    observation["camera"]["rgb"], **initialize_kwargs
                )
                for frame in range(steps):
                    prediction = policy.step(
                        rgb=observation["camera"]["rgb"],
                        sock_depth=perceived.sock_depth,
                        leg_depth=perceived.leg_depth,
                        angle=observation["angle"],
                        torque=observation["torque"],
                        step_index=frame,
                    )
                    predicted_writer.writerow(prediction["action"].tolist())
                    command_action = np.asarray(
                        prediction["action"], dtype=float
                    ).copy()
                    if reference_actions is not None and reference_blend > 0:
                        reference_action = _reference_action_at_frame(
                            reference_actions,
                            frame=frame,
                            steps=steps,
                            hold_steps=reference_hold,
                            interpolation=reference_interpolation,
                        )
                        command_action[:18] = (
                            (1 - reference_blend) * command_action[:18]
                            + reference_blend * reference_action
                        )
                    applied = environment.command(
                        command_action, observation["angle"]
                    )
                    environment.advance_physics(physics_steps_per_action - 1)
                    if (
                        cartesian_reference is not None
                        and (frame + 1) % reference_hold == 0
                    ):
                        progress = (frame + 1) / steps
                        distance = reference_pull * progress
                        alignment = environment.move_grippers_to_targets(
                            cartesian_reference["left"]
                            + cartesian_reference["direction"] * distance,
                            cartesian_reference["right"]
                            + cartesian_reference["direction"] * distance,
                        )
                        if not alignment["ok"]:
                            raise RuntimeError(
                                "reference Cartesian projection failed: "
                                + json.dumps(alignment)
                            )
                    observation = environment.observe()
                    renderer_masks = _renderer_masks(observation["camera"])
                    if renderer_masks is not None:
                        last_renderer_masks = renderer_masks
                    elif (
                        settings.get("use_renderer_masks", False)
                        and settings.get("hold_last_renderer_masks", False)
                        and last_renderer_masks is not None
                    ):
                        renderer_masks = last_renderer_masks
                    perceived = (
                        perception.track(
                            observation["camera"]["rgb"],
                            renderer_masks=renderer_masks,
                        )
                        if renderer_masks is not None
                        else perception.track(observation["camera"]["rgb"])
                    )
                    quality_by_frame.append(perceived.quality)
                    coverage_by_frame.append(
                        _measured_coverage(environment, observation["camera"])
                    )
                    grasp_report = _grasp_frame_report(observation, config)
                    grasp_quality_by_frame.append(grasp_report)
                    cloth_quality_by_frame.append(
                        _cloth_frame_report(
                            observation,
                            config,
                            cloth_following_baseline,
                        )
                    )
                    dressing_report = dict(observation.get("dressing_qa", {}) or {})
                    dressing_quality_by_frame.append(dressing_report)
                    rigid_qa = dict(
                        observation.get("diagnostics", {}).get(
                            "robot_human_rigid_collision_qa", {}
                        )
                    )
                    rigid_collision_qa_by_frame.append(rigid_qa)
                    human_chair_lock_qa_by_frame.append(
                        dict(
                            observation.get("diagnostics", {}).get(
                                "human_chair_lock_qa", {}
                            )
                            or {}
                        )
                    )
                    foot_contact_ids_by_frame.append(
                        sorted(
                            {
                                int(item.get("collider_id", -1))
                                for item in observation.get("contact_force", ())
                                if 2101
                                <= int(item.get("collider_id", -1))
                                <= 2105
                            }
                        )
                    )
                    _save_mask_overlay(
                        episode / "mask_overlays" / f"{frame}.png",
                        observation["camera"]["rgb"],
                        perceived.sock_mask,
                        perceived.leg_mask,
                    )
                    writer.append(
                        rgb=observation["camera"]["rgb"],
                        sock_mask=perceived.sock_mask,
                        leg_mask=perceived.leg_mask,
                        camera_depth=perceived.depth,
                        angle=observation["angle"],
                        torque=observation["torque"],
                        external_torque=observation["external_torque"],
                    )
                    video_camera = observation.get("recording_camera")
                    if video_camera is None:
                        video_camera = observation["camera"]
                    _write_video_frame(
                        video,
                        video_camera["rgb"],
                        frame,
                        crop_xywh=settings.get("recording_crop_xywh"),
                    )
                    if cartesian_reference is not None:
                        applied = np.asarray(observation["angle"], dtype=float)
                    applied_writer.writerow(applied.tolist())
                    predicted_file.flush()
                    applied_file.flush()
                    frames += 1
                    if (
                        config["rcareworld"].get("profile") == "custom_player"
                        and bool(
                            settings.get("abort_on_grasp_qa_failure", True)
                        )
                        and not grasp_report["ok"]
                    ):
                        raise RuntimeError(
                            "continuous bimanual grasp failed: "
                            + json.dumps(grasp_report, sort_keys=True)
                        )
                    maximum_penetration = float(
                        config.get("dressing_player", {}).get(
                            "maximum_robot_human_penetration_m", float("inf")
                        )
                    )
                    allow_ignored_rigid_pairs = bool(
                        config.get("dressing_player", {}).get(
                            "allow_ignored_robot_human_collision_pairs", False
                        )
                    )
                    if (
                        (
                            int(rigid_qa.get("ignored_pair_count", 0)) > 0
                            and not allow_ignored_rigid_pairs
                        )
                        or float(rigid_qa.get("maximum_penetration_m", 0.0))
                        > maximum_penetration
                    ):
                        raise RuntimeError(
                            "robot/human rigid collision QA failed: "
                            + json.dumps(rigid_qa, sort_keys=True)
                        )
                    maximum_cloth_foot_penetration = float(
                        config.get("dressing_player", {}).get(
                            "maximum_cloth_foot_penetration_m", float("inf")
                        )
                    )
                    if (
                        bool(
                            config.get("dressing_player", {}).get(
                                "abort_on_cloth_foot_qa_failure", False
                            )
                        )
                        and (
                            not bool(dressing_report.get("valid", False))
                            or float(
                                dressing_report.get(
                                    "maximum_cloth_foot_penetration_m",
                                    float("inf"),
                                )
                            )
                            > maximum_cloth_foot_penetration
                        )
                    ):
                        raise RuntimeError(
                            "cloth/foot dressing QA failed: "
                            + json.dumps(dressing_report, sort_keys=True)
                        )
            except (RuntimeError, ValueError) as error:
                stop_reason = f"fail_closed: {error}"
            finally:
                predicted_file.close()
                applied_file.close()
                video.release()
                final_task_success = _task_success(
                    observation,
                    quality_by_frame,
                    coverage_by_frame,
                    config,
                    application=application,
                    grasp_quality=grasp_quality_by_frame,
                    cloth_quality=cloth_quality_by_frame,
                    dressing_quality=dressing_quality_by_frame,
                    foot_contact_ids_by_frame=foot_contact_ids_by_frame,
                    rigid_collision_qa_by_frame=rigid_collision_qa_by_frame,
                    human_chair_lock_qa_by_frame=human_chair_lock_qa_by_frame,
                )
                writer.update_metadata(
                    {
                        "stop_reason": stop_reason,
                        "frames_inferred": frames,
                        "video": str(video_path),
                        "perception_quality_by_frame": quality_by_frame,
                        "coverage_by_frame": coverage_by_frame,
                        "grasp_quality_by_frame": grasp_quality_by_frame,
                        "cloth_quality_by_frame": cloth_quality_by_frame,
                        "dressing_quality_by_frame": dressing_quality_by_frame,
                        "foot_contact_ids_by_frame": foot_contact_ids_by_frame,
                        "rigid_collision_qa_by_frame": rigid_collision_qa_by_frame,
                        "human_chair_lock_qa_by_frame": (
                            human_chair_lock_qa_by_frame
                        ),
                        "task_success": final_task_success,
                    }
                )
    manifest = output_root / "dataset_phase4.yaml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        yaml.safe_dump(
            {
                config["dataset"]["name"]: {
                    "train": {
                        episode.name: {"start": 0, "end": frames}
                    },
                    "test": {},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return {
        "ok": stop_reason == "max_steps" and frames == steps,
        "task_success": bool(final_task_success.get("success", False)),
        "episode": str(episode),
        "manifest": str(manifest),
        "video": str(video_path),
        "frames": frames,
        "stop_reason": stop_reason,
    }


def _renderer_masks(camera: Mapping) -> Optional[dict]:
    sock = camera.get("sock_mask")
    leg = camera.get("leg_mask")
    if sock is None or leg is None:
        return None
    sock = np.asarray(sock, dtype=bool)
    leg = np.asarray(leg, dtype=bool)
    if (
        sock.shape != leg.shape
        or not sock.any()
        or not leg.any()
        or sock.all()
        or leg.all()
        or np.array_equal(sock, leg)
    ):
        return None
    return {"sock": sock, "leg": leg}


def _mask_centroid_point(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(rows):
        raise ValueError("renderer mask does not contain a promptable pixel")
    center = np.asarray([columns.mean(), rows.mean()])
    distances = (columns - center[0]) ** 2 + (rows - center[1]) ** 2
    index = int(np.argmin(distances))
    return [int(columns[index]), int(rows[index])]


def _save_mask_overlay(
    path: Path, rgb: np.ndarray, sock_mask: np.ndarray, leg_mask: np.ndarray
) -> None:
    image = np.asarray(rgb, dtype=np.uint8).copy()
    sock = np.asarray(sock_mask, dtype=bool)
    leg = np.asarray(leg_mask, dtype=bool)
    image[sock] = (0.55 * image[sock] + 0.45 * np.array([255, 64, 64])).astype(np.uint8)
    image[leg] = (0.55 * image[leg] + 0.45 * np.array([64, 255, 64])).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, "RGB").save(path)


def _measured_coverage(environment, camera: Mapping) -> Optional[float]:
    measure = getattr(environment, "measured_coverage", None)
    return measure(camera) if callable(measure) else None


def _reference_action_at_frame(
    actions: np.ndarray,
    *,
    frame: int,
    steps: int,
    hold_steps: int,
    interpolation: str,
) -> np.ndarray:
    if interpolation == "hold" or len(actions) == 1:
        return actions[min(frame // hold_steps, len(actions) - 1)].copy()
    if interpolation != "linear":
        raise ValueError(f"unsupported reference action interpolation: {interpolation}")
    if steps <= 1:
        return actions[-1].copy()
    position = frame * (len(actions) - 1) / (steps - 1)
    lower = int(np.floor(position))
    upper = min(lower + 1, len(actions) - 1)
    weight = position - lower
    return (1.0 - weight) * actions[lower] + weight * actions[upper]


def _grasp_frame_report(observation: Mapping, config: Mapping) -> dict:
    diagnostics = observation.get("diagnostics", {})
    states = diagnostics.get("grasp_state", ())
    attached_sides = {
        str(item.get("side"))
        for item in states
        if item.get("attached", False)
    }
    geometry = diagnostics.get("scene_geometry", {})
    target_to_edge = {}
    for side in ("left", "right"):
        grasp = geometry.get(f"{side}_grasp_position")
        edge = geometry.get(f"{side}_opening_edge")
        if grasp is None or edge is None:
            continue
        grasp_array = np.asarray(grasp, dtype=float)
        edge_array = np.asarray(edge, dtype=float)
        if (
            grasp_array.shape == (3,)
            and edge_array.shape == (3,)
            and np.all(np.isfinite(grasp_array))
            and np.all(np.isfinite(edge_array))
        ):
            target_to_edge[side] = float(np.linalg.norm(grasp_array - edge_array))
    constraint_errors = {
        str(item.get("side")): float(item["constraint_error"])
        for item in states
        if item.get("attached", False)
        and item.get("constraint_error") is not None
        and np.isfinite(float(item["constraint_error"]))
    }
    errors = (
        constraint_errors
        if len(constraint_errors) == 2
        else target_to_edge
    )
    required = int(config.get("dressing_player", {}).get("required_grippers", 2))
    maximum_error = float(
        config["inference"].get("maximum_grasp_edge_error_m", 0.03)
    )
    available = len(states) > 0 and len(errors) == 2
    opening_span = geometry.get("opening_span_m")
    if opening_span is None and len(errors) == 2:
        opening_span = float(
            np.linalg.norm(
                np.asarray(geometry["right_opening_edge"], dtype=float)
                - np.asarray(geometry["left_opening_edge"], dtype=float)
            )
        )
    if opening_span is not None:
        opening_span = float(opening_span)
    return {
        "available": available,
        "attached_grippers": len(attached_sides),
        "attached_sides": sorted(attached_sides),
        "edge_errors_m": errors,
        "target_to_edge_distances_m": target_to_edge,
        "maximum_edge_error_m": max(errors.values()) if errors else None,
        "opening_span_m": opening_span,
        "threshold_m": maximum_error,
        "ok": (
            available
            and len(attached_sides) >= required
            and max(errors.values()) <= maximum_error
        ),
    }


def _cloth_following_state(observation: Mapping, config: Mapping) -> Optional[dict]:
    particles = np.asarray(observation.get("cloth", {}).get("particles", ()), dtype=float)
    radial_segments = int(config["scenario"]["sock"]["radial_segments"])
    closed_toe = bool(config["scenario"]["sock"].get("closed_toe", False))
    toe_count = radial_segments + (1 if closed_toe else 0)
    geometry = observation.get("diagnostics", {}).get("scene_geometry", {})
    left = np.asarray(geometry.get("left_grasp_position", ()), dtype=float)
    right = np.asarray(geometry.get("right_grasp_position", ()), dtype=float)
    if (
        particles.ndim != 2
        or particles.shape[1:] != (3,)
        or particles.shape[0] < radial_segments + toe_count
        or left.shape != (3,)
        or right.shape != (3,)
        or not np.all(np.isfinite(particles))
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return None
    return {
        "grasp_midpoint": (0.5 * (left + right)),
        "opening_center": particles[:radial_segments].mean(axis=0),
        "toe_center": particles[-toe_count:].mean(axis=0),
    }


def _cloth_frame_report(
    observation: Mapping,
    config: Mapping,
    baseline: Optional[Mapping],
) -> dict:
    obi_settings = config["obi"].get(
        "expected", config["obi"].get("requested", {})
    )
    stretch = SockDressingEnv.cloth_radius_qa(
        observation.get("cloth", {}),
        radial_segments=int(config["scenario"]["sock"]["radial_segments"]),
        maximum_circumferential_stretch=float(
            obi_settings.get("maximum_circumferential_stretch", 1.5)
        ),
    )
    current = _cloth_following_state(observation, config)
    settings = config["inference"]
    minimum_displacement = float(
        settings.get("minimum_follow_displacement_m", 0.02)
    )
    maximum_error = float(
        settings.get("maximum_distal_follow_error_m", 0.05)
    )
    minimum_ratio = float(
        settings.get("minimum_distal_follow_ratio", 0.5)
    )
    if baseline is None or current is None:
        return {
            "available": False,
            "ok": False,
            "stretch": stretch,
            "maximum_distal_follow_error_m": maximum_error,
            "minimum_distal_follow_ratio": minimum_ratio,
        }
    grasp_delta = np.asarray(current["grasp_midpoint"]) - np.asarray(
        baseline["grasp_midpoint"]
    )
    toe_delta = np.asarray(current["toe_center"]) - np.asarray(
        baseline["toe_center"]
    )
    opening_delta = np.asarray(current["opening_center"]) - np.asarray(
        baseline["opening_center"]
    )
    grasp_displacement = float(np.linalg.norm(grasp_delta))
    toe_displacement = float(np.linalg.norm(toe_delta))
    # Compare the distal toe with the cloth cuff itself. Scene-level gripper
    # transforms can differ from the pin frame under robot articulation, while
    # opening-to-toe motion directly measures whether the sock follows its
    # grasped end.
    follow_reference = opening_delta
    follow_error = float(np.linalg.norm(toe_delta - follow_reference))
    follow_displacement = float(np.linalg.norm(follow_reference))
    follow_ratio = (
        toe_displacement / follow_displacement
        if follow_displacement >= minimum_displacement
        else None
    )
    foot_contact = any(
        2101 <= int(item.get("collider_id", -1)) <= 2105
        for item in observation.get("contact_force", ())
    )
    following_ok = (
        True
        if follow_ratio is None
        else (
            foot_contact
            or (
                follow_error <= maximum_error
                and follow_ratio >= minimum_ratio
            )
        )
    )
    return {
        "available": True,
        "ok": bool(stretch.get("passes", False)) and following_ok,
        "stretch": stretch,
        "grasp_midpoint_displacement_m": grasp_displacement,
        "opening_center_displacement_m": float(np.linalg.norm(opening_delta)),
        "toe_center_displacement_m": toe_displacement,
        "distal_follow_error_m": follow_error,
        "distal_follow_ratio": follow_ratio,
        "following_ok": following_ok,
        "foot_contact_allows_distal_anchoring": foot_contact,
        "minimum_follow_displacement_m": minimum_displacement,
        "maximum_distal_follow_error_m": maximum_error,
        "minimum_distal_follow_ratio": minimum_ratio,
    }


def _task_success(
    observation: Mapping,
    quality: Sequence[Mapping],
    coverage: Sequence[Optional[float]],
    config: Mapping,
    *,
    application: Optional[Mapping] = None,
    grasp_quality: Optional[Sequence[Mapping]] = None,
    cloth_quality: Optional[Sequence[Mapping]] = None,
    dressing_quality: Optional[Sequence[Mapping]] = None,
    foot_contact_ids_by_frame: Optional[Sequence[Sequence[int]]] = None,
    rigid_collision_qa_by_frame: Optional[Sequence[Mapping]] = None,
    human_chair_lock_qa_by_frame: Optional[Sequence[Mapping]] = None,
) -> dict:
    diagnostics = observation.get("diagnostics", {})
    verified = [
        item for item in diagnostics.get("grasp_attachments", []) if item.get("verified")
    ]
    required = int(config.get("dressing_player", {}).get("required_grippers", 2))
    semantic_ok = bool(quality and all(item.get("ok", False) for item in quality))
    measured = [value for value in coverage if value is not None]
    gain = measured[-1] - measured[0] if len(measured) >= 2 else None
    minimum_gain = float(
        config.get("dressing_player", {}).get("minimum_coverage_gain", 0.10)
    )
    coverage_ok = gain is not None and gain >= minimum_gain
    pose_ok = bool(
        (application or {}).get("initial_pose_contract", {}).get("ok", False)
    )
    obi_settings = config["obi"].get(
        "expected", config["obi"].get("requested", {})
    )
    maximum_stretch = float(
        obi_settings.get("maximum_circumferential_stretch", 1.5)
    )
    stretch = SockDressingEnv.cloth_radius_qa(
        observation.get("cloth", {}),
        radial_segments=int(config["scenario"]["sock"]["radial_segments"]),
        maximum_circumferential_stretch=maximum_stretch,
    )
    stretch_ok = bool(stretch.get("passes", False))
    cloth_quality = list(cloth_quality or ())
    continuous_cloth_required = (
        config["rcareworld"].get("profile") == "custom_player"
    )
    continuous_stretch_ok = bool(
        cloth_quality
        and all(
            item.get("stretch", {}).get("passes", False)
            for item in cloth_quality
        )
    )
    distal_follow_ok = bool(
        cloth_quality
        and all(item.get("following_ok", False) for item in cloth_quality)
    )
    if not continuous_cloth_required or not cloth_quality:
        continuous_stretch_ok = stretch_ok
        distal_follow_ok = True
    grasp_quality = list(grasp_quality or ())
    continuous_grasp_required = (
        config["rcareworld"].get("profile") == "custom_player"
    )
    continuous_grasp_ok = bool(
        grasp_quality and all(item.get("ok", False) for item in grasp_quality)
    )
    if not continuous_grasp_required:
        continuous_grasp_ok = len(verified) >= required
    maximum_opening_span = float(
        config["scene"]
        .get("initial_pose_contract", {})
        .get("maximum_opening_span_m", 0.12)
    )
    opening_spans = [
        float(item["opening_span_m"])
        for item in grasp_quality
        if item.get("opening_span_m") is not None
    ]
    continuous_opening_span_ok = bool(
        opening_spans and max(opening_spans) <= maximum_opening_span
    )
    maximum_bounds_span = float(
        config["scene"]
        .get("initial_pose_contract", {})
        .get("maximum_cloth_bounds_span_m", 0.35)
    )
    measured_bounds_spans = []
    for item in cloth_quality:
        bounds = item.get("stretch", {}).get("bounds_world", {})
        minimum = np.asarray(bounds.get("minimum", ()), dtype=float)
        maximum = np.asarray(bounds.get("maximum", ()), dtype=float)
        if minimum.shape == (3,) and maximum.shape == (3,):
            measured_bounds_spans.append(maximum - minimum)
    continuous_bounds_ok = bool(
        measured_bounds_spans
        and all(
            np.all(span <= maximum_bounds_span)
            for span in measured_bounds_spans
        )
    )
    if not continuous_cloth_required:
        continuous_opening_span_ok = True
        continuous_bounds_ok = True
    minimum_attached = (
        min(int(item.get("attached_grippers", 0)) for item in grasp_quality)
        if grasp_quality
        else 0
    )
    measured_edge_errors = [
        float(item["maximum_edge_error_m"])
        for item in grasp_quality
        if item.get("maximum_edge_error_m") is not None
    ]
    foot_contact_ids = sorted(
        {
            int(collider_id)
            for frame_ids in (foot_contact_ids_by_frame or ())
            for collider_id in frame_ids
            if 2101 <= int(collider_id) <= 2105
        }
    )
    foot_contact_required = bool(
        config.get("dressing_player", {}).get("require_foot_contact", False)
    )
    foot_contact_ok = bool(foot_contact_ids) or not foot_contact_required
    if (
        foot_contact_ok
        and coverage_ok
        and continuous_stretch_ok
        and continuous_opening_span_ok
        and continuous_bounds_ok
    ):
        # During successful dressing the closed sock toe can remain anchored
        # on the foot while the cuff advances. In that state cuff/toe rigid
        # co-translation is neither expected nor a useful anti-tearing gate;
        # continuous structural stretch and bounds provide that safety check.
        distal_follow_ok = True
    rigid_collision_qa = list(rigid_collision_qa_by_frame or ())
    maximum_allowed_penetration = float(
        config.get("dressing_player", {}).get(
            "maximum_robot_human_penetration_m", float("inf")
        )
    )
    maximum_penetration = max(
        (
            float(item.get("maximum_penetration_m", 0.0))
            for item in rigid_collision_qa
        ),
        default=None,
    )
    rigid_collision_ok = bool(rigid_collision_qa) and all(
        (
            int(item.get("ignored_pair_count", 0)) == 0
            or bool(
                config.get("dressing_player", {}).get(
                    "allow_ignored_robot_human_collision_pairs", False
                )
            )
        )
        and float(item.get("maximum_penetration_m", 0.0))
        <= maximum_allowed_penetration
        for item in rigid_collision_qa
    )
    if config["rcareworld"].get("profile") != "custom_player":
        rigid_collision_ok = True
    dressing_settings = config.get("dressing_player", {})
    dressing_quality = list(dressing_quality or ())
    hold_frames = max(
        1, int(dressing_settings.get("final_coverage_hold_frames", 3))
    )
    final_dressing_frames = dressing_quality[-hold_frames:]
    minimum_surface_containment = float(
        dressing_settings.get("minimum_final_surface_containment", 0.90)
    )
    minimum_section_containment = float(
        dressing_settings.get("minimum_final_section_containment", 0.80)
    )
    minimum_cuff_progress = float(
        dressing_settings.get("minimum_cuff_progress_toward_ankle_m", 0.0)
    )
    maximum_cuff_reverse = float(
        dressing_settings.get("maximum_cuff_reverse_m", 0.002)
    )
    maximum_cuff_beyond_toe = float(
        dressing_settings.get("maximum_cuff_beyond_distal_toe_m", 0.002)
    )
    maximum_cloth_foot_penetration = float(
        dressing_settings.get("maximum_cloth_foot_penetration_m", 0.002)
    )
    dressing_observations_ok = bool(dressing_quality) and all(
        bool(item.get("valid", False)) for item in dressing_quality
    )
    final_surface_containment_ok = (
        len(final_dressing_frames) == hold_frames
        and all(
            float(item.get("surface_containment_ratio", -1.0))
            >= minimum_surface_containment
            for item in final_dressing_frames
        )
    )
    final_section_containment_ok = (
        len(final_dressing_frames) == hold_frames
        and all(
            bool(item.get("sections"))
            and all(
                bool(section.get("valid", False))
                and float(section.get("containment_ratio", -1.0))
                >= minimum_section_containment
                for section in item.get("sections", ())
            )
            for item in final_dressing_frames
        )
    )
    cuff_progress_ok = bool(dressing_quality) and (
        float(
            dressing_quality[-1].get(
                "cuff_progress_toward_ankle_m", float("-inf")
            )
        )
        >= minimum_cuff_progress
        and all(
            float(item.get("maximum_cuff_reverse_step_m", float("inf")))
            <= maximum_cuff_reverse
            and float(item.get("cuff_beyond_distal_toe_m", float("inf")))
            <= maximum_cuff_beyond_toe
            for item in dressing_quality
        )
    )
    cloth_foot_penetration_ok = bool(dressing_quality) and all(
        float(
            item.get("maximum_cloth_foot_penetration_m", float("inf"))
        )
        <= maximum_cloth_foot_penetration
        for item in dressing_quality
    )
    if config["rcareworld"].get("profile") != "custom_player":
        dressing_observations_ok = True
        final_surface_containment_ok = True
        final_section_containment_ok = True
        cuff_progress_ok = True
        cloth_foot_penetration_ok = True
    lock_qa = list(human_chair_lock_qa_by_frame or ())
    maximum_lock_drift = float(
        config["scene"]
        .get("initial_pose_contract", {})
        .get("maximum_lock_drift_m", 0.002)
    )
    lock_fields = (
        "right_toe_drift_m",
        "chair_drift_m",
        "human_root_drift_m",
        "human_anchor_drift_m",
    )
    human_chair_lock_ok = bool(lock_qa) and all(
        bool(item.get("valid", False))
        and all(
            np.isfinite(float(item.get(name, float("inf"))))
            and float(item.get(name, float("inf"))) <= maximum_lock_drift
            for name in lock_fields
        )
        for item in lock_qa
    )
    if not config["scene"].get("initial_pose_contract", {}).get(
        "lock_human_and_chair", False
    ):
        human_chair_lock_ok = True
    success_gates = {
        "semantic_masks_ok": semantic_ok,
        "verified_grippers_ok": len(verified) >= required,
        "continuous_grasp_ok": continuous_grasp_ok,
        "continuous_opening_span_ok": continuous_opening_span_ok,
        "continuous_bounds_ok": continuous_bounds_ok,
        "coverage_gain_ok": coverage_ok,
        "initial_pose_ok": pose_ok,
        "final_stretch_ok": stretch_ok,
        "continuous_stretch_ok": continuous_stretch_ok,
        "distal_follow_ok": distal_follow_ok,
        "foot_contact_ok": foot_contact_ok,
        "rigid_collision_ok": rigid_collision_ok,
        "dressing_observations_ok": dressing_observations_ok,
        "final_surface_containment_ok": final_surface_containment_ok,
        "final_section_containment_ok": final_section_containment_ok,
        "cuff_progress_ok": cuff_progress_ok,
        "cloth_foot_penetration_ok": cloth_foot_penetration_ok,
        "human_chair_lock_ok": human_chair_lock_ok,
    }
    return {
        "success": all(success_gates.values()),
        "failed_gates": [
            name for name, passed in success_gates.items() if not passed
        ],
        "semantic_masks_ok": semantic_ok,
        "verified_grippers": len(verified),
        "required_grippers": required,
        "continuous_grasp_ok": continuous_grasp_ok,
        "minimum_attached_grippers": minimum_attached,
        "maximum_grasp_edge_error_m": (
            max(measured_edge_errors) if measured_edge_errors else None
        ),
        "maximum_opening_span_m": (
            max(opening_spans) if opening_spans else None
        ),
        "maximum_allowed_opening_span_m": maximum_opening_span,
        "continuous_opening_span_ok": continuous_opening_span_ok,
        "maximum_cloth_bounds_span_m": (
            np.max(np.stack(measured_bounds_spans), axis=0).tolist()
            if measured_bounds_spans
            else None
        ),
        "maximum_allowed_cloth_bounds_span_m": maximum_bounds_span,
        "continuous_bounds_ok": continuous_bounds_ok,
        "coverage_gain": gain,
        "minimum_coverage_gain": minimum_gain,
        "coverage_ok": coverage_ok,
        "initial_pose_ok": pose_ok,
        "foot_contact_required": foot_contact_required,
        "foot_contact_ok": foot_contact_ok,
        "foot_contact_collider_ids": foot_contact_ids,
        "rigid_collision_ok": rigid_collision_ok,
        "maximum_robot_human_penetration_m": maximum_penetration,
        "maximum_allowed_robot_human_penetration_m": maximum_allowed_penetration,
        "dressing_observations_ok": dressing_observations_ok,
        "final_coverage_hold_frames": hold_frames,
        "final_surface_containment_ratio": (
            float(final_dressing_frames[-1].get("surface_containment_ratio"))
            if final_dressing_frames
            and final_dressing_frames[-1].get("surface_containment_ratio")
            is not None
            else None
        ),
        "minimum_final_surface_containment": minimum_surface_containment,
        "final_surface_containment_ok": final_surface_containment_ok,
        "final_section_containment_ok": final_section_containment_ok,
        "minimum_final_section_containment": minimum_section_containment,
        "cuff_progress_toward_ankle_m": (
            float(
                dressing_quality[-1].get("cuff_progress_toward_ankle_m")
            )
            if dressing_quality
            and dressing_quality[-1].get("cuff_progress_toward_ankle_m")
            is not None
            else None
        ),
        "minimum_cuff_progress_toward_ankle_m": minimum_cuff_progress,
        "maximum_cuff_reverse_m": max(
            (
                float(item.get("maximum_cuff_reverse_step_m", 0.0))
                for item in dressing_quality
            ),
            default=None,
        ),
        "maximum_allowed_cuff_reverse_m": maximum_cuff_reverse,
        "maximum_cuff_beyond_distal_toe_m": max(
            (
                float(item.get("cuff_beyond_distal_toe_m", 0.0))
                for item in dressing_quality
            ),
            default=None,
        ),
        "maximum_allowed_cuff_beyond_distal_toe_m": maximum_cuff_beyond_toe,
        "cuff_progress_ok": cuff_progress_ok,
        "maximum_cloth_foot_penetration_m": max(
            (
                float(item.get("maximum_cloth_foot_penetration_m", 0.0))
                for item in dressing_quality
            ),
            default=None,
        ),
        "maximum_allowed_cloth_foot_penetration_m": (
            maximum_cloth_foot_penetration
        ),
        "cloth_foot_penetration_ok": cloth_foot_penetration_ok,
        "human_chair_lock_ok": human_chair_lock_ok,
        "maximum_allowed_lock_drift_m": maximum_lock_drift,
        "maximum_human_chair_lock_drift_m": max(
            (
                float(item.get(name, 0.0))
                for item in lock_qa
                for name in lock_fields
            ),
            default=None,
        ),
        "cloth_qa": stretch,
        "stretch_ok": stretch_ok,
        "continuous_stretch_ok": continuous_stretch_ok,
        "distal_follow_ok": distal_follow_ok,
        "maximum_distal_follow_error_m": max(
            (
                float(item["distal_follow_error_m"])
                for item in cloth_quality
                if item.get("distal_follow_error_m") is not None
            ),
            default=None,
        ),
        "note": (
            None
            if gain is not None
            else "coverage gain is unavailable unless non-degenerate foot masks are provided"
        ),
    }


def _episode_name() -> str:
    from datetime import datetime, timezone

    return "phase4_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _open_video(path: Path, frame_shape: tuple, fps: float):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("video recording requires opencv-python") from error
    height, width = frame_shape[:2]
    video = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not video.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    return video


def _write_video_frame(
    video,
    rgb: np.ndarray,
    frame: int,
    *,
    crop_xywh: Optional[Sequence[int]] = None,
) -> None:
    import cv2

    image = np.asarray(rgb, dtype=np.uint8)
    if crop_xywh is not None:
        if len(crop_xywh) != 4:
            raise ValueError("recording_crop_xywh must contain x, y, width, height")
        x, y, width, height = map(int, crop_xywh)
        source_height, source_width = image.shape[:2]
        if (
            min(x, y) < 0
            or min(width, height) < 1
            or x + width > source_width
            or y + height > source_height
        ):
            raise ValueError("recording_crop_xywh is outside the recording frame")
        image = cv2.resize(
            image[y : y + height, x : x + width],
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        )
    bgr = image[..., ::-1].copy()
    cv2.putText(
        bgr,
        f"AIREC inference frame {frame}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    video.write(bgr)
