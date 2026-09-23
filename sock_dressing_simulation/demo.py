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
    foot_contact_ids_by_frame = []
    rigid_collision_qa_by_frame = []
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
                    rigid_qa = dict(
                        observation.get("diagnostics", {}).get(
                            "robot_human_rigid_collision_qa", {}
                        )
                    )
                    rigid_collision_qa_by_frame.append(rigid_qa)
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
                    if (
                        int(rigid_qa.get("ignored_pair_count", 0)) > 0
                        or float(rigid_qa.get("maximum_penetration_m", 0.0))
                        > maximum_penetration
                    ):
                        raise RuntimeError(
                            "robot/human rigid collision QA failed: "
                            + json.dumps(rigid_qa, sort_keys=True)
                        )
            except (RuntimeError, ValueError) as error:
                stop_reason = f"fail_closed: {error}"
            finally:
                predicted_file.close()
                applied_file.close()
                video.release()
                writer.update_metadata(
                    {
                        "stop_reason": stop_reason,
                        "frames_inferred": frames,
                        "video": str(video_path),
                        "perception_quality_by_frame": quality_by_frame,
                        "coverage_by_frame": coverage_by_frame,
                        "grasp_quality_by_frame": grasp_quality_by_frame,
                        "foot_contact_ids_by_frame": foot_contact_ids_by_frame,
                        "rigid_collision_qa_by_frame": rigid_collision_qa_by_frame,
                        "task_success": _task_success(
                            observation,
                            quality_by_frame,
                            coverage_by_frame,
                            config,
                            application=application,
                            grasp_quality=grasp_quality_by_frame,
                            foot_contact_ids_by_frame=foot_contact_ids_by_frame,
                            rigid_collision_qa_by_frame=rigid_collision_qa_by_frame,
                        ),
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
        "opening_area_m2": geometry.get("opening_area_m2"),
        "opening_convex_hull_area_m2": geometry.get(
            "opening_convex_hull_area_m2"
        ),
        "opening_convexity_ratio": geometry.get("opening_convexity_ratio"),
        "opening_major_diameter_m": geometry.get(
            "opening_major_diameter_m"
        ),
        "opening_minor_diameter_m": geometry.get(
            "opening_minor_diameter_m"
        ),
        "threshold_m": maximum_error,
        "ok": (
            available
            and len(attached_sides) >= required
            and max(errors.values()) <= maximum_error
        ),
    }


def _task_success(
    observation: Mapping,
    quality: Sequence[Mapping],
    coverage: Sequence[Optional[float]],
    config: Mapping,
    *,
    application: Optional[Mapping] = None,
    grasp_quality: Optional[Sequence[Mapping]] = None,
    foot_contact_ids_by_frame: Optional[Sequence[Sequence[int]]] = None,
    rigid_collision_qa_by_frame: Optional[Sequence[Mapping]] = None,
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
    stretch = SockDressingEnv.cloth_radius_qa(
        observation.get("cloth", {}),
        radial_segments=int(config["scenario"]["sock"]["radial_segments"]),
    )
    stretch_ok = bool(stretch.get("passes", False))
    grasp_quality = list(grasp_quality or ())
    continuous_grasp_required = (
        config["rcareworld"].get("profile") == "custom_player"
    )
    continuous_grasp_ok = bool(
        grasp_quality and all(item.get("ok", False) for item in grasp_quality)
    )
    if not continuous_grasp_required:
        continuous_grasp_ok = len(verified) >= required
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
        int(item.get("ignored_pair_count", 0)) == 0
        and float(item.get("maximum_penetration_m", 0.0))
        <= maximum_allowed_penetration
        for item in rigid_collision_qa
    )
    if config["rcareworld"].get("profile") != "custom_player":
        rigid_collision_ok = True
    return {
        "success": (
            semantic_ok
            and len(verified) >= required
            and continuous_grasp_ok
            and coverage_ok
            and pose_ok
            and stretch_ok
            and foot_contact_ok
            and rigid_collision_ok
        ),
        "semantic_masks_ok": semantic_ok,
        "verified_grippers": len(verified),
        "required_grippers": required,
        "continuous_grasp_ok": continuous_grasp_ok,
        "minimum_attached_grippers": minimum_attached,
        "maximum_grasp_edge_error_m": (
            max(measured_edge_errors) if measured_edge_errors else None
        ),
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
        "cloth_qa": stretch,
        "stretch_ok": stretch_ok,
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
