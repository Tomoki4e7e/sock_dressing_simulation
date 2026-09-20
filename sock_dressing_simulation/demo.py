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
    }
    stop_reason = "max_steps"
    frames = 0
    quality_by_frame = []
    coverage_by_frame = []
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
        observation = environment.observe()
        if observation["camera"] is None:
            raise RuntimeError("camera observation is unavailable")
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
        video_camera = observation.get("recording_camera")
        video_camera_source = (
            "recording_camera" if video_camera is not None else "inference_camera"
        )
        if video_camera is None:
            video_camera = observation["camera"]
        scene = config["scene"]
        metadata["video_camera"] = {
            "source": video_camera_source,
            "position": scene.get("recording_camera_position")
            if video_camera_source == "recording_camera"
            else scene.get("camera_position"),
            "rotation": scene.get("recording_camera_rotation")
            if video_camera_source == "recording_camera"
            else scene.get("camera_rotation"),
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
                    if frame:
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
                    _write_video_frame(video, video_camera["rgb"], frame)
                    prediction = policy.step(
                        rgb=observation["camera"]["rgb"],
                        sock_depth=perceived.sock_depth,
                        leg_depth=perceived.leg_depth,
                        angle=observation["angle"],
                        torque=observation["torque"],
                        step_index=frame,
                    )
                    predicted_writer.writerow(prediction["action"].tolist())
                    applied = environment.command(
                        prediction["action"], observation["angle"]
                    )
                    applied_writer.writerow(applied.tolist())
                    predicted_file.flush()
                    applied_file.flush()
                    frames += 1
                    observation = environment.observe()
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
                        "task_success": _task_success(
                            observation, quality_by_frame, coverage_by_frame, config
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


def _task_success(
    observation: Mapping,
    quality: Sequence[Mapping],
    coverage: Sequence[Optional[float]],
    config: Mapping,
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
    return {
        "success": semantic_ok and len(verified) >= required and coverage_ok,
        "semantic_masks_ok": semantic_ok,
        "verified_grippers": len(verified),
        "required_grippers": required,
        "coverage_gain": gain,
        "minimum_coverage_gain": minimum_gain,
        "coverage_ok": coverage_ok,
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


def _write_video_frame(video, rgb: np.ndarray, frame: int) -> None:
    import cv2

    bgr = np.asarray(rgb, dtype=np.uint8)[..., ::-1].copy()
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
