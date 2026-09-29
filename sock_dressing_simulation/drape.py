from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np
from PIL import Image

from .demo import _open_video, _tip_drape_sample, _write_video_frame
from .environment import SockDressingEnv
from .scenario import scenario_from_config


def run_signed_opening_drape(
    config: Mapping,
    *,
    prepared: Mapping,
    output_root: Path,
    rotation_degrees: float,
    rotation_steps: int = 180,
    settle_steps: int = 250,
    side_distance_m: float = 0.65,
    side_reverse: bool = False,
    seed: Optional[int] = None,
    environment_factory: Callable = SockDressingEnv,
) -> dict:
    rotation = float(rotation_degrees)
    if not np.isfinite(rotation) or abs(rotation) > 180 or abs(rotation) < 1e-8:
        raise ValueError(
            "rotation_degrees must be finite, nonzero, and in [-180, 180]"
        )
    if rotation_steps < 1 or settle_steps < 1:
        raise ValueError("rotation_steps and settle_steps must be positive")
    scenario = scenario_from_config(config, seed=seed)
    sign_name = "positive" if rotation > 0 else "negative"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    episode = (
        output_root
        / config["dataset"]["name"]
        / "train"
        / f"drape_{sign_name}_{timestamp}"
    )
    episode.mkdir(parents=True, exist_ok=True)
    overview_path = episode / "drape_overview.mp4"
    side_path = episode / "drape_side.mp4"
    output = Path(config["assets"]["output_dir"])
    if not output.is_absolute():
        from .config import resolve_package_path

        output = resolve_package_path(output)

    with environment_factory(config) as environment:
        environment.load(
            Path(prepared["runtime_urdf"]),
            output / "sock.obj",
            initial_joints=scenario.initial_joints,
        )
        application = environment.apply_scenario(scenario)
        observation = environment.observe()
        initial_geometry = observation["diagnostics"]["scene_geometry"]
        release_y = float(initial_geometry["sock_tip_center"][1])
        environment.release_initial_tip_guidance()

        wait_settings = dict(
            config.get("inference", {}).get("tip_drape_wait", {}) or {}
        )
        maximum_wait = int(wait_settings.get("maximum_steps", 250))
        required_consecutive = int(wait_settings.get("consecutive_steps", 4))
        consecutive = 0
        wait_samples = []
        previous_y = release_y
        for wait_step in range(1, maximum_wait + 1):
            observation = environment.observe()
            sample = _tip_drape_sample(
                wait_step,
                observation,
                config,
                release_y=release_y,
                previous_y=previous_y,
                settings=wait_settings,
            )
            wait_samples.append(sample)
            previous_y = float(sample["sock_tip_world_y_m"])
            consecutive = consecutive + 1 if sample["ok"] else 0
            if consecutive >= required_consecutive:
                break
        else:
            raise RuntimeError(
                "initial drape did not settle before signed rotation: "
                + json.dumps(wait_samples[-1], sort_keys=True)
            )

        environment.arm_opening_body_barrier_predictive_skin(True)
        delta = rotation / rotation_steps
        for _ in range(rotation_steps):
            environment.rotate_grasped_opening_about_span(delta)
            environment.step_physics()

        side_frame = environment.frame_side_camera_on_sock(
            side_distance_m, reverse=side_reverse
        )
        first = environment.observe_drape_cameras()
        fps = 1.0 / float(config["obi"].get("timestep_s", 0.02))
        overview_video = _open_video(
            overview_path, first["overview"]["rgb"].shape, fps
        )
        side_video = _open_video(side_path, first["side"]["rgb"].shape, fps)
        frames = []
        try:
            for frame in range(settle_steps):
                sample = (
                    first
                    if frame == 0
                    else environment.observe_drape_cameras()
                )
                overview_rgb = sample["overview"]["rgb"]
                side_rgb = sample["side"]["rgb"]
                _write_video_frame(
                    overview_video,
                    overview_rgb,
                    frame,
                    crop_xywh=config.get("inference", {}).get(
                        "recording_crop_xywh"
                    ),
                )
                _write_video_frame(side_video, side_rgb, frame)
                geometry = asdict(sample["geometry"])
                frames.append(geometry)
                if frame in {0, settle_steps - 1}:
                    Image.fromarray(
                        overview_rgb.astype("uint8"), "RGB"
                    ).save(episode / f"overview_{frame:04d}.png")
                    Image.fromarray(
                        side_rgb.astype("uint8"), "RGB"
                    ).save(episode / f"side_{frame:04d}.png")
        finally:
            overview_video.release()
            side_video.release()

    maximum_penetration = max(
        item["opening_body_barrier_maximum_penetration_m"]
        for item in frames
    )
    maximum_violations = max(
        item["opening_body_barrier_violation_count"] for item in frames
    )
    final = frames[-1]
    rotation_error = abs(
        final["signed_opening_span_rotation_degrees"] - rotation
    )
    maximum_allowed_penetration = float(
        config.get("inference", {}).get(
            "maximum_opening_body_penetration_m", 0.0005
        )
    )
    ok = bool(
        rotation_error <= 0.01
        and maximum_violations == 0
        and maximum_penetration <= maximum_allowed_penetration
        and final["opening_target_normal_alignment"] >= 0.98
        and final["opening_ring_target_alignment"] >= 0.98
    )
    metadata = {
        "mode": "signed-opening-drape-only",
        "seed": scenario.seed,
        "rotation_degrees": rotation,
        "rotation_steps": rotation_steps,
        "rotation_delta_degrees": delta,
        "settle_steps": settle_steps,
        "policy_inference": False,
        "scenario_application": application,
        "initial_drape_wait_steps": len(wait_samples),
        "initial_drape_final": wait_samples[-1],
        "side_camera": side_frame,
        "maximum_opening_body_penetration_m": maximum_penetration,
        "maximum_opening_body_barrier_violation_count": maximum_violations,
        "final_geometry": final,
        "overview_video": str(overview_path),
        "side_video": str(side_path),
        "ok": ok,
    }
    (episode / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "ok": ok,
        "episode": str(episode),
        "overview_video": str(overview_path),
        "side_video": str(side_path),
        "frames": settle_steps,
        "rotation_degrees": rotation,
        "maximum_opening_body_penetration_m": maximum_penetration,
        "maximum_opening_body_barrier_violation_count": maximum_violations,
    }
