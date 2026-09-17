from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

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
    max_steps: Optional[int] = None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    environment_factory: Callable = SockDressingEnv,
    perception_factory: Callable = SAMDepthPerception,
    policy_factory: Callable = SAMDAMSARNNPolicy,
) -> dict:
    settings = config["inference"]
    steps = int(max_steps if max_steps is not None else settings.get("max_steps", 250))
    if steps < 1:
        raise ValueError("max_steps must be positive")
    joints = JointMap.from_config(config)
    scenario = scenario_from_config(config)
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
        metadata["prompt_points"] = {
            "sock": list(map(list, sock_points)),
            "leg": list(map(list, leg_points)),
        }
        metadata["scenario_application"] = application
        metadata["diagnostics"] = observation["diagnostics"]
        with EpisodeWriter(episode, joints.names, metadata) as writer:
            Image.fromarray(observation["camera"]["rgb"].astype("uint8"), "RGB").save(
                episode / "prompt_frame.png"
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
                perceived = perception.initialize(
                    observation["camera"]["rgb"],
                    sock_points=sock_points,
                    leg_points=leg_points,
                    sock_labels=sock_labels,
                    leg_labels=leg_labels,
                )
                for frame in range(steps):
                    if frame:
                        perceived = perception.track(observation["camera"]["rgb"])
                    quality_by_frame.append(perceived.quality)
                    writer.append(
                        rgb=observation["camera"]["rgb"],
                        sock_mask=perceived.sock_mask,
                        leg_mask=perceived.leg_mask,
                        camera_depth=perceived.depth,
                        angle=observation["angle"],
                        torque=observation["torque"],
                        external_torque=observation["external_torque"],
                    )
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
                writer.update_metadata(
                    {
                        "stop_reason": stop_reason,
                        "frames_inferred": frames,
                        "perception_quality_by_frame": quality_by_frame,
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
        "frames": frames,
        "stop_reason": stop_reason,
    }


def _episode_name() -> str:
    from datetime import datetime, timezone

    return "phase4_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
