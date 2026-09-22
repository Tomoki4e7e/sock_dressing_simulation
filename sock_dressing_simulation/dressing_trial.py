from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .environment import SockDressingEnv
from .episode import EpisodeWriter
from .joints import JointMap
from .scenario import scenario_from_config


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _video_writer(path: Path, image: np.ndarray, fps: float):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("dressing trial video requires opencv-python") from error
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to create dressing trial video: {path}")
    return writer


def _write_frame(writer, image: np.ndarray, index: int) -> None:
    import cv2

    bgr = np.asarray(image, dtype=np.uint8)[..., ::-1].copy()
    cv2.putText(
        bgr,
        f"verified dressing trial {index}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    writer.write(bgr)


def _grasp_summary(observation: Mapping[str, Any]) -> Dict[str, bool]:
    values = observation.get("cloth", {}).get("grasp_state", ())
    return {
        str(value.get("side")): bool(value.get("attached", False))
        for value in values
    }


def _trial_score(report: Mapping[str, Any]) -> tuple:
    return (
        int(bool(report.get("physical_sock_dressing_success"))),
        int(report.get("minimum_attached_grippers", 0)),
        float(report.get("coverage_gain") or -1.0),
        -float(report.get("maximum_tracking_error_m") or 1e9),
        -float(report.get("maximum_stretch") or 1e9),
    )


def run_dressing_trial(
    config: Mapping[str, Any],
    *,
    prepared: Mapping[str, Any],
    output_root: Path,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    config = deepcopy(dict(config))
    settings = config.get("dressing_trial", {})
    config["rcareworld"]["graphics"] = True
    scenario = scenario_from_config(config, seed=seed)
    output = Path(output_root) / f"dressing_trial_{_timestamp()}"
    output.mkdir(parents=True, exist_ok=False)

    trace = []
    coverage_values = []
    stretch_values = []
    attached_counts = []
    tracking_errors = []
    training_frames = []
    writer = None
    with SockDressingEnv(config) as environment:
        environment.load(
            Path(prepared["runtime_urdf"]),
            Path(prepared["sock_obj"]),
            initial_joints=scenario.initial_joints,
        )
        application = environment.apply_scenario(scenario)
        initial_geometry = environment._request_scene_geometry()
        start = {
            "left": np.asarray(initial_geometry.left_grasp_position, dtype=float),
            "right": np.asarray(initial_geometry.right_grasp_position, dtype=float),
        }
        direction = (
            np.asarray(initial_geometry.right_toe_position, dtype=float)
            - np.asarray(initial_geometry.opening_center, dtype=float)
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            direction = -np.asarray(initial_geometry.opening_normal, dtype=float)
            norm = float(np.linalg.norm(direction))
        direction /= norm

        pull = float(settings.get("axial_pull_m", 0.18))
        waypoints = int(settings.get("waypoint_count", 18))
        hold_steps = int(settings.get("hold_steps", 5))
        trajectory_mode = str(settings.get("trajectory_mode", "arm_jacobian"))
        robot_start = np.asarray(config["scene"]["robot_position"], dtype=float)
        toe_start = np.asarray(initial_geometry.right_toe_position, dtype=float)
        if pull <= 0 or waypoints < 2 or hold_steps < 0:
            raise ValueError("invalid dressing trial trajectory configuration")

        for index, distance in enumerate(np.linspace(0.0, pull, waypoints)):
            target_left = start["left"] + direction * distance
            target_right = start["right"] + direction * distance
            if trajectory_mode == "foot_insertion":
                environment.sock_cloth.set_task_right_toe_position(
                    toe_start - direction * distance
                )
                for _ in range(3):
                    environment._env.step()
                geometry = environment._request_scene_geometry()
                errors = {
                    "left": float(
                        np.linalg.norm(
                            start["left"]
                            - np.asarray(geometry.left_grasp_position, dtype=float)
                        )
                    ),
                    "right": float(
                        np.linalg.norm(
                            start["right"]
                            - np.asarray(geometry.right_grasp_position, dtype=float)
                        )
                    ),
                }
                tolerance = float(
                    config["scene"]["grasp_alignment"].get("tolerance_m", 0.025)
                )
                alignment = {
                    "ok": max(errors.values()) <= tolerance,
                    "tolerance_m": tolerance,
                    "iterations": 0,
                    "errors_m": errors,
                    "trace": [],
                }
            elif trajectory_mode == "base_translation":
                environment.robot.SetTransform(
                    position=(robot_start + direction * distance).tolist(),
                    rotation=list(config["scene"].get("robot_rotation", [0, 0, 0])),
                    scale=[1.0, 1.0, 1.0],
                )
                for _ in range(3):
                    environment._env.step()
                geometry = environment._request_scene_geometry()
                errors = {
                    "left": float(
                        np.linalg.norm(
                            target_left
                            - np.asarray(geometry.left_grasp_position, dtype=float)
                        )
                    ),
                    "right": float(
                        np.linalg.norm(
                            target_right
                            - np.asarray(geometry.right_grasp_position, dtype=float)
                        )
                    ),
                }
                tolerance = float(
                    config["scene"]["grasp_alignment"].get("tolerance_m", 0.025)
                )
                alignment = {
                    "ok": max(errors.values()) <= tolerance,
                    "tolerance_m": tolerance,
                    "iterations": 0,
                    "errors_m": errors,
                    "trace": [],
                }
            else:
                alignment = environment.move_grippers_to_targets(
                    target_left, target_right
                )
            for _ in range(hold_steps):
                environment._env.step()
            observation = environment.observe()
            camera = observation.get("recording_camera") or observation.get("camera")
            if camera is not None:
                if writer is None:
                    writer = _video_writer(
                        output / "demo.mp4",
                        camera["rgb"],
                        float(config["inference"].get("rate_hz", 5.0)),
                    )
                _write_frame(writer, camera["rgb"], index)

            coverage = (
                environment.measured_coverage(observation["camera"])
                if observation.get("camera") is not None
                else None
            )
            coverage_values.append(coverage)
            grasp = _grasp_summary(observation)
            attached = sum(grasp.values())
            attached_counts.append(attached)
            tracking = max(alignment["errors_m"].values())
            tracking_errors.append(tracking)
            stretch = environment.cloth_radius_qa(
                observation["cloth"],
                radial_segments=scenario.sock_mesh.radial_segments,
            )
            stretch_value = stretch.get("circumferential_stretch_proxy")
            stretch_values.append(stretch_value)
            trace.append(
                {
                    "waypoint": index,
                    "distance_m": float(distance),
                    "alignment": alignment,
                    "coverage": coverage,
                    "grasp_attached": grasp,
                    "cloth_qa": stretch,
                    "contact_count": len(observation.get("contact_force") or ()),
                }
            )
            if observation.get("camera") is not None:
                training_frames.append(
                    {
                        "rgb": observation["camera"]["rgb"].copy(),
                        "sock_mask": observation["camera"]["sock_mask"].copy(),
                        "leg_mask": observation["camera"]["leg_mask"].copy(),
                        "camera_depth": observation["camera"]["camera_depth"].copy(),
                        "angle": observation["angle"].copy(),
                        "torque": observation["torque"].copy(),
                        "external_torque": observation["external_torque"].copy(),
                    }
                )
            if not alignment["ok"] or attached < 2:
                break

    if writer is not None:
        writer.release()

    measured = [value for value in coverage_values if value is not None]
    coverage_gain = measured[-1] - measured[0] if len(measured) >= 2 else None
    maximum_stretch = max(
        (float(value) for value in stretch_values if value is not None),
        default=None,
    )
    maximum_tracking = max(tracking_errors, default=None)
    minimum_attached = min(attached_counts, default=0)
    minimum_gain = float(settings.get("minimum_coverage_gain", 0.10))
    tracking_limit = float(settings.get("maximum_tracking_error_m", 0.025))
    report = {
        "physical_sock_dressing_success": bool(
            len(trace) == int(settings.get("waypoint_count", 18))
            and application.get("initial_pose_contract", {}).get("ok", False)
            and minimum_attached >= int(settings.get("required_grippers", 2))
            and coverage_gain is not None
            and coverage_gain >= minimum_gain
            and maximum_tracking is not None
            and maximum_tracking <= tracking_limit
            and maximum_stretch is not None
            and maximum_stretch <= 1.5
        ),
        "output": str(output),
        "application": application,
        "parameters": {
            "seed": scenario.seed,
            "axial_pull_m": pull,
            "waypoint_count": waypoints,
            "hold_steps": hold_steps,
            "trajectory_mode": trajectory_mode,
            "obi": deepcopy(config.get("obi", {}).get("expected", {})),
            "grasp": deepcopy(config["scene"].get("grasp_anchors", {})),
            "alignment": deepcopy(config["scene"].get("grasp_alignment", {})),
        },
        "coverage_by_waypoint": coverage_values,
        "coverage_gain": coverage_gain,
        "minimum_coverage_gain": minimum_gain,
        "minimum_attached_grippers": minimum_attached,
        "maximum_tracking_error_m": maximum_tracking,
        "maximum_stretch": maximum_stretch,
        "trace": trace,
    }
    expert_trajectory_complete = bool(
        len(trace) == int(settings.get("waypoint_count", 18))
        and minimum_attached >= int(settings.get("required_grippers", 2))
        and coverage_gain is not None
        and coverage_gain >= float(
            settings.get("expert_minimum_coverage_gain", minimum_gain)
        )
        and maximum_tracking is not None
        and maximum_tracking <= tracking_limit
    )
    if expert_trajectory_complete and training_frames:
        training_episode = output / "training_episode"
        with EpisodeWriter(
            training_episode,
            JointMap.from_config(config).names,
            {
                "mode": "robot-arm-expert",
                "seed": scenario.seed,
                "source_report": str(output / "report.json"),
            },
        ) as episode:
            for frame in training_frames:
                episode.append(**frame)
            episode.update_metadata(
                {
                    "physical_sock_dressing_success": report[
                        "physical_sock_dressing_success"
                    ],
                    "expert_trajectory_complete": True,
                    "coverage_gain": coverage_gain,
                    "maximum_stretch": maximum_stretch,
                }
            )
        report["training_episode"] = str(training_episode)
        report["training_frames"] = len(training_frames)
    else:
        report["training_episode"] = None
        report["training_frames"] = 0
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_staged_tuning(
    config: Mapping[str, Any],
    *,
    prepared: Mapping[str, Any],
    output_root: Path,
    max_trials: int = 0,
) -> Dict[str, Any]:
    """Tune one parameter family at a time to avoid an intractable full grid."""
    best_config = deepcopy(dict(config))
    tuning = best_config.get("dressing_trial", {}).get("tuning", {})
    candidates = []
    paths = {
        "grasp_max_distance_m": ("scene", "grasp_anchors", "max_distance_m"),
        "friction": ("obi", "expected", "friction"),
        "grasp_linear_compliance": (
            "obi",
            "expected",
            "grasp_linear_compliance",
        ),
        "grasp_break_threshold": ("obi", "expected", "grasp_break_threshold"),
        "slip_constraint_error_m": (
            "obi",
            "expected",
            "slip_constraint_error_m",
        ),
        "slip_opening_span_m": ("obi", "expected", "slip_opening_span_m"),
        "axial_pull_m": ("dressing_trial", "axial_pull_m"),
    }
    for name, values in tuning.items():
        if name not in paths:
            continue
        family = []
        for value in values:
            if max_trials and len(candidates) >= max_trials:
                break
            trial_config = deepcopy(best_config)
            node = trial_config
            for key in paths[name][:-1]:
                node = node[key]
            node[paths[name][-1]] = value
            report = run_dressing_trial(
                trial_config,
                prepared=prepared,
                output_root=output_root / name,
            )
            item = {"parameter": name, "value": value, "report": report}
            candidates.append(item)
            family.append((item, trial_config))
        if family:
            winner, best_config = max(
                family, key=lambda item: _trial_score(item[0]["report"])
            )
            if winner["report"]["physical_sock_dressing_success"]:
                break
        if max_trials and len(candidates) >= max_trials:
            break

    result = {
        "ok": bool(candidates),
        "physical_sock_dressing_success": any(
            item["report"]["physical_sock_dressing_success"] for item in candidates
        ),
        "trial_count": len(candidates),
        "best": (
            max(candidates, key=lambda item: _trial_score(item["report"]))
            if candidates
            else None
        ),
        "trials": candidates,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tuning_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
