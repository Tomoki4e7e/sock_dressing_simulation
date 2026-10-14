#!/usr/bin/env python3
"""Isolated Dry-AIREC/toe contact probe; not a dressing-success trial."""
import argparse
import json
from pathlib import Path

import numpy as np

from sock_dressing_simulation.config import load_config, resolve_package_path
from sock_dressing_simulation.environment import SockDressingEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/rigid_contact_feedback.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["rcareworld"].update(graphics=False, port=5019)
    # This probe measures rigid-body sensing; cameras and preparation alignment
    # are exercised by the existing dressing trials.
    config["scene"]["camera_id"] = None
    config["scene"]["recording_camera_id"] = None
    config["scene"]["side_recording_camera_id"] = None
    generated = resolve_package_path(config["assets"]["output_dir"])
    trace = []
    with SockDressingEnv(config) as env:
        env.load(generated / "dry_airec3_gripper_v2_rcareworld.urdf", generated / "sock.obj")
        _, right = env._request_grasp_target_positions()
        original = list(env._request_scene_geometry().right_toe_position)
        target = env.robot_signals()["angle"]
        config["scene"]["robot_direct_joint_control"] = False
        env.command(target, target)
        env.robot.GetJointInverseDynamicsForce()
        env._env.step()
        baseline_motor = env.robot_signals()["torque"].copy()
        hit = None
        for offset in ([0, 0, 0], [0, 0, .03], [0, 0, -.03], [.03, 0, 0], [-.03, 0, 0]):
            env.sock_cloth.set_task_right_toe_position_articulated(right + offset)
            env._env.step()
            for _ in range(30):
                env.robot.GetJointInverseDynamicsForce()
                env._env.step()
                signals = env.robot_signals()
                contacts = [c for c in signals["rigid_contact"] if 2101 <= int(c["collider_id"]) <= 2105]
                row = {"step": signals["rigid_contact_step"], "contacts": contacts,
                       "inverse_dynamics_step": signals["inverse_dynamics_step"],
                       "external_torque": signals["external_torque"].tolist(),
                       "torque": signals["torque"].tolist(),
                       "angle": signals["angle"].tolist(),
                       "source": signals["external_torque_source"],
                       "available": signals["rigid_contact_force_available"]}
                trace.append(row)
                if contacts and np.linalg.norm(signals["external_torque"]) > .01:
                    hit = row
            if hit:
                break
        # Preserve an exact sample while requesting inverse dynamics read-only.
        before = env.robot_signals()
        env.robot.GetJointInverseDynamicsForce()
        env._env.step(simulate=False)
        after = env.robot_signals()
        read_only_ok = before["rigid_contact_step"] == after["rigid_contact_step"] and np.array_equal(
            before["external_torque"], after["external_torque"])
        env.sock_cloth.set_task_right_toe_position(original)
        env._env.step()
        synchronized = all(row["step"] == row["inverse_dynamics_step"] for row in trace)
        motor_response = hit is not None and float(np.linalg.norm(np.asarray(hit["torque"]) - baseline_motor)) > .01
        report = {"ok": hit is not None and read_only_ok and synchronized and motor_response,
                  "human_contact_sample": hit, "read_only_preserves_sample": read_only_ok,
                  "inverse_dynamics_synchronized": synchronized,
                  "motor_effort_responds": motor_response,
                  "joint_displacement_rad_or_m": (np.asarray(hit["angle"]) - target).tolist() if hit else None,
                  "physics_dt_s": after["rigid_contact_step_seconds"], "trace": trace}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "trace"}, indent=2))
        if not report["ok"]:
            raise RuntimeError("Dry-AIREC/toe contact signal validation failed")


if __name__ == "__main__":
    main()
