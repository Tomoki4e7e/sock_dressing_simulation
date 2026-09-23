#!/usr/bin/env python3
"""Run non-rendering live Obi/grasp acceptance and save compact evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from sock_dressing_simulation.config import load_config, resolve_package_path
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config


def particle_snapshot(environment):
    environment.sock_cloth.request_particles()
    environment.sock_cloth.request_particle_velocities()
    environment._env.step()
    return (
        environment.sock_cloth.particles().copy(),
        environment.sock_cloth.particle_velocities().copy(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grasp-distance", type=float, default=0.1)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--insertion-step-m", type=float, default=0.005)
    parser.add_argument("--insertion-steps", type=int, default=20)
    parser.add_argument("--insertion-lift-m", type=float, default=0.10)
    parser.add_argument("--pull-step-m", type=float, default=0.002)
    parser.add_argument("--pull-steps", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    config["rcareworld"]["graphics"] = False
    generated = resolve_package_path(config["assets"]["output_dir"])
    urdf = generated / "dry_airec3_gripper_v2_rcareworld.urdf"
    sock = generated / "sock.obj"

    with SockDressingEnv(config) as environment:
        scenario = scenario_from_config(config)
        environment.load(urdf, sock, initial_joints=scenario.initial_joints)
        application = environment.apply_scenario(scenario)
        initial, _ = particle_snapshot(environment)
        foot_contact_ids = set()
        for _ in range(args.hold_steps):
            environment.sock_cloth.request_contacts()
            environment._env.step()
            foot_contact_ids.update(
                int(contact.collider_id)
                for contact in environment.sock_cloth.contacts()
                if 2101 <= int(contact.collider_id) <= 2105
            )
        environment.sock_cloth.request_grasp_state()
        environment.sock_cloth.request_configuration()
        environment._env.step()
        held = [state.__dict__ for state in environment.sock_cloth.grasp_states()]
        settled, _ = particle_snapshot(environment)
        attached_indices = {
            int(index)
            for state in held
            for index in state["particle_indices"]
        }
        body_indices = np.asarray(
            [
                index
                for index in range(initial.shape[0])
                if index not in attached_indices
            ],
            dtype=int,
        )
        body_displacement = (
            settled[body_indices] - initial[body_indices]
            if body_indices.size and settled.shape == initial.shape
            else np.empty((0, 3), dtype=float)
        )
        maximum_unpinned_displacement = (
            float(np.linalg.norm(body_displacement, axis=1).max())
            if body_displacement.size
            else None
        )
        maximum_unpinned_downward_displacement = (
            float(np.maximum(0.0, -body_displacement[:, 1]).max())
            if body_displacement.size
            else None
        )
        initial_contract = application["initial_pose_contract"]
        left_position = np.asarray(
            initial_contract["left_grasp_position"], dtype=float
        )
        right_position = np.asarray(
            initial_contract["right_grasp_position"], dtype=float
        )
        opening_position = np.asarray(
            initial_contract["opening_center"], dtype=float
        )
        toe_position = np.asarray(
            initial_contract["right_toe_position"], dtype=float
        )
        insertion_direction = np.asarray(
            initial_contract["opening_outward_normal"], dtype=float
        )
        insertion_direction /= np.linalg.norm(insertion_direction)
        toe_direction = toe_position - opening_position
        toe_direction /= np.linalg.norm(toe_direction)
        if float(np.dot(insertion_direction, toe_direction)) < float(
            initial_contract["opening_to_toe_alignment_min"]
        ):
            raise RuntimeError("sock opening is not oriented toward the right toe")
        insertion_trace = []
        for index in range(args.insertion_steps):
            progress = float(index + 1) / float(args.insertion_steps)
            offset = (
                insertion_direction * args.insertion_step_m * float(index + 1)
                + np.asarray([0.0, args.insertion_lift_m * progress, 0.0])
            )
            environment.sock_cloth.set_grasp_target_position(
                "left", left_position + offset
            )
            environment.sock_cloth.set_grasp_target_position(
                "right", right_position + offset
            )
            environment._env.step()
            environment.sock_cloth.request_contacts()
            environment.sock_cloth.request_grasp_state()
            environment._env.step()
            states = {
                state.side: state for state in environment.sock_cloth.grasp_states()
            }
            contacts = environment.sock_cloth.contacts()
            frame_foot_ids = sorted(
                {
                    int(contact.collider_id)
                    for contact in contacts
                    if 2101 <= int(contact.collider_id) <= 2105
                }
            )
            foot_contact_ids.update(frame_foot_ids)
            insertion_trace.append(
                {
                    "step": index + 1,
                    "attached_grippers": sum(
                        int(state.attached) for state in states.values()
                    ),
                    "foot_contact_ids": frame_foot_ids,
                }
            )
        insertion_offset = (
            insertion_direction * args.insertion_step_m * args.insertion_steps
            + np.asarray([0.0, args.insertion_lift_m, 0.0])
        )
        left_position += insertion_offset
        right_position += insertion_offset
        before_pull, _ = particle_snapshot(environment)
        before_pull_qa = environment.cloth_radius_qa(
            {
                "particles": before_pull,
                "particle_edges": environment.sock_cloth.data.get(
                    "particle_edges", ()
                ),
                "particle_rest_edge_lengths": environment.sock_cloth.data.get(
                    "particle_rest_edge_lengths", ()
                ),
                "grasp_state": held,
            },
            radial_segments=scenario.sock_mesh.radial_segments,
        )
        pull_direction = right_position - left_position
        pull_direction /= np.linalg.norm(pull_direction)
        pull_trace = []
        slipped = False
        slipped_side = None
        for index in range(args.pull_steps):
            environment.sock_cloth.set_grasp_target_position(
                "right",
                right_position + pull_direction * args.pull_step_m * (index + 1),
            )
            environment._env.step()
            environment.sock_cloth.request_grasp_state()
            environment.sock_cloth.request_particles()
            environment.sock_cloth.request_scene_geometry()
            environment.sock_cloth.request_contacts()
            environment._env.step()
            states = {
                state.side: state for state in environment.sock_cloth.grasp_states()
            }
            qa = environment.cloth_radius_qa(
                {
                    "particles": environment.sock_cloth.particles(),
                    "particle_edges": environment.sock_cloth.data.get(
                        "particle_edges", ()
                    ),
                    "particle_rest_edge_lengths": environment.sock_cloth.data.get(
                        "particle_rest_edge_lengths", ()
                    ),
                    "grasp_state": [
                        state.__dict__ for state in states.values()
                    ],
                },
                radial_segments=scenario.sock_mesh.radial_segments,
            )
            geometry = environment.sock_cloth.scene_geometry()
            foot_contact_ids.update(
                int(contact.collider_id)
                for contact in environment.sock_cloth.contacts()
                if 2101 <= int(contact.collider_id) <= 2105
            )
            anchor_span = float(
                np.linalg.norm(
                    np.asarray(geometry.right_grasp_position)
                    - np.asarray(geometry.left_grasp_position)
                )
            )
            pull_trace.append(
                {
                    "step": index + 1,
                    "right": states["right"].__dict__,
                    "circumferential_stretch_proxy": qa.get(
                        "circumferential_stretch_proxy"
                    ),
                    "grasp_anchor_span_m": anchor_span,
                    "opening_span_stretch_proxy": (
                        anchor_span / (2.0 * scenario.sock_mesh.radius_m)
                    ),
                }
            )
            released_by_tension = [
                side
                for side, state in states.items()
                if not state.attached and state.release_reason == "over_tension"
            ]
            if released_by_tension:
                slipped = True
                slipped_side = released_by_tension[0]
                break

        final, velocity = particle_snapshot(environment)
        environment.sock_cloth.request_contacts()
        environment._env.step()
        released = [
            state.__dict__ for state in environment.sock_cloth.grasp_states()
        ]
        contacts = environment.sock_cloth.contacts()
        stretch_values = [
            float(item["circumferential_stretch_proxy"])
            for item in pull_trace
            if item["circumferential_stretch_proxy"] is not None
        ]
        maximum_particle_ring_stretch = max(stretch_values, default=None)
        opening_stretch_values = [
            float(item["opening_span_stretch_proxy"]) for item in pull_trace
        ]
        maximum_stretch = max(opening_stretch_values, default=None)
        configuration = environment.sock_cloth.configuration()
        pin_limit = int(configuration["maximum_grasp_particles_per_side"])
        localized_pins = all(
            len(state["particle_indices"]) <= pin_limit for state in held
        ) and int(configuration["non_cuff_grasp_particle_count"]) == 0
        cuff_insertion_ok = (
            float(initial_contract["left_cuff_insertion_depth_m"])
            >= float(initial_contract["minimum_cuff_insertion_depth_m"])
            and float(initial_contract["right_cuff_insertion_depth_m"])
            >= float(initial_contract["minimum_cuff_insertion_depth_m"])
        )

        displacement = (
            np.linalg.norm(final - initial, axis=1)
            if initial.shape == final.shape and initial.size
            else np.asarray([], dtype=float)
        )
        speed = (
            np.linalg.norm(velocity, axis=1)
            if velocity.ndim == 2 and velocity.shape[1:] == (3,)
            else np.asarray([], dtype=float)
        )
        report = {
            "ok": bool(
                initial.shape == (800, 3)
                and displacement.size
                and float(displacement.max()) > 0
                and all(state["attached"] for state in held)
                and localized_pins
                and cuff_insertion_ok
                and maximum_unpinned_downward_displacement is not None
                and maximum_unpinned_downward_displacement > 0.001
                and bool(foot_contact_ids)
                and before_pull_qa.get("passes", False)
                and slipped
                and application["initial_pose_contract"]["ok"]
                and maximum_stretch is not None
                and maximum_stretch <= 1.5
                and maximum_particle_ring_stretch is not None
                and maximum_particle_ring_stretch <= 1.5
            ),
            "particle_count": int(initial.shape[0]),
            "maximum_particle_displacement_m": (
                float(displacement.max()) if displacement.size else None
            ),
            "maximum_particle_speed_m_s": float(speed.max()) if speed.size else None,
            "application": application,
            "held_after_settle": held,
            "localized_cuff_pins": localized_pins,
            "cuff_insertion_ok": cuff_insertion_ok,
            "maximum_unpinned_displacement_m": maximum_unpinned_displacement,
            "maximum_unpinned_downward_displacement_m": (
                maximum_unpinned_downward_displacement
            ),
            "insertion_trace": insertion_trace,
            "before_pull_cloth_qa": before_pull_qa,
            "released": released,
            "pull_trace": pull_trace,
            "slipped_due_to_over_tension": slipped,
            "slipped_side": slipped_side,
            "maximum_circumferential_stretch_proxy": maximum_stretch,
            "maximum_particle_ring_stretch_proxy": maximum_particle_ring_stretch,
            "particle_displacement_after_pull_m": (
                float(np.linalg.norm(final - before_pull, axis=1).max())
                if final.shape == before_pull.shape and final.size
                else None
            ),
            "contact_count": len(contacts),
            "contact_collider_ids": sorted(
                {int(contact.collider_id) for contact in contacts}
            ),
            "foot_contact_collider_ids": sorted(foot_contact_ids),
            "grasp_max_distance_m": float(args.grasp_distance),
            "insertion_lift_m": float(args.insertion_lift_m),
            "configuration": configuration,
            "registered_collider_count": len(
                environment.sock_cloth.registered_colliders()
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
