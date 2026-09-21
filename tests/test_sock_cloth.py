import numpy as np
import pytest
from pathlib import Path

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.quality import assess_observation_quality
from sock_dressing_simulation.sock_cloth import (
    PROTOCOL_VERSION,
    ClothContact,
    GraspState,
    SockClothAttr,
    validate_configuration,
)


class FakeEnvironment:
    def __init__(self):
        self.messages = []

    def _send_instance_data(self, *args):
        self.messages.append(args)


def test_sock_cloth_commands_match_unity_contract():
    env = FakeEnvironment()
    cloth = SockClothAttr(env, 1200)
    cloth.request_particles()
    cloth.request_particle_velocities()
    cloth.configure(
        stretch_compliance=0.0005,
        bend_compliance=0.005,
        particle_radius_m=0.008,
        particle_mass_kg=0.005,
        collision_margin_m=0.002,
        friction=0.5,
        self_collision=True,
        substeps=4,
        solver_iterations=8,
    )
    cloth.set_grasp_targets(2201, 2202)
    cloth.configure_right_leg_colliders(2000)
    cloth.configure_mask_proxy_cameras()
    cloth.request_configuration()
    cloth.request_registered_colliders()
    cloth.grasp("left", 0.03)
    cloth.release("right")
    cloth.request_grasp_state()
    cloth.request_attached_particle_indices("left")
    cloth.request_grasp_constraint_error("right")
    cloth.request_contacts()
    cloth.request_coverage()
    cloth.reset()
    assert env.messages == [
        (1200, "GetParticles"),
        (1200, "GetParticleVelocities"),
        (1200, "ConfigureSock", 0.0005, 0.005, 0.008, 0.005, 0.002, 0.5, True, 4, 8),
        (1200, "SetGraspTargets", 2201, 2202),
        (1200, "ConfigureRightLegColliders", 2000),
        (1200, "ConfigureMaskProxyCameras", 31),
        (1200, "GetClothConfiguration"),
        (1200, "GetRegisteredObiColliders"),
        (1200, "GraspLeft", 0.03),
        (1200, "ReleaseRight"),
        (1200, "GetGraspState"),
        (1200, "GetAttachedParticleIndices", "left"),
        (1200, "GetGraspForceOrConstraintError", "right"),
        (1200, "GetClothContacts"),
        (1200, "GetCoverageObservations"),
        (1200, "ResetSock"),
    ]


def test_configuration_is_fail_closed():
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "stretch_compliance": 0.0005,
        "bend_compliance": 0.005,
        "self_collision": True,
    }
    actual = {
        **expected,
        "obi_version": "7.0",
        "timestep_s": 0.02,
        "substeps": 4,
        "solver_iterations": 8,
        "particle_radius_m": 0.008,
    }
    assert validate_configuration(actual, expected)["ok"]
    actual["stretch_compliance"] = 0.5
    report = validate_configuration(actual, expected)
    assert not report["ok"]
    assert "stretch_compliance" in report["mismatches"]


def test_typed_observations_reject_invalid_values():
    state = GraspState.from_mapping(
        {
            "side": "left",
            "attached": True,
            "particle_indices": [1, 3],
            "constraint_error": 0.002,
        }
    )
    assert state.particle_indices == (1, 3)
    contact = ClothContact.from_mapping(
        {
            "particle_index": 1,
            "collider_id": 2103,
            "position": [0, 1, 2],
            "normal": [0, 1, 0],
            "penetration_m": 0.001,
            "force_proxy": 2.0,
        }
    )
    assert contact.collider_id == 2103
    with pytest.raises(ValueError):
        GraspState.from_mapping(
            {
                "side": "left",
                "attached": True,
                "particle_indices": [],
                "constraint_error": 0,
            }
        )


def test_particle_arrays_are_finite_n_by_three():
    cloth = SockClothAttr(
        FakeEnvironment(),
        1200,
        {"particles": [[0, 1, 2]], "particle_velocities": [[0, 0, 0]]},
    )
    assert cloth.particles().shape == (1, 3)
    assert cloth.particle_velocities().dtype == np.float32
    cloth.data["particles"] = [[np.nan, 0, 0]]
    with pytest.raises(ValueError):
        cloth.particles()


def test_custom_profile_and_quality_require_player_verification():
    config = load_config(Path("config/custom_player.yaml"))
    assert config["rcareworld"]["profile"] == "custom_player"
    image = np.zeros((3, 4), dtype=np.uint8)
    rgb0 = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb0[0, 0] = 255
    rgb1 = rgb0.copy()
    rgb1[0, 1] = 255
    sock = image.copy()
    sock[0, 0] = 255
    leg = image.copy()
    leg[0, :2] = 255
    cameras = [
        {"rgb": rgb0, "camera_depth": rgb0[:, :, 0], "sock_mask": sock, "leg_mask": leg},
        {"rgb": rgb1, "camera_depth": rgb1[:, :, 0], "sock_mask": sock, "leg_mask": leg},
    ]
    diagnostics = {
        "obi_contract": {"ok": True},
        "robot_obi_collider_verified": True,
        "registered_obi_colliders": [
            {"region": name, "enabled": True}
            for name in ("calf", "ankle", "heel", "forefoot", "toes")
        ],
    }
    report = assess_observation_quality(
        cameras,
        expected_width=4,
        expected_height=3,
        exact_foot_colliders=True,
        player_diagnostics=diagnostics,
    )
    assert report["learning_ready"]
    diagnostics["obi_contract"]["ok"] = False
    report = assess_observation_quality(
        cameras,
        expected_width=4,
        expected_height=3,
        exact_foot_colliders=True,
        player_diagnostics=diagnostics,
    )
    assert not report["learning_ready"]
    assert "player_contract_verified" in report["failed_checks"]
