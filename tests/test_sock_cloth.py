import json
import numpy as np
import pytest
from pathlib import Path

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.quality import assess_observation_quality
from sock_dressing_simulation.sock_cloth import (
    PROTOCOL_VERSION,
    ClothContact,
    GraspState,
    SceneGeometry,
    SockClothAttr,
    validate_configuration,
)


class FakeEnvironment:
    def __init__(self):
        self.messages = []

    def _send_instance_data(self, *args):
        self.messages.append(args)


def test_custom_sock_visual_is_two_sided_and_inherits_canonical_materials():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()
    stencil = Path(
        "RCareUnity/Assets/RCareCommon/Library/Bulitin/ObiClothStencil.prefab"
    ).read_text()

    assert "sockVisualMesh.subMeshCount = 2;" in source
    assert "sockVisualMesh.SetTriangles(frontTriangles, 0);" in source
    assert "sockVisualMesh.SetTriangles(backTriangles, 1);" in source
    assert 'new[] { frontMaterial, backMaterial }' in source
    assert '"SockVisualFront"' in source
    assert '"SockVisualBack"' in source
    assert source.count("canonicalMaterials[0]") >= 2
    assert 'material.SetTexture("_MainTex", null)' not in source
    assert "CreateOpeningRimMaterial(" in source
    assert 'material.SetColor("_EmissionColor", color * 0.6f)' in source
    assert "ObiClothStencil canonical materials are unresolved" in source
    assert "cluster.vertexIndices" in source
    assert "sockVisualVertexParticles[vertexIndex] = cluster.index;" in source
    assert 'new GameObject("sock_opening_rim")' in source
    assert '"SockOpeningRim"' in source
    assert "openingRimRenderer.loop = true;" in source
    assert "openingRimRenderer.SetPosition(" in source
    assert "cloth.GetParticleRuntimeIndex(openingParticles[i])" in source
    assert "Mathf.Atan2(" in source
    assert "Vector3.Dot(offset, bitangent)" in source
    assert '"sock_" + side + "_grasp_patch"' in source
    assert "entry.Value.transform.position = GraspParticleCenter(grasp);" in source
    assert "new Color(0.95f, 0.12f, 0.03f, 1)" not in source
    assert "new Color(1.0f, 0.45f, 0.02f, 1)" not in source
    assert "guid: 3862df0f523254bcda0733ce0d63c68a" in stencil
    assert "guid: a5d1f0e373f1848d98dcc85a2607d980" in stencil


def test_custom_player_obeys_canonical_sock_physics_contract():
    config = load_config(Path("config/custom_player.yaml"))
    expected = config["obi"]["expected"]
    contract = json.loads(
        Path(
            "unity/SockDressingPlayer/Assets/StreamingAssets/"
            "SockDressingContract.json"
        ).read_text()
    )["obi"]

    for name, value in contract.items():
        if name == "timestep_s":
            continue
        assert expected[name] == value
    assert config["obi"]["timestep_s"] == contract["timestep_s"]
    assert config["obi"]["substeps"] == contract["substeps"]
    assert config["obi"]["solver_iterations"] == contract["solver_iterations"]
    assert expected["grasp_linear_compliance"] == 0.00005
    assert expected["grasp_rotational_compliance"] == 1000000.0
    assert expected["grasp_break_threshold"] == 20.0
    assert expected["slip_constraint_error_m"] == 0.20
    assert expected["slip_opening_span_m"] == 0.11
    assert expected["slip_consecutive_steps"] == 2
    assert expected["maximum_grasp_particles_per_side"] == 2


def test_sock_opening_alignment_writes_solver_local_particle_positions():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()

    assert "solver.transform.InverseTransformPoint(aligned)" in source
    assert "Vector3.ProjectOnPlane(" in source
    assert "Physics.gravity" in source
    assert "Quaternion.AngleAxis(" in source
    assert "CaptureResetStateIfReady(true)" in source


def test_sock_geometry_reports_particle_derived_hanging_direction():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()
    config = load_config(Path("config/custom_player.yaml"))

    assert "Vector3 sockBodyDirection = ClothCenter() - openingCenter;" in source
    assert "Vector3 openingNormal = OpeningNormal();" in source
    assert "normal += Vector3.Cross(current, next);" in source
    assert '"opening_target_center", openingTargetCenter' in source
    assert '"opening_target_normal", openingTargetNormal' in source
    assert '"opening_outward_normal", -openingNormal' in source
    assert '"opening_to_toe_alignment", openingToToeAlignment' in source
    assert "targetMidpoint - toeTarget" in source
    assert "Physics.gravity," not in source[
        source.index("public void AlignSockOpeningToGraspTargets"):
        source.index("public void SetGraspTargetPosition")
    ]
    assert "Vector3 footOffset = toePosition - openingTargetCenter;" in source
    assert "Vector3.Dot(footOffset, openingTargetNormal)" in source
    assert '"sock_body_direction", sockBodyDirection' in source
    assert '"sock_body_gravity_alignment", sockBodyGravityAlignment' in source
    assert "solver.positions[solverIndex] = aligned;" not in source
    assert "Vector3.Dot(point, axis)" in source
    assert "Physics.IgnoreCollision(robotCollider, humanCollider, true)" in source
    assert "Physics.ComputePenetration(" in source
    assert "IgnoreNonGripperRobotHumanRigidCollisions" in source
    assert "IsGripperCollider(value)" in source
    assert not config["scene"]["ignore_robot_human_rigid_collisions"]
    assert config["scene"]["ignore_non_gripper_robot_human_rigid_collisions"]
    assert config["scene"]["robot_direct_joint_control"]
    assert not config["scene"]["robot_rollout_direct_joint_control"]


def test_grasp_pins_small_inner_cuff_patches_and_leaves_rim_dynamic():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()

    assert ".Take(maximumGraspParticlesPerSide)" in source
    assert "graspRotationalCompliance" in source
    assert '"non_cuff_grasp_particle_count"' in source
    assert "int[] selected = openingParticles" in source
    assert "targetMidpoint - toeTarget" in source
    assert "minimum + cuffInsertionDepth" in source
    assert '"cuff_insertion_depth_m", cuffInsertionDepth' in source
    assert "cloth.tetherConstraintsEnabled = false;" in source
    assert "cloth.volumeConstraintsEnabled = false;" in source
    assert "solver.invMasses[solverIndex] = 1.0f / particleMass;" in source
    assert "leftOpeningEdge = GraspParticleCenter(grasps[\"left\"]);" not in source
    assert "rightOpeningEdge = GraspParticleCenter(grasps[\"right\"]);" not in source
    assert '"opening_span_m", openingSpan' in source
    assert "Release(movingSide, \"over_tension\")" not in source


def test_right_leg_colliders_follow_actual_bones():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()

    assert "SyncRightLegCollidersToBones();" in source
    assert "Transform foot = bones.RightFoot;" in source
    assert "Transform toes = bones.RightToes ?? foot;" in source
    assert "item.SetParent(bone, false);" in source
    assert "toe + new Vector3(0, 0, -0.15f)" not in source


def test_human_task_pose_preserves_rig_bone_lengths():
    pose_source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()
    player_source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Main/PlayerMain.cs"
    ).read_text()

    assert "SolveTwoBoneKnee(" in pose_source
    assert "taskStraightRightLeg" in pose_source
    assert '"right_knee_flexion_degrees"' in pose_source
    assert "visualRightHip += correction;" in pose_source
    assert "visualSeatPosition += correction;" in pose_source
    assert "RotateBoneToward(" in pose_source
    assert "maximumTaskBoneLengthError <= 1e-4f" in pose_source
    assert 'Resources.Load<CanonicalHumanPose>("CanonicalHumanPose")' in pose_source
    assert "PoseTwoBoneChain(" not in pose_source
    assert "chair_intersection_vertex_fraction" in pose_source
    assert "SetBonePosition(" not in pose_source
    assert 'LeftShoulder = exact("left_collar")' in player_source
    assert 'LeftUpperArm = exact("left_shoulder")' in player_source
    assert 'RightShoulder = exact("right_collar")' in player_source
    assert 'RightUpperArm = exact("right_shoulder")' in player_source
    assert 'LeftMiddle1 = exact("left_middle1")' in player_source
    assert 'RightMiddle1 = exact("right_middle1")' in player_source


def test_canonical_human_pose_contains_terminal_bones():
    pose = Path(
        "RCareUnity/Assets/SockDressing/Resources/CanonicalHumanPose.asset"
    ).read_text()
    build_source = Path(
        "RCareUnity/Assets/SockDressing/Editor/SockDressingBuild.cs"
    ).read_text()

    for name in (
        "left_wrist",
        "left_middle1",
        "right_wrist",
        "right_middle1",
        "left_ankle",
        "left_foot",
        "right_ankle",
        "right_foot",
    ):
        assert f"name: {name}" in pose
    assert "clip.SampleAnimation(instance, 0);" in build_source
    assert "AssetDatabase.CreateAsset(pose, CanonicalHumanPosePath);" in build_source


def test_human_task_pose_rejects_nonfinite_plantarflexion():
    cloth = SockClothAttr(FakeEnvironment(), 1200)
    with pytest.raises(ValueError, match="plantarflexion"):
        cloth.configure_human_task_pose(
            [0.0, 0.5, 0.0],
            [0.0, 0.6, 0.7],
            [0.6, 0.1, 0.5],
            float("nan"),
        )


def test_human_task_pose_rejects_nonpositive_seat_scale():
    cloth = SockClothAttr(FakeEnvironment(), 1200)
    with pytest.raises(ValueError, match="seat scale"):
        cloth.configure_human_task_pose(
            [0.0, 0.5, 0.0],
            [0.0, 0.6, 0.7],
            [0.6, 0.0, 0.5],
        )


def test_sock_cloth_commands_match_unity_contract():
    env = FakeEnvironment()
    cloth = SockClothAttr(env, 1200)
    cloth.request_particles()
    cloth.request_particle_velocities()
    cloth.configure(
        stretch_compliance=0.0,
        bend_compliance=0.01,
        stretching_scale=1.0,
        particle_radius_m=0.008,
        particle_mass_kg=0.005,
        collision_margin_m=0.002,
        friction=0.5,
        self_collision=True,
        damping=0.95,
        substeps=8,
        solver_iterations=20,
    )
    cloth.configure_grasp(
        linear_compliance=0.0002,
        rotational_compliance=1000000.0,
        break_threshold=20.0,
        slip_constraint_error_m=0.20,
        slip_opening_span_m=0.11,
        slip_consecutive_steps=2,
        maximum_particles_per_side=2,
        cuff_insertion_depth_m=0.03,
    )
    cloth.set_grasp_targets(2201, 2202)
    cloth.align_grasp_targets_to_opening()
    cloth.clamp_grasp_target_span(0.115)
    cloth.align_sock_opening_to_grasp_targets([0.0, 0.5, 0.6])
    cloth.ignore_robot_human_rigid_collisions(1100)
    cloth.ignore_non_gripper_robot_human_rigid_collisions(1100)
    cloth.configure_right_leg_colliders(2000)
    cloth.translate_human_and_ik([0.1, 0.0, -0.2])
    cloth.freeze_human_right_toe_at([0.0, 0.5, 0.6])
    cloth.configure_human_task_pose(
        [0.0, 0.5, 0.0],
        [0.0, 0.6, 0.7],
        [0.6, 0.1, 0.5],
    )
    cloth.set_task_right_toe_position([0.0, 0.6, 0.8])
    cloth.set_task_right_toe_position_articulated([0.1, 0.5, 0.8])
    cloth.set_foot_clearance_target(0.1, 2301)
    cloth.stop_foot_clearance_tracking()
    cloth.lock_human_and_chair()
    cloth.arm_slip_detection()
    cloth.configure_mask_proxy_cameras()
    cloth.request_configuration()
    cloth.request_registered_colliders()
    cloth.grasp("left", 0.03)
    cloth.release("right")
    cloth.request_grasp_state()
    cloth.request_scene_geometry()
    cloth.request_attached_particle_indices("left")
    cloth.request_grasp_constraint_error("right")
    cloth.request_contacts()
    cloth.request_robot_human_rigid_collision_qa(1100)
    cloth.request_coverage()
    cloth.reset()
    assert env.messages == [
        (1200, "GetParticles"),
        (1200, "GetParticleVelocities"),
        (
            1200,
            "ConfigureSock",
            0.0,
            0.01,
            1.0,
            0.008,
            0.005,
            0.002,
            0.5,
            True,
            0.95,
            8,
            20,
        ),
        (
            1200,
            "ConfigureGrasp",
            0.0002,
            1000000.0,
            20.0,
            0.20,
            0.11,
            2,
            2,
            0.03,
        ),
        (1200, "SetGraspTargets", 2201, 2202),
        (1200, "AlignGraspTargetsToOpening"),
        (1200, "ClampGraspTargetSpan", 0.115),
        (1200, "AlignSockOpeningToGraspTargets", 0.0, 0.5, 0.6),
        (1200, "IgnoreRobotHumanRigidCollisions", 1100),
        (1200, "IgnoreNonGripperRobotHumanRigidCollisions", 1100),
        (1200, "ConfigureRightLegColliders", 2000),
        (1200, "TranslateHumanAndIK", 0.1, 0.0, -0.2),
        (1200, "FreezeHumanRightToeAt", 0.0, 0.5, 0.6),
        (
            1200,
            "ConfigureHumanTaskPose",
            0.0,
            0.5,
            0.0,
            0.0,
            0.6,
            0.7,
            0.0,
            0.6,
            0.1,
            0.5,
            False,
        ),
        (1200, "SetTaskRightToePosition", 0.0, 0.6, 0.8),
        (1200, "SetTaskRightToePositionArticulated", 0.1, 0.5, 0.8),
        (1200, "SetFootClearanceTarget", 0.1, 2301),
        (1200, "StopFootClearanceTracking"),
        (1200, "LockHumanAndChair"),
        (1200, "ArmSlipDetection", True),
        (1200, "ConfigureMaskProxyCameras", 31),
        (1200, "GetClothConfiguration"),
        (1200, "GetRegisteredObiColliders"),
        (1200, "GraspLeft", 0.03),
        (1200, "ReleaseRight"),
        (1200, "GetGraspState"),
        (1200, "GetSceneGeometry"),
        (1200, "GetAttachedParticleIndices", "left"),
        (1200, "GetGraspForceOrConstraintError", "right"),
        (1200, "GetClothContacts"),
        (1200, "GetRobotHumanRigidCollisionQA", 1100),
        (1200, "GetCoverageObservations"),
        (1200, "ResetSock"),
    ]


def test_configuration_is_fail_closed():
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "stretch_compliance": 0.0005,
        "bend_compliance": 0.005,
        "self_collision": True,
        "damping": 0.8,
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
            "peak_constraint_error": 0.003,
            "over_threshold_steps": 1,
        }
    )
    assert state.particle_indices == (1, 3)
    assert state.peak_constraint_error == pytest.approx(0.003)
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


def test_scene_geometry_is_typed_and_fail_closed():
    geometry = SceneGeometry.from_mapping(
        {
            "valid": True,
            "opening_center": [0, 0, 0],
            "opening_normal": [1, 0, 0],
            "opening_outward_normal": [-1, 0, 0],
            "opening_to_toe_alignment": 1.0,
            "left_cuff_insertion_depth_m": 0.03,
            "right_cuff_insertion_depth_m": 0.03,
            "sock_body_direction": [0, -1, 0],
            "sock_body_gravity_alignment": 0.95,
            "right_toe_position": [-0.1, 0, 0],
            "foot_to_opening_plane_m": 0.1,
            "foot_to_opening_lateral_m": 0.0,
            "right_leg_raise_degrees": 90,
            "left_grasp_position": [0, -0.04, 0],
            "right_grasp_position": [0, 0.04, 0],
        }
    )
    assert geometry.foot_to_opening_plane_m == pytest.approx(0.1)
    assert geometry.opening_target_normal == (1.0, 0.0, 0.0)
    assert geometry.opening_outward_normal == (-1.0, 0.0, 0.0)
    assert geometry.opening_to_toe_alignment == pytest.approx(1.0)
    assert geometry.left_cuff_insertion_depth_m == pytest.approx(0.03)
    assert geometry.sock_body_direction == (0.0, -1.0, 0.0)
    assert geometry.sock_body_gravity_alignment == pytest.approx(0.95)
    assert geometry.opening_span_m == pytest.approx(0.08)
    with pytest.raises(ValueError, match="valid"):
        SceneGeometry.from_mapping({"valid": False})


def test_grasp_configuration_rejects_invalid_thresholds():
    cloth = SockClothAttr(FakeEnvironment(), 1200)
    with pytest.raises(ValueError, match="grasp/slip"):
        cloth.configure_grasp(
            linear_compliance=0,
            rotational_compliance=0.01,
            break_threshold=10,
            slip_constraint_error_m=0,
            slip_opening_span_m=0.085,
            slip_consecutive_steps=3,
            maximum_particles_per_side=2,
            cuff_insertion_depth_m=0.03,
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
