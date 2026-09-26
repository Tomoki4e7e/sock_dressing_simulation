from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.quality import assess_observation_quality
from sock_dressing_simulation.sock_cloth import (
    PROTOCOL_VERSION,
    ClothContact,
    DressingQA,
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
    assert expected["grasp_linear_compliance"] == pytest.approx(0.00005)
    assert expected["grasp_rotational_compliance"] == 1000000.0
    assert expected["grasp_break_threshold"] == pytest.approx(20.0)
    assert expected["slip_constraint_error_m"] == pytest.approx(0.20)
    assert expected["slip_opening_span_m"] == 0.11
    assert expected["slip_consecutive_steps"] == 2
    assert expected["maximum_grasp_particles_per_side"] == 4
    assert expected["grasp_thickness_half_width_m"] == pytest.approx(0.022)
    assert not expected["tether_constraints"]
    assert expected["tether_compliance"] == 0.0
    assert expected["tether_scale"] == 1.0


def test_strict_autonomous_profile_restores_straight_leg_pose_baseline():
    config = load_config(
        Path("config/autonomous_real_only_strict_dressing.yaml")
    )
    scene = config["scene"]
    pose = scene["initial_pose_contract"]

    assert scene["human_position"] == [0.06, 1.48, -0.72]
    assert scene["chair"]["parts"][0]["position"] == [0.0, 0.48, -0.18]
    assert pose["straight_right_leg"]
    assert pose["right_knee_flexion_max_degrees"] == pytest.approx(2.0)
    assert pose["right_leg_raise_degrees"] == pytest.approx(90.0)
    assert "right_toe_offset_world_m" not in pose
    assert config["scenario"]["foot"]["plantarflexion_degrees"] == pytest.approx(
        30.0
    )
    assert config["inference"]["reference_action_blend"] == pytest.approx(0.0)
    assert config["inference"]["reference_actions"] is None


def test_wide_cuff_small_foot_profile_uses_requested_geometry():
    config = load_config(
        Path(
            "config/"
            "autonomous_real_only_straight_neutral_wide_cuff_small_foot.yaml"
        )
    )

    assert config["obi"]["expected"][
        "grasp_thickness_half_width_m"
    ] == pytest.approx(0.035)
    assert config["scene"][
        "right_foot_collider_cross_section_scale"
    ] == pytest.approx(0.8)
    assert config["scenario"]["foot"][
        "plantarflexion_degrees"
    ] == pytest.approx(0.0)
    assert config["scene"]["initial_pose_contract"]["straight_right_leg"]
    assert config["inference"]["reference_action_blend"] == pytest.approx(0.0)
    assert config["inference"]["reference_actions"] is None


def test_rigid_collision_corrected_profile_enables_gripper_foot_contacts():
    config = load_config(
        Path(
            "config/"
            "autonomous_real_only_straight_neutral_wide_cuff_"
            "rigid_collision_corrected.yaml"
        )
    )
    scene = config["scene"]
    player = config["dressing_player"]

    assert not scene["ignore_robot_human_rigid_collisions"]
    assert scene["ignore_non_gripper_robot_human_rigid_collisions"]
    assert scene["initial_pose_contract"][
        "maximum_robot_human_penetration_m"
    ] == pytest.approx(0.005)
    assert player["abort_on_rigid_collision_qa_failure"]
    assert not player["allow_ignored_robot_human_collision_pairs"]
    assert player["maximum_robot_human_penetration_m"] == pytest.approx(0.005)


def test_plate_normal_profile_uses_exactly_four_grasp_points():
    config = load_config(
        Path("config/autonomous_real_only_plate_normal_four_point.yaml")
    )
    scene = config["scene"]
    pose = scene["initial_pose_contract"]

    assert scene["grasp_anchors"]["align_sock_to_gripper_plate"]
    assert config["obi"]["expected"]["maximum_grasp_particles_per_side"] == 2
    assert config["obi"]["expected"][
        "grasp_thickness_half_width_m"
    ] == pytest.approx(0.022)
    assert pose["required_grasp_particles_per_side"] == 2
    assert pose["lock_human_and_chair"]
    assert pose["vertical_toe_drop_m"] == pytest.approx(0.40)
    assert pose["away_from_robot_m"] is None
    assert pose["down_m"] is None
    assert pose["opening_to_toe_alignment_min"] == pytest.approx(0.90)
    assert pose["enforce"]


def test_triaxial_offset_profile_only_adds_requested_pose_offsets():
    baseline = load_config(
        Path("config/autonomous_real_only_plate_normal_four_point.yaml")
    )
    config = load_config(
        Path(
            "config/"
            "autonomous_real_only_plate_normal_four_point_"
            "human_triaxial_offset.yaml"
        )
    )
    pose = config["scene"]["initial_pose_contract"]

    assert pose["away_from_robot_m"] == pytest.approx(0.10)
    assert pose["up_m"] == pytest.approx(0.10)
    assert pose["right_from_robot_m"] == pytest.approx(0.05)
    assert pose["down_m"] is None
    assert pose["required_grasp_particles_per_side"] == 2
    assert pose["opening_to_toe_alignment_min"] == pytest.approx(0.90)
    assert config["scene"]["grasp_anchors"]["align_sock_to_gripper_plate"]
    assert config["obi"]["expected"]["maximum_grasp_particles_per_side"] == 2
    expected = deepcopy(baseline)
    expected["scene"]["initial_pose_contract"].update(
        {
            "away_from_robot_m": 0.10,
            "up_m": 0.10,
            "right_from_robot_m": 0.05,
        }
    )
    assert config == expected


def test_human_chair_locked_profile_restores_recorded_world_coordinates():
    config = load_config(
        Path("config/autonomous_real_only_human_chair_locked_pose.yaml")
    )
    pose = config["scene"]["initial_pose_contract"]
    baseline = pose["locked_pose_baseline"]

    assert baseline["human_root_position"] == pytest.approx(
        [0.0599999987, 1.4800000191, -0.7200000286]
    )
    assert baseline["chair_position"] == pytest.approx(
        [-0.0712867528, 0.5095953941, -0.3411580324]
    )
    assert baseline["human_anchor_position"] == pytest.approx(
        [-0.0712867528, 0.7595953345, -0.3411580324]
    )
    assert baseline["right_toe_position"] == pytest.approx(
        [-0.1310119629, 0.5210645795, 0.6446849108]
    )
    assert pose["right_toe_offset_world_m"] == [0.05, -0.05, 0.0]
    assert config["inference"]["reference_action_blend"] == pytest.approx(0.0)


def test_downward_plate_profile_enables_autonomous_geometry_contract():
    config = load_config(
        Path("config/autonomous_real_only_downward_plate_human_chair.yaml")
    )
    pose = config["scene"]["initial_pose_contract"]

    assert config["scene"]["grasp_anchors"]["align_sock_to_gripper_plate"]
    assert pose["right_toe_offset_world_m"] is None
    assert pose["vertical_toe_drop_m"] == pytest.approx(0.05)
    assert pose["enforce"]
    assert config["inference"]["reference_actions"] is None
    assert config["inference"]["reference_action_blend"] == pytest.approx(0.0)


def test_restore_human_chair_world_pose_sends_recorded_vectors():
    environment = FakeEnvironment()
    cloth = SockClothAttr(environment, 1200)

    cloth.restore_human_chair_world_pose(
        [0.06, 1.48, -0.72],
        [-0.071, 0.510, -0.341],
        [-0.071, 0.760, -0.341],
        [-0.131, 0.521, 0.645],
        2301,
    )

    message = environment.messages[-1]
    assert message[:2] == (1200, "RestoreHumanChairWorldPose")
    assert message[-1] == 2301
    assert len(message) == 15


def test_sock_opening_alignment_writes_solver_local_particle_positions():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()
    alignment = source[
        source.index("public void AlignSockOpeningToGraspTargets"):
        source.index("public void SetGraspTargetPosition")
    ]

    assert "GetParticleEndpoints(openingParticles" in alignment
    assert "GetParticleEndpoints(cuffGraspParticles" not in alignment
    assert "openingAxisScale = targetOpeningSpan / sourceOpeningSpan" in alignment
    assert "openingCrossAxisScale" in alignment
    assert "rectangleBoundary" in alignment
    assert "CaptureStructuralRestLengths();" in alignment
    assert "openingStretchWeight" in alignment
    assert "SnapNearestOpeningParticle(" in alignment
    assert "alignedLeftOpeningCorners" in source
    assert "alignedRightOpeningCorners" in source
    assert "left.transform.TransformPoint(" in alignment
    assert "right.transform.TransformPoint(" in alignment
    assert "solver.transform.InverseTransformPoint(aligned)" in alignment
    assert "Vector3.ProjectOnPlane(" in alignment
    assert "Physics.gravity" not in alignment
    assert "Quaternion.AngleAxis(" in alignment
    assert "Quaternion.LookRotation(" in alignment
    assert "left.transform.rotation = targetFrameRotation;" in alignment
    assert "right.transform.rotation = targetFrameRotation;" in alignment
    assert "graspOrientationLocked = true;" in alignment
    assert "targetMidpoint - toeTarget" in alignment
    assert alignment.index("left.transform.rotation = targetFrameRotation;") < (
        alignment.index("SnapNearestOpeningParticle(")
    )
    assert "CaptureResetStateIfReady(true)" in alignment
    assert "AlignSockOpeningToGraspTargetsAndGrasp" in source
    assert "AlignSockOpeningToGraspPlateAndGrasp" in source
    assert "Vector3.Cross(openingAxis, shortAxis)" in source
    assert "if (inwardNormal.y < 0)" in source
    assert 'Grasp("left", leftTargetId, maxDistance);' in source
    assert 'Grasp("right", rightTargetId, maxDistance);' in source


def test_custom_player_keeps_grasp_targets_at_frames_and_enforces_edge_match():
    config = load_config(Path("config/custom_player.yaml"))

    anchors = config["scene"]["grasp_anchors"]
    contract = config["scene"]["initial_pose_contract"]
    assert anchors["maximum_anchor_span_m"] is None
    assert all(anchor["local_position"] == [0.0, 0.0, 0.0] for anchor in anchors["anchors"])
    assert contract["minimum_cuff_insertion_depth_m"] == 0.0
    assert contract["maximum_opening_edge_error_m"] == pytest.approx(0.01)
    assert contract["required_grasp_particles_per_side"] == 4
    assert config["obi"]["expected"]["grasp_thickness_half_width_m"] == pytest.approx(
        0.022
    )
    assert config["scene"]["grasp_alignment"]["target_span_m"] == pytest.approx(
        0.105
    )
    assert 0.105 * (2 * 0.022) == pytest.approx(0.00462)


def test_custom_player_reports_geometric_dressing_qa():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()

    assert "public void GetDressingQA()" in source
    assert "FootSurfaceSamples" in source
    assert "PointInsideSock" in source
    assert "RayIntersectsTriangle" in source
    assert '"surface_containment_ratio"' in source
    assert '"cuff_progress_toward_ankle_m"' in source
    assert '"maximum_cloth_foot_penetration_m"' in source
    assert '"geometric_maximum_cloth_foot_penetration_m"' in source
    assert '"geometric_overlap_particle_count"' in source
    assert '{ "toes", "forefoot", "heel", "ankle", "calf" }' in source


def test_custom_player_projects_strain_before_collision_solving():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()

    assert "solver.OnSimulationStart += OnSolverSimulationStart;" in source
    assert "solver.OnSimulationEnd += OnSolverSimulationEnd;" in source
    callback = source.split("private void OnSolverSimulationStart(", 1)[1].split(
        "private void OnSolverCollision(", 1
    )[0]
    assert "SyncRightLegCollidersToBones();" in callback
    assert "LimitStructuralStretch();" in callback
    end_callback = source.split("private void OnSolverSimulationEnd(", 1)[1].split(
        "private Dictionary<string, object> GeometricFootPenetration()", 1
    )[0]
    assert "LimitStructuralStretch();" in end_callback
    late_update = source.split("private void LateUpdate()", 1)[1].split(
        "[RFUAPI]", 1
    )[0]
    get_particles = source.split("public void GetParticles()", 1)[1].split(
        "[RFUAPI]", 1
    )[0]
    assert "LimitStructuralStretch();" not in late_update
    assert "LimitStructuralStretch();" not in get_particles
    collision_callback = source.split(
        "private void OnSolverCollision(", 1
    )[1].split("private Dictionary<string, object> GeometricFootPenetration()", 1)[
        0
    ]
    assert "contacts.Clear();" not in collision_callback
    assert "MaximumAggregatedContactRecords" in collision_callback
    get_contacts = source.split("public void GetClothContacts()", 1)[1].split(
        "[RFUAPI]", 1
    )[0]
    assert "contacts.Clear();" in get_contacts
    rest_length = source.split(
        "private float StructuralRestLength", 1
    )[1].split("private void LimitStructuralStretch", 1)[0]
    assert "Mathf.Max(alignedRest, blueprintRest)" in rest_length


def test_dressing_qa_requires_finite_cross_section_observations():
    value = {
        "valid": True,
        "simulation_frame": 12,
        "surface_containment_ratio": 0.95,
        "sections": [
            {"name": "toes", "valid": True, "containment_ratio": 0.875}
        ],
        "cuff_progress_toward_ankle_m": 0.03,
        "maximum_cuff_reverse_step_m": 0.001,
        "cuff_beyond_distal_toe_m": 0.0,
        "foot_contact_count": 4,
        "maximum_cloth_foot_penetration_m": 0.001,
        "maximum_cloth_foot_force_proxy": 0.2,
    }

    report = DressingQA.from_mapping(value)

    assert report.valid
    assert report.surface_containment_ratio == pytest.approx(0.95)
    with pytest.raises(ValueError):
        DressingQA.from_mapping(
            {**value, "surface_containment_ratio": float("nan")}
        )


def test_dressing_qa_preserves_geometric_overlap_without_obi_contacts():
    value = {
        "valid": True,
        "simulation_frame": 12,
        "surface_containment_ratio": 0.0,
        "sections": [
            {"name": "calf", "valid": True, "containment_ratio": 0.0}
        ],
        "cuff_progress_toward_ankle_m": 0.0,
        "maximum_cuff_reverse_step_m": 0.0,
        "cuff_beyond_distal_toe_m": 0.0,
        "foot_contact_count": 0,
        "obi_foot_contact_count": 0,
        "obi_maximum_cloth_foot_penetration_m": 0.0,
        "geometric_overlap_particle_count": 3,
        "geometric_maximum_cloth_foot_penetration_m": 0.004,
        "geometric_foot_penetration": {
            "regions": [
                {
                    "name": "calf",
                    "overlap_particle_count": 3,
                    "maximum_penetration_m": 0.004,
                }
            ]
        },
        "maximum_cloth_foot_penetration_m": 0.004,
        "maximum_cloth_foot_force_proxy": 0.0,
    }

    report = DressingQA.from_mapping(value)

    assert report.foot_contact_count == 0
    assert report.geometric_overlap_particle_count == 3
    assert report.geometric_maximum_cloth_foot_penetration_m == pytest.approx(
        0.004
    )
    assert report.geometric_regions[0]["name"] == "calf"


def test_sock_geometry_reports_particle_derived_hanging_direction():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()
    config = load_config(Path("config/custom_player.yaml"))

    assert "Vector3 sockBodyDirection = ClothCenter() - openingCenter;" in source
    assert "Vector3 openingNormal = OpeningNormal();" in source
    assert "Vector3 pinnedNormal = Vector3.Cross(" in source
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
    assert "IsGripperCollider(value, robot.transform)" in source
    classifier = source[
        source.index("private static bool IsGripperCollider"):
        source.index("private static string TransformPath")
    ]
    assert "current != robotRoot" in classifier
    assert '"maximum_enabled_penetration_m"' in source
    assert '"maximum_ignored_penetration_m"' in source
    assert '"robot_collider_path"' in source
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
    assert "float.PositiveInfinity" in source
    assert '"non_cuff_grasp_particle_count"' in source
    assert "int[] selected = selectedValues.ToArray();" in source
    assert "HashSet<long> structuralEdges" in source
    assert "adjacentToPinned" in source
    assert ".ThenBy(value => value.index)" in source
    assert "Array.IndexOf(alignedCorners, actorIndex)" in source
    assert "new Vector3(-graspThicknessHalfWidth, 0, 0)" in source
    assert "new Vector3(graspThicknessHalfWidth, 0, 0)" in source
    assert "triangle[0].index" in source
    assert "counts.Where(item => item.Value == 1)" in source
    assert "RestDistancesFromOpening" in source
    assert "closest + neighbor.Value" in source
    assert "distance - cuffInsertionDepth" in source
    assert '"cuff_insertion_depth_m", cuffInsertionDepth' in source
    assert "cloth.tetherConstraintsEnabled = tetherEnabled;" in source
    assert "cloth.tetherCompliance = tetherCompliance;" in source
    assert "cloth.tetherScale = tetherScale;" in source
    assert "LimitStructuralStretch();" in source
    assert source.index("LimitStructuralStretch();", source.index("public void GetParticles")) > 0
    assert "solver.positions[cloth.GetParticleRuntimeIndex(i)]" in source
    assert "EnforceGraspParticlePositions();" in source
    assert "EnforceGraspTargetOrientations();" in source
    assert "if (span > slipOpeningSpan)" in source
    assert "grasp.target.TransformPoint(grasp.localOffsets[i])" in source
    assert "ApplyGraspCenterTranslation();" in source
    assert "rest * maximumStructuralStretch" in source
    assert '"aligned_grasp_initialization"' in source
    assert "StructuralRestLength(" in source
    assert "if (pinned.Count == 0)" in source
    limiter = source[
        source.index("private void LimitStructuralStretch"):
        source.index("private void EnforceGraspParticlePositions")
    ]
    assert "openingParticles.Contains(edge.x)" not in limiter
    assert "solver.prevPositions[firstSolver]" in limiter
    assert "solver.prevPositions[secondSolver]" in limiter
    assert "cloth.volumeConstraintsEnabled = false;" in source
    assert "solver.invMasses[solverIndex] = 1.0f / particleMass;" in source
    assert "leftOpeningEdge = GraspParticleCenter(grasps[\"left\"]);" not in source
    assert "rightOpeningEdge = GraspParticleCenter(grasps[\"right\"]);" not in source
    assert "alignedLeftOpeningCorners" in source
    assert "alignedRightOpeningCorners" in source
    assert '"left_grasp_patch_span_m", leftGraspPatchSpan' in source
    assert '"right_grasp_patch_span_m", rightGraspPatchSpan' in source
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
    assert "configuredFootColliderCrossSectionScale" in source
    assert "ScaleFootCrossSection" in source


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


def test_human_chair_lock_restores_world_anchors_and_reports_drift():
    source = Path(
        "RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs"
    ).read_text()

    assert "EnforceHumanChairLock();" in source
    assert "bodyRoot.position = lockedHumanRootPosition;" in source
    assert "visualRightHip = lockedVisualRightHip;" in source
    assert "chair.transform.position = lockedChairPosition;" in source
    assert '"human_root_drift_m"' in source
    assert '"human_anchor_drift_m"' in source
    restore = source.split(
        "public void RestoreHumanChairWorldPose", 1
    )[1].split("public void LockHumanAndChair", 1)[0]
    assert "rigidTaskPoseTranslation = true;" in restore
    assert "UpdateCanonicalHumanTaskPose();" in restore
    diagnostics = source.split(
        "public void GetVisualDiagnostics", 1
    )[1].split("[RFUAPI]", 1)[0]
    assert "if (sockHuman != null)" in diagnostics
    assert '{ "locked", humanChairLocked }' in diagnostics


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
        grasp_thickness_half_width_m=0.02,
    )
    cloth.set_grasp_targets(2201, 2202)
    cloth.align_grasp_targets_to_opening()
    cloth.clamp_grasp_target_span(0.115)
    cloth.align_sock_opening_to_grasp_targets([0.0, 0.5, 0.6])
    cloth.align_sock_opening_to_grasp_targets_and_grasp(
        [0.0, 0.5, 0.6], 0.03
    )
    cloth.align_sock_opening_to_grasp_plate_and_grasp(0.03)
    cloth.ignore_robot_human_rigid_collisions(1100)
    cloth.ignore_non_gripper_robot_human_rigid_collisions(1100)
    cloth.configure_right_leg_colliders(2000, 0.8)
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
    cloth.lock_human_and_chair(2301)
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
            True,
            0.0,
            1.0,
            1.5,
            8,
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
            0.02,
        ),
        (1200, "SetGraspTargets", 2201, 2202),
        (1200, "AlignGraspTargetsToOpening"),
        (1200, "ClampGraspTargetSpan", 0.115),
        (1200, "AlignSockOpeningToGraspTargets", 0.0, 0.5, 0.6),
        (
            1200,
            "AlignSockOpeningToGraspTargetsAndGrasp",
            0.0,
            0.5,
            0.6,
            0.03,
        ),
        (1200, "AlignSockOpeningToGraspPlateAndGrasp", 0.03),
        (1200, "IgnoreRobotHumanRigidCollisions", 1100),
        (1200, "IgnoreNonGripperRobotHumanRigidCollisions", 1100),
        (1200, "ConfigureRightLegColliders", 2000, 0.8),
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
        (1200, "LockHumanAndChair", 2301),
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
            "left_grasp_thickness_axis": [0, 0, 1],
            "right_grasp_thickness_axis": [0, 0, 1],
            "left_grasp_inward_axis": [1, 0, 0],
            "right_grasp_inward_axis": [1, 0, 0],
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
    assert geometry.left_grasp_thickness_axis == (0.0, 0.0, 1.0)
    assert geometry.right_grasp_inward_axis == (1.0, 0.0, 0.0)
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
            grasp_thickness_half_width_m=0.02,
        )


@pytest.mark.parametrize("scale", [0.0, -0.1, 1.01, np.nan, np.inf])
def test_foot_collider_scale_rejects_invalid_values(scale):
    cloth = SockClothAttr(FakeEnvironment(), 1200)

    with pytest.raises(ValueError, match="foot_cross_section_scale"):
        cloth.configure_right_leg_colliders(2000, scale)


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
