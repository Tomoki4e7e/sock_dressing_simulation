import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from sock_dressing_simulation.assets import (
    generate_rcareworld_runtime_urdf,
    generate_sock_obj,
    movable_joint_names,
    resolve_package_uri,
    vendor_package_meshes,
)
from sock_dressing_simulation.cli import _offline_smoke
from sock_dressing_simulation.config import DRESSING_PLAYER_COMMIT, load_config
from sock_dressing_simulation.dressing_player import _centroid_displacement
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.episode import EpisodeWriter
from sock_dressing_simulation.joints import JointMap


def test_joint_mapping_bounds_and_preserves_mimic():
    mapping = JointMap.from_config(load_config())
    previous = np.zeros(18)
    command = np.full(18, 100.0)
    bounded = mapping.bound(command, previous)
    indexes = mapping.index()
    assert bounded.shape == (18,)
    assert np.all(bounded <= mapping.upper)
    assert bounded[indexes["left_gripper/mimic_joint"]] == (
        bounded[indexes["left_gripper/finger_joint"]] * 0.5
    )
    assert bounded[indexes["right_gripper/mimic_joint"]] == (
        bounded[indexes["right_gripper/finger_joint"]] * 0.5
    )
    full = np.zeros(29)
    simulator = mapping.merge_for_simulator(bounded, full)
    round_trip = mapping.from_simulator(simulator)
    np.testing.assert_allclose(round_trip, bounded)
    np.testing.assert_allclose(
        simulator[mapping.simulator_indices[:7]], np.rad2deg(bounded[:7])
    )
    assert mapping.fixed_names == (
        "torso/joint_1",
        "torso/joint_2",
        "torso/joint_3",
    )
    np.testing.assert_allclose(
        simulator[mapping.fixed_simulator_indices],
        [-45.0, 100.0, 0.0],
    )


def test_sock_obj_is_open_triangulated_tube(tmp_path):
    path = tmp_path / "sock.obj"
    vertices, triangles = generate_sock_obj(path, radial_segments=8, length_segments=3)
    lines = path.read_text().splitlines()
    assert vertices == 32
    assert triangles == 48
    assert "o sock" in lines
    assert "g sock" in lines
    assert "usemtl sock" in lines
    assert path.with_suffix(".mtl").is_file()
    assert len([line for line in lines if line.startswith("f ")]) == triangles
    assert all(len(line.split()) == 4 for line in lines if line.startswith("f "))
    points = np.asarray(
        [[float(value) for value in line.split()[1:]] for line in lines if line.startswith("v ")]
    )
    assert np.linalg.norm(points[:, :2], axis=1).max() == pytest.approx(0.04)
    assert points[:, 2].ptp() == pytest.approx(0.30)


def test_package_meshes_are_vendored_and_rewritten(tmp_path):
    package = tmp_path / "source" / "example"
    mesh = package / "meshes" / "part.stl"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"solid test\nendsolid test\n")
    urdf = tmp_path / "output" / "robot.urdf"
    urdf.parent.mkdir()
    urdf.write_text(
        '<robot name="r"><link name="x"><visual><geometry>'
        '<mesh filename="package://example/meshes/part.stl"/>'
        "</geometry></visual></link></robot>"
    )
    assert resolve_package_uri("package://example/meshes/part.stl", {"example": package}) == mesh
    assert vendor_package_meshes(urdf, urdf.parent, {"example": package}) == 1
    filename = ET.parse(str(urdf)).getroot().find(".//mesh").get("filename")
    assert filename == "meshes/example/meshes/part.stl"
    assert (urdf.parent / filename).is_file()


def test_rcareworld_runtime_urdf_replaces_collada_without_changing_joints(tmp_path):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "paired.dae").write_text("<COLLADA/>")
    (meshes / "paired.stl").write_bytes(b"solid paired\nendsolid paired\n")
    (meshes / "visual_only.dae").write_text("<COLLADA/>")
    source = tmp_path / "robot.urdf"
    source.write_text(
        '<robot name="r">'
        '<link name="base">'
        '<visual><geometry><mesh filename="meshes/paired.dae"/></geometry></visual>'
        '<collision><geometry><mesh filename="meshes/paired.stl"/></geometry></collision>'
        "</link>"
        '<link name="wheel">'
        '<visual><geometry><mesh filename="meshes/visual_only.dae"/></geometry></visual>'
        "</link>"
        '<joint name="wheel_joint" type="continuous">'
        '<parent link="base"/><child link="wheel"/>'
        "</joint>"
        "</robot>"
    )
    runtime = tmp_path / "robot_rcareworld.urdf"

    report = generate_rcareworld_runtime_urdf(source, runtime)

    assert report["profile"] == "stl_visual_fallback"
    assert report["replaced_visual_count"] == 1
    assert report["removed_visual_count"] == 1
    assert report["mesh_extensions"] == [".stl"]
    assert report["missing_mesh_references"] == []
    root = ET.parse(str(runtime)).getroot()
    assert [
        mesh.get("filename") for mesh in root.iter("mesh")
    ] == ["meshes/paired.stl", "meshes/paired.stl"]
    assert root.find("./link[@name='wheel']/visual") is None
    assert movable_joint_names(runtime) == movable_joint_names(source)


def test_episode_writer_contract(tmp_path):
    mapping = JointMap.from_config(load_config())
    episode = tmp_path / "episode"
    with EpisodeWriter(episode, mapping.names, {"source": "unit-test"}) as writer:
        writer.append(
            rgb=np.zeros((3, 4, 3), dtype=np.uint8),
            sock_mask=np.ones((3, 4), dtype=bool),
            leg_mask=np.zeros((3, 4), dtype=bool),
            camera_depth=np.full((3, 4), 125, dtype=np.uint8),
            angle=np.arange(18),
            torque=np.arange(18),
            external_torque=np.arange(18),
        )
    for signal in ("angle", "torque", "external_torque"):
        with (episode / f"{signal}.csv").open() as stream:
            assert len(next(csv.reader(stream))) == 18
    assert (episode / "camera_right/0.png").is_file()
    assert (episode / "camera_right_mask/sock_mask/0.png").is_file()
    assert np.asarray(Image.open(episode / "camera_depth/0.png")).dtype == np.uint8
    assert np.all(np.asarray(Image.open(episode / "depth_mask/leg_depth/0.png")) == 0)
    metadata = json.loads((episode / "metadata.json").read_text())
    assert metadata["frames"] == 1
    assert len(metadata["joint_names"]) == 18
    processing = Path(__file__).parents[2] / "dress_regrasping" / "data-processing"
    sys.path.insert(0, str(processing))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from shareset_io import EpisodeReader

        frame = EpisodeReader(episode).load_frame(0)
        assert frame.ok
        assert frame.sock_depth.dtype == np.uint8
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.remove(str(processing))


def test_offline_smoke_persists_manifest_and_episode(tmp_path):
    result = _offline_smoke(load_config(), tmp_path, frames=2)
    assert result["mode"] == "offline-contract"
    assert Path(result["manifest"]).is_file()
    episode = Path(result["episode"])
    assert len(list((episode / "camera_right").glob("*.png"))) == 2
    manifest = yaml.safe_load(Path(result["manifest"]).read_text())
    assert manifest["data_sock_sim_smoke"]["train"]


class _Backend:
    def step(self):
        pass

    def close(self):
        self.closed = True


class _Robot:
    def __init__(self):
        self.data = {
            "joint_positions": np.zeros(29),
            "joint_force": np.arange(29, dtype=float),
        }
        self.target = None

    def SetJointPosition(self, target):
        self.target = np.asarray(target)


def test_environment_is_injectable_and_reports_api_limitations():
    environment = SockDressingEnv(load_config(), backend=_Backend())
    diagnostics = environment.diagnostics()
    assert diagnostics["contact_force_available"] is False
    assert diagnostics["exact_foot_colliders_configured"] is False
    environment.close()


def test_environment_maps_full_simulator_state_and_command():
    config = load_config()
    environment = SockDressingEnv(config, backend=_Backend())
    environment.robot = _Robot()
    signals = environment.robot_signals()
    assert signals["angle"].shape == (18,)
    assert signals["external_torque_source"].startswith("joint_force proxy")
    command = np.zeros(18)
    command[0] = 0.01
    applied = environment.command(command, np.zeros(18))
    assert applied[0] == 0.01
    assert environment.robot.target[21] == np.rad2deg(0.01)
    np.testing.assert_allclose(environment.robot.target[7:10], [-45.0, 100.0, 0.0])

    command[9] = 0.02
    environment.command(command, applied)
    np.testing.assert_allclose(environment.robot.target[7:10], [-45.0, 100.0, 0.0])


def test_fixed_joint_mapping_rejects_wrong_simulator_name():
    config = load_config()
    config["joints"]["fixed"]["names"][0] = "torso/not_joint_1"
    environment = SockDressingEnv(config, backend=_Backend())
    environment.robot = _Robot()
    simulator_names = [f"unused/joint_{index}" for index in range(29)]
    simulator_names[7:10] = [
        "torso/joint_1",
        "torso/joint_2",
        "torso/joint_3",
    ]
    simulator_names[10:13] = [
        "head/joint_1",
        "head/joint_2",
        "head/joint_3",
    ]
    simulator_names[13:21] = [
        *(f"right_arm/joint_{index}" for index in range(1, 8)),
        "right_gripper/finger_joint",
    ]
    simulator_names[21:29] = [
        *(f"left_arm/joint_{index}" for index in range(1, 8)),
        "left_gripper/finger_joint",
    ]
    environment.robot.data["names"] = simulator_names
    environment.robot.data["types"] = ["RevoluteJoint"] * len(simulator_names)

    with pytest.raises(RuntimeError, match="fixed joint mapping mismatch"):
        environment._validate_simulator_joint_mapping()


def test_dressing_player_profile_is_isolated():
    profile = Path(__file__).parents[1] / "config" / "dressing_player.yaml"
    config = load_config(profile)
    assert config["rcareworld"]["profile"] == "dressing_player"
    assert config["rcareworld"]["commit"] == DRESSING_PLAYER_COMMIT
    assert "RCareWorld-phy-robo-care" in config["rcareworld"]["executable"]
    assert config["dressing_player"]["required_grippers"] == 2


def test_cloth_centroid_displacement_is_measured_not_fabricated():
    first = np.asarray([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    second = first + np.asarray([0.0, 0.25, 0.0])
    assert _centroid_displacement(first, second) == pytest.approx(0.25)
    assert _centroid_displacement(np.empty((0, 3)), second) is None
