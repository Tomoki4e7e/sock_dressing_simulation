from __future__ import annotations

import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Mapping, Tuple


def discover_ros_packages(root: Path) -> Dict[str, Path]:
    packages = {}
    for manifest in Path(root).rglob("package.xml"):
        try:
            name = ET.parse(str(manifest)).getroot().findtext("name")
        except ET.ParseError:
            continue
        if name:
            packages[name.strip()] = manifest.parent.resolve()
    return packages


def resolve_package_uri(uri: str, packages: Mapping[str, Path]) -> Path:
    if not uri.startswith("package://"):
        raise ValueError(f"not a package URI: {uri}")
    payload = uri[len("package://") :]
    package, separator, relative = payload.partition("/")
    if not separator or package not in packages:
        raise FileNotFoundError(f"cannot resolve ROS mesh URI: {uri}")
    path = (Path(packages[package]) / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"mesh does not exist: {path}")
    return path


def vendor_package_meshes(urdf_path: Path, output_dir: Path, packages: Mapping[str, Path]) -> int:
    tree = ET.parse(str(urdf_path))
    count = 0
    for mesh in tree.getroot().iter("mesh"):
        uri = mesh.get("filename", "")
        if not uri.startswith("package://"):
            continue
        payload = uri[len("package://") :]
        package, _, relative = payload.partition("/")
        source = resolve_package_uri(uri, packages)
        destination = output_dir / "meshes" / package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
        mesh.set("filename", (Path("meshes") / package / relative).as_posix())
        count += 1
    tree.write(str(urdf_path), encoding="utf-8", xml_declaration=True)
    return count


def movable_joint_names(urdf_path: Path) -> Tuple[str, ...]:
    """Return non-fixed URDF joints in the order used by RCareWorld."""
    root = ET.parse(str(urdf_path)).getroot()
    return tuple(
        joint.get("name")
        for joint in root.findall("joint")
        if joint.get("type") != "fixed"
    )


def generate_rcareworld_runtime_urdf(urdf_path: Path, runtime_path: Path) -> Dict[str, object]:
    """Create a TriLib-safe URDF by replacing Collada visuals with collision STL."""
    urdf_path = Path(urdf_path).resolve()
    runtime_path = Path(runtime_path).resolve()
    tree = ET.parse(str(urdf_path))
    replaced_visuals = []
    removed_visuals = []

    for link in tree.getroot().findall("link"):
        collision = link.find("./collision/geometry/mesh")
        collision_filename = collision.get("filename", "") if collision is not None else ""
        for visual in list(link.findall("visual")):
            mesh = visual.find("./geometry/mesh")
            if mesh is None or Path(mesh.get("filename", "")).suffix.lower() != ".dae":
                continue
            original = mesh.get("filename", "")
            if collision_filename and Path(collision_filename).suffix.lower() == ".stl":
                mesh.set("filename", collision_filename)
                replaced_visuals.append(
                    {
                        "link": link.get("name", ""),
                        "source": original,
                        "replacement": collision_filename,
                    }
                )
            else:
                link.remove(visual)
                removed_visuals.append(
                    {"link": link.get("name", ""), "source": original}
                )

    references = [
        mesh.get("filename", "") for mesh in tree.getroot().iter("mesh")
    ]
    collada_references = [
        reference for reference in references if Path(reference).suffix.lower() == ".dae"
    ]
    missing_references = [
        reference
        for reference in references
        if not (runtime_path.parent / reference).is_file()
    ]
    if collada_references:
        raise RuntimeError(
            "RCareWorld runtime URDF still contains Collada meshes: "
            + ", ".join(collada_references)
        )
    if missing_references:
        raise FileNotFoundError(
            "RCareWorld runtime URDF has missing mesh references: "
            + ", ".join(missing_references)
        )

    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(runtime_path), encoding="utf-8", xml_declaration=True)
    return {
        "profile": "stl_visual_fallback",
        "source_urdf": str(urdf_path),
        "runtime_urdf": str(runtime_path),
        "replaced_visual_count": len(replaced_visuals),
        "removed_visual_count": len(removed_visuals),
        "replaced_visuals": replaced_visuals,
        "removed_visuals": removed_visuals,
        "mesh_reference_count": len(references),
        "mesh_extensions": sorted(
            {Path(reference).suffix.lower() for reference in references}
        ),
        "missing_mesh_references": missing_references,
    }


def generate_dry_airec_urdf(torobo_ros: Path, product_config: Path, output_dir: Path) -> Tuple[Path, int]:
    torobo_ros = Path(torobo_ros).resolve()
    packages = discover_ros_packages(torobo_ros)
    description = packages.get("torobo_description")
    if description is None:
        raise FileNotFoundError(f"torobo_description package not found below {torobo_ros}")
    xacro_file = description / "urdf" / "load_robot.urdf.xacro"
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = output_dir / "dry_airec3_gripper_v2.urdf"
    environment = os.environ.copy()
    roots = sorted({str(path.parent) for path in packages.values()})
    environment["ROS_PACKAGE_PATH"] = os.pathsep.join(
        roots + ([environment["ROS_PACKAGE_PATH"]] if environment.get("ROS_PACKAGE_PATH") else [])
    )
    process = subprocess.run(
        [
            shutil.which("xacro") or "xacro",
            str(xacro_file),
            f"path:={Path(product_config).resolve()}",
            "sim:=false",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if process.returncode:
        raise RuntimeError(f"xacro failed:\n{process.stderr.strip()}")
    urdf_path.write_text(process.stdout, encoding="utf-8")
    return urdf_path, vendor_package_meshes(urdf_path, output_dir, packages)


def generate_sock_obj(
    path: Path,
    length: float = 0.30,
    radius: float = 0.04,
    radial_segments: int = 32,
    length_segments: int = 24,
    closed_toe: bool = False,
    bend_start: float = 0.0,
    bend_length: float = 0.0,
    bend_degrees: float = 0.0,
    bend_azimuth_degrees: float = 0.0,
) -> Tuple[int, int]:
    if (
        length <= 0
        or radius <= 0
        or radial_segments < 3
        or length_segments < 1
        or bend_start < 0
        or bend_length < 0
        or abs(bend_degrees) > 180
        or not math.isfinite(bend_azimuth_degrees)
        or bend_start + bend_length > length
        or (bend_degrees != 0 and bend_length <= 0)
    ):
        raise ValueError("invalid tubular sock dimensions")
    vertices = []
    bend_sign = 1.0 if bend_degrees >= 0 else -1.0
    total_angle = abs(math.radians(bend_degrees))
    bend_azimuth = math.radians(bend_azimuth_degrees)
    bend_direction = (
        math.sin(bend_azimuth),
        math.cos(bend_azimuth),
    )
    cross_direction = (
        math.cos(bend_azimuth),
        -math.sin(bend_azimuth),
    )
    bend_radius = bend_length / total_angle if total_angle > 0 else 0.0
    for row in range(length_segments + 1):
        axial = length * row / length_segments
        after_start = max(0.0, axial - bend_start)
        bent_distance = min(after_start, bend_length)
        fraction = bent_distance / bend_length if bend_length > 0 else 0.0
        bend_angle = total_angle * fraction
        center_offset = (
            -bend_sign * bend_radius * (1.0 - math.cos(bend_angle))
            if total_angle > 0
            else 0.0
        )
        center_x = center_offset * bend_direction[0]
        center_y = center_offset * bend_direction[1]
        center_z = (
            bend_start + bend_radius * math.sin(bend_angle)
            if axial > bend_start and total_angle > 0
            else axial
        )
        if after_start > bend_length and total_angle > 0:
            remainder = after_start - bend_length
            center_x -= (
                bend_sign * remainder * math.sin(total_angle) *
                bend_direction[0]
            )
            center_y -= (
                bend_sign * remainder * math.sin(total_angle) *
                bend_direction[1]
            )
            center_z += remainder * math.cos(total_angle)
        for column in range(radial_segments):
            angle = 2.0 * math.pi * column / radial_segments
            radial_cross = radius * math.cos(angle)
            radial_bend = radius * math.sin(angle)
            vertices.append(
                (
                    center_x + radial_cross * cross_direction[0] +
                    radial_bend * math.cos(bend_angle) * bend_direction[0],
                    center_y + radial_cross * cross_direction[1] +
                    radial_bend * math.cos(bend_angle) * bend_direction[1],
                    center_z + bend_sign * radial_bend * math.sin(bend_angle),
                )
            )
    faces = []
    for row in range(length_segments):
        for column in range(radial_segments):
            nxt = (column + 1) % radial_segments
            a = row * radial_segments + column + 1
            b = row * radial_segments + nxt + 1
            c = (row + 1) * radial_segments + column + 1
            d = (row + 1) * radial_segments + nxt + 1
            faces.extend(((a, c, b), (b, c, d)))
    if closed_toe:
        last_ring_start = length_segments * radial_segments
        toe_center = tuple(
            sum(vertices[last_ring_start + column][axis] for column in range(radial_segments))
            / radial_segments
            for axis in range(3)
        )
        vertices.append(toe_center)
        center = len(vertices)
        for column in range(radial_segments):
            nxt = (column + 1) % radial_segments
            a = last_ring_start + column + 1
            b = last_ring_start + nxt + 1
            faces.append((a, b, center))
    path.parent.mkdir(parents=True, exist_ok=True)
    material_path = path.with_suffix(".mtl")
    material_path.write_text(
        "newmtl sock\nKa 0.2 0.2 0.2\nKd 0.85 0.2 0.25\nKs 0 0 0\n",
        encoding="ascii",
    )
    lines = [
        "# Triangulated sock cloth mesh with open cuff"
        + (" and closed toe" if closed_toe else ""),
        f"mtllib {material_path.name}",
        "o sock",
        "g sock",
        "usemtl sock",
    ]
    lines.extend("v {:.9g} {:.9g} {:.9g}".format(*vertex) for vertex in vertices)
    lines.extend("f {} {} {}".format(*face) for face in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return len(vertices), len(faces)
