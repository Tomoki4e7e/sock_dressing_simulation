"""Static right-hand/skin geometry, matching the URDF importer's convex meshes.

Optional dependencies: scipy, trimesh and python-fcl. No simulator mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import fcl
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


# ROS-to-Unity is (-y,z,x), det=-1. The canonical search frame keeps approach
# and closing axes, and reflects the third axis. A proper SO(3) rotation then
# places that left-handed URDF hand in the numerical Unity world coordinates.
HANDEDNESS = np.diag([1., 1., -1.])


def unit(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        raise ValueError('Degenerate direction')
    return np.asarray(v, dtype=float)/norm


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, :3] = Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
        transform[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    return transform


def convex_geometry(vertices):
    vertices = np.unique(np.asarray(vertices, dtype=float), axis=0)
    vertices = vertices[ConvexHull(vertices).vertices]
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    for face, plane in zip(faces, hull.equations):
        a, b, c = vertices[face]
        if np.dot(np.cross(b-a, c-a), plane[:3]) < 0:
            face[1], face[2] = face[2], face[1]
    packed = np.c_[np.full(len(faces), 3), faces].astype(np.int32).ravel()
    return fcl.Convex(vertices, len(faces), packed), vertices, faces


@dataclass
class Part:
    label: str
    obj: object
    center: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray


def make_part(label, vertices):
    geometry, vertices, faces = convex_geometry(vertices)
    return Part(label, fcl.CollisionObject(geometry), vertices.mean(0), vertices, faces)


class RightHand:
    """Five collision pieces: palm, two finger bases, two adaptive fingers."""
    root_link = 'right_gripper/gripper_base'

    def __init__(self, urdf, tip_inset_m=.010, maximum_gap_m=.050):
        self.urdf = Path(urdf)
        self.tree = ET.parse(self.urdf).getroot()
        self.joints = {j.find('child').get('link'): j for j in self.tree.findall('joint')}
        self.meshes = {}
        self.mesh_paths = []
        for link in self.tree.findall('link'):
            name = link.get('name')
            if not name.startswith('right_gripper/'):
                continue
            for i, collision in enumerate(link.findall('collision')):
                mesh = collision.find('geometry/mesh')
                if mesh is None:
                    continue
                path = self.urdf.parent/mesh.get('filename')
                loaded = trimesh.load_mesh(str(path), process=False)
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                points = np.asarray(loaded.vertices)*scale
                matrix = origin(collision.find('origin'))
                points = points@matrix[:3, :3].T+matrix[:3, 3]
                self.meshes[name+('' if i == 0 else f'/{i}')] = (name, points)
                self.mesh_paths.append(path)
        if len(self.meshes) != 5:
            raise ValueError(f'Expected five hand collision meshes, got {list(self.meshes)}')
        raw = self.meshes['right_gripper/right_finger_link'][1]
        finger = self.link_transform('right_gripper/right_finger_link', 0.)
        physical_tip = float((raw@finger[:3, :3].T+finger[:3, 3])[:, 0].max())
        self.grasp_center_urdf = np.array([physical_tip-tip_inset_m, 0., -.040])
        # Exact inner face at the grasp center and the mechanical width center.
        local = np.linalg.solve(finger[:3, :3], self.grasp_center_urdf-finger[:3, 3])
        planes = ConvexHull(np.unique(raw, axis=0)).equations
        lower = planes[:, 2] < -1e-10
        inner_z = float(np.max(-(planes[lower, 0]*local[0]+planes[lower, 1]*local[1]+planes[lower, 3])/planes[lower, 2]))
        self.gap_offset_m = .020+2*inner_z
        self.maximum_gap_m = maximum_gap_m
        self.tip_inset_m = tip_inset_m
        self.physical_tip_x_m = physical_tip
        self.inner_face_local_z_m = inner_z
        limit = self.tree.find("joint[@name='right_gripper/finger_joint']/limit")
        self.joint_limits = [float(limit.get(k)) for k in ('lower', 'upper')]
        for gap in (0., maximum_gap_m):
            q = self.joint_for_gap(gap)
            if not self.joint_limits[0] <= q <= self.joint_limits[1]:
                raise ValueError('Requested physical gap exceeds URDF joint limits')
        self._banks = {}

    def link_transform(self, name, q):
        if name == self.root_link:
            return np.eye(4)
        joint = self.joints[name]
        parent = self.link_transform(joint.find('parent').get('link'), q)
        transform = origin(joint.find('origin'))
        if joint.get('type') == 'prismatic':
            mimic = joint.find('mimic')
            value = q if mimic is None else q*float(mimic.get('multiplier', '1'))+float(mimic.get('offset', '0'))
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            transform[:3, 3] += transform[:3, :3]@axis*value
        return parent@transform

    def joint_for_gap(self, gap):
        return float(gap-self.gap_offset_m)

    def vertices(self, gap):
        result = {}
        for label, (link, points) in self.meshes.items():
            matrix = self.link_transform(link, self.joint_for_gap(gap))
            base = points@matrix[:3, :3].T+matrix[:3, 3]
            result[label] = (base-self.grasp_center_urdf)@HANDEDNESS.T
        return result

    def bank(self, gap, kind='endpoint', pinch_gap=.005, approach=.050):
        key = (float(gap), kind, float(pinch_gap), float(approach))
        if key not in self._banks:
            vertices = self.vertices(gap)
            low = self.vertices(pinch_gap) if kind in ('closure','combined') else None
            parts = []
            for label, points in vertices.items():
                if kind == 'closure':
                    points = np.r_[points, low[label]]
                elif kind == 'combined':
                    points = np.r_[points,low[label]]
                    points = np.r_[points,points-np.array([approach,0.,0.])]
                elif kind == 'approach':
                    points = np.r_[points, points-np.array([approach, 0., 0.])]
                elif kind != 'endpoint':
                    raise ValueError(kind)
                parts.append(make_part(label, points))
            # Test the usually most restricted inside finger first.
            priority = ['left_finger_link', 'right_finger_link', 'gripper_base', 'left_finger_base_link', 'right_finger_base_link']
            parts.sort(key=lambda p: priority.index(p.label.split('/')[-1]))
            self._banks[key] = parts
        return self._banks[key]


def closed_skin(skin):
    """Seal only existing boundary loops; retain all actual skin triangles."""
    mesh = trimesh.Trimesh(np.asarray(skin['vertices']), np.asarray(skin['triangles']).reshape(-1, 3), process=True)
    if not mesh.is_winding_consistent:
        raise ValueError('Inconsistent actual skin triangle orientation')
    directed = np.concatenate([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    _, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    if (counts > 2).any():
        raise ValueError('Nonmanifold actual skin')
    boundary = directed[counts[inverse] == 1]
    vertices = mesh.vertices.tolist(); faces = mesh.faces.tolist(); caps = []
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), set()).add(int(b)); adjacency.setdefault(int(b), set()).add(int(a))
    if any(len(v) != 2 for v in adjacency.values()):
        raise ValueError('Skin boundary is not a collection of loops')
    remaining = set(adjacency)
    while remaining:
        first = next(iter(remaining)); todo = [first]; component = set()
        while todo:
            a = todo.pop()
            if a in component: continue
            component.add(a); todo.extend(adjacency[a]-component)
        remaining -= component
        center = len(vertices); vertices.append(mesh.vertices[list(component)].mean(0).tolist())
        for a, b in boundary:
            if int(a) in component:
                face = [int(b), int(a), center]; caps.append(face); faces.append(face)
    result = trimesh.Trimesh(np.array(vertices), np.array(faces), process=False)
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError('Failed to close actual skin')
    return result, np.array(caps, dtype=int).reshape(-1, 3)


class SkinBody:
    def __init__(self, skin):
        self.mesh, self.caps = closed_skin(skin)
        model = fcl.BVHModel()
        model.beginModel(len(self.mesh.faces), len(self.mesh.vertices))
        model.addSubModel(np.asarray(self.mesh.vertices), np.asarray(self.mesh.faces, dtype=np.int32))
        model.endModel()
        self.obj = fcl.CollisionObject(model)
        self.request = fcl.DistanceRequest()
        self.minimum = self.mesh.vertices.min(0); self.maximum = self.mesh.vertices.max(0)
        self.triangles = self.mesh.vertices[self.mesh.faces]

    def winding(self, points):
        x = np.asarray(points)[:, None, :]
        a, b, c = [self.triangles[None, :, k, :]-x for k in range(3)]
        la = np.linalg.norm(a, axis=2); lb = np.linalg.norm(b, axis=2); lc = np.linalg.norm(c, axis=2)
        numerator = np.einsum('pti,pti->pt', a, np.cross(b, c))
        denominator = la*lb*lc+np.einsum('pti,pti->pt', a, b)*lc+np.einsum('pti,pti->pt', b, c)*la+np.einsum('pti,pti->pt', c, a)*lb
        return np.abs((2*np.arctan2(numerator, denominator)).sum(axis=1)/(4*np.pi))

    def inspect(self, parts, rotation, position, margin=.001):
        """Solid containment plus convex-part/actual-skin BVH distance."""
        if not parts:raise ValueError('No hand collision pieces')
        minimum = float('inf'); limiting = None
        centers = []
        for part in parts:
            part.obj.setTransform(fcl.Transform(rotation, position))
            distance = float(fcl.distance(part.obj, self.obj, self.request, fcl.DistanceResult()))
            if not np.isfinite(distance):raise ValueError('Nonfinite FCL distance')
            if distance < margin:
                return False, max(0., distance), part.label, 'intersection' if distance <= 0 else 'clearance'
            if distance < minimum: minimum = distance; limiting = part.label
            centers.append(rotation@part.center+position)
        centers = np.asarray(centers)
        in_bounds = ((centers >= self.minimum) & (centers <= self.maximum)).all(axis=1)
        if in_bounds.any():
            winding = self.winding(centers[in_bounds])
            if np.any((winding > .05) & (winding < .95)):
                raise ValueError('Ambiguous solid containment in closed skin')
            if (winding >= .5).any():
                index = np.flatnonzero(in_bounds)[np.flatnonzero(winding >= .5)[0]]
                return False, 0., parts[index].label, 'contained_in_body'
        return True, minimum, limiting, 'clear'


class ConvexFootProxy:
    """The separately recorded conservative foot/calf contact solids."""
    def __init__(self, surfaces):
        self.objects=[fcl.CollisionObject(convex_geometry(s['vertices'])[0]) for s in surfaces]
        self.request=fcl.DistanceRequest()

    def distances(self, parts, rotation, position):
        distances=np.full(len(self.objects),np.inf);limiting=[None]*len(self.objects)
        for part in parts:
            part.obj.setTransform(fcl.Transform(rotation,position))
            for i,body in enumerate(self.objects):
                if distances[i]<=0:continue
                value=float(fcl.distance(part.obj,body,self.request,fcl.DistanceResult()))
                if not np.isfinite(value):raise ValueError('Nonfinite proxy distance')
                if value<distances[i]:distances[i]=value;limiting[i]=part.label
        return distances,limiting


def candidate_samples(state, circumferential_samples=32, depths_m=(0., .005, .010, .015, .020, .025, .030), band_width=.030):
    """Interpolate actual cloth faces in material (theta,z) coordinates."""
    points = np.asarray(state['particles_world']); rest = np.asarray(state['source']['rest_positions'])
    faces = np.asarray(state['triangles']).reshape(-1, 3)
    theta = np.mod(np.arctan2(rest[:, 1], rest[:, 0]), 2*np.pi)
    material_z = rest[:, 2]-rest[:, 2].min()
    normal_sum = np.zeros_like(points)
    for face in faces:
        r = rest[face]; v = points[face]
        sign = 1 if np.dot(np.cross(r[1]-r[0], r[2]-r[0]), np.r_[r.mean(0)[:2], 0.]) >= 0 else -1
        normal_sum[face] += sign*np.cross(v[1]-v[0], v[2]-v[0])
    normal_sum /= np.maximum(np.linalg.norm(normal_sum, axis=1)[:, None], 1e-12)
    samples = []
    for depth in depths_m:
        eligible = np.flatnonzero((material_z[faces].min(1) <= depth+1e-7) & (material_z[faces].max(1) >= depth-1e-7)
                                 & (np.ptp(material_z[faces], axis=1) > 1e-7))
        for i in range(circumferential_samples):
            angle = i*2*np.pi/circumferential_samples; found = []
            for face_id in eligible:
                face = faces[face_id]
                relative = (theta[face]-angle+np.pi) % (2*np.pi)-np.pi
                if np.ptp(relative) > np.pi or relative.min() > 1e-6 or relative.max() < -1e-6:continue
                uv = np.c_[relative, material_z[face]]
                matrix = np.c_[uv[1]-uv[0], uv[2]-uv[0]]
                if abs(np.linalg.det(matrix)) < 1e-12:continue
                b, c = np.linalg.solve(matrix, np.array([0., depth])-uv[0]); weights = np.array([1-b-c, b, c])
                if weights.min() < -1e-5:continue
                active = bool(np.all(material_z[face] <= band_width+1e-6))
                derivatives = np.c_[points[face[1]]-points[face[0]], points[face[2]]-points[face[0]]]@np.linalg.inv(matrix)
                normal = unit(weights@normal_sum[face])
                along = unit(derivatives[:, 1]-normal*np.dot(normal, derivatives[:, 1]))
                found.append((active, dict(index=len(samples),theta_index=i,theta_deg=float(np.degrees(angle)),depth_m=float(depth),
                    position=(weights@points[face]).tolist(),normal=normal.tolist(),along=along.tolist(),
                    face=int(face_id),weights=weights.tolist(),native_contact_face=active)))
            if not found:raise ValueError(f'No actual cloth face at theta={angle}, material depth={depth}')
            samples.append(max(found, key=lambda x:x[0])[1])
    return samples


def orientation(sample, yaw_deg, tilt_deg, closing_sign=1):
    normal = np.asarray(sample['normal']); along = np.asarray(sample['along']); across = unit(np.cross(normal, along))
    yaw, tilt = np.radians([yaw_deg, tilt_deg])
    tangent = along*np.cos(yaw)+across*np.sin(yaw)
    approach = unit(tangent*np.cos(tilt)+normal*np.sin(tilt))
    closing = closing_sign*unit(normal-approach*np.dot(approach, normal))
    return np.column_stack([approach, closing, unit(np.cross(approach, closing))])
