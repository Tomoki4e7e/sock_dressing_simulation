"""Open-hand cloth/foot contact goals; deliberately no closure or area rule.

Distances use the actual saved surfaces and the URDF's convex hand pieces.
The rim condition concerns a local positive-length boundary segment at EACH
hand, rather than trying to put the entire opening circle in each small hand.
"""
from dataclasses import asdict, dataclass

import fcl
import numpy as np
from scipy.spatial import ConvexHull

from .geometric_closure import clip_polygon
from .grasp_cloth_geometry import ClothScreen
from .grasp_geometry import SkinBody
from .held_ankle_poses import foot_basis


@dataclass(frozen=True)
class ContactCriteria:
    contact_tolerance_m: float = .0005
    band_min_m: float = .015
    band_max_m: float = .030
    other_skin_clearance_m: float = .001
    minimum_rim_length_m: float = .000001

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or min(values) <= 0 or self.band_min_m >= self.band_max_m:
            raise ValueError('Positive finite contact criteria and an ordered band required')


def material_band(points, faces, depth, lower, upper):
    """Clip triangles at BOTH material-depth limits, including crossing faces."""
    vertices, triangles = [], []
    for face in np.asarray(faces).reshape(-1, 3):
        z = depth[face]
        if z.max() < lower or z.min() > upper:
            continue
        polygon = np.c_[points[face], z]
        polygon = clip_polygon(polygon, np.array([0., 0., 0., -1., lower]))
        polygon = clip_polygon(polygon, np.array([0., 0., 0., 1., -upper]))
        if len(polygon) < 3:
            continue
        start = len(vertices)
        vertices.extend(polygon[:, :3])
        for i in range(1, len(polygon)-1):
            if np.linalg.norm(np.cross(polygon[i, :3]-polygon[0, :3], polygon[i+1, :3]-polygon[0, :3])) > 1e-16:
                triangles.append([start, start+i, start+i+1])
    if not triangles:
        raise ValueError('No cloth surface in the requested material band')
    return np.asarray(vertices), np.asarray(triangles, dtype=int)


def opening_edges(faces, depth, tolerance=1e-7):
    """Use real mesh boundary edges, never artificial chords across the rim."""
    f = np.asarray(faces).reshape(-1, 3)
    edges, counts = np.unique(np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0, return_counts=True)
    result = edges[(counts == 1) & (np.max(depth[edges], axis=1) <= tolerance)]
    if not len(result):
        raise ValueError('No actual boundary edges at material depth zero')
    return result


def aperture_planes(negative, positive):
    """Halfspaces for points between finite opposing convex finger surfaces.

    Eliminating y gives each finger's projected footprint. Keep every lower /
    upper pair: a point must be in BOTH footprints, beyond the negative
    finger's upper envelope and below the positive finger's lower envelope.
    """
    hulls = [np.unique(np.round(ConvexHull(v).equations, 12), axis=0) for v in (negative, positive)]
    planes = []
    for h in hulls:
        flat = h[np.abs(h[:, 1]) < 1e-9]
        planes.extend(flat)
        low = h[h[:, 1] < -1e-9]
        high = h[h[:, 1] > 1e-9]
        for a in low:
            for b in high:
                projection = -a/a[1]+b/b[1]
                projection[1] = 0.
                planes.append(projection)
    # The upper surface of the negative jaw is the MIN of its upper bounds.
    # Thus its complement is a union of cells, handled by rim_intervals.
    return hulls, np.asarray(planes)


def clip_segment(a, b, planes, tolerance=1e-12):
    planes = np.asarray(planes)
    value = planes[:, :3]@a+planes[:, 3]
    slope = planes[:, :3]@(b-a)
    flat = np.abs(slope) < 1e-15
    if np.any(value[flat] > tolerance):
        return None
    negative, positive = slope < -1e-15, slope > 1e-15
    lo = max(0., float(np.max((tolerance-value[negative])/slope[negative]))) if negative.any() else 0.
    hi = min(1., float(np.min((tolerance-value[positive])/slope[positive]))) if positive.any() else 1.
    if (hi-lo)*np.linalg.norm(b-a) <= 1e-10:
        return None
    return lo, hi


def rim_intervals(local_segments, negative, positive):
    """Exact linear clipping per rim edge; union intervals without bridging."""
    hulls, projected = aperture_planes(negative, positive)
    lower_faces = hulls[0][hulls[0][:, 1] > 1e-9]
    upper_faces = hulls[1][hulls[1][:, 1] < -1e-9]
    result = []
    for edge_id, (a, b) in enumerate(local_segments):
        footprint = clip_segment(a, b, projected)
        if footprint is None:
            continue
        def merge(intervals):
            merged = []
            for lo, hi in sorted(intervals):
                if merged and lo <= merged[-1][1]+1e-10:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            return merged
        def face_union(faces):
            intervals = []
            for face in faces:
                clipped = clip_segment(a, b, -face[None, :])
                if clipped is not None:
                    lo, hi = max(footprint[0], clipped[0]), min(footprint[1], clipped[1])
                    if lo < hi:
                        intervals.append((lo, hi))
            return merge(intervals)
        lows, highs = face_union(lower_faces), face_union(upper_faces)
        merged = merge([(max(a0, b0), min(a1, b1)) for a0, a1 in lows for b0, b1 in highs if max(a0, b0) < min(a1, b1)])
        for lo, hi in merged:
            length = float(np.linalg.norm(b-a)*(hi-lo))
            if length > 1e-10:
                result.append(dict(edge_index=edge_id, t=[lo, hi], length_m=length,
                                   endpoints_local_m=[(a+(b-a)*lo).tolist(), (a+(b-a)*hi).tolist()]))
    return result


def ray_bounds(h, points):
    """Vectorized finite-hull cross sections for a smooth-ish search proxy."""
    rhs = -(points[:, 0, None]*h[None, :, 0]+points[:, 2, None]*h[None, :, 2]+h[None, :, 3])
    low = np.max(rhs[:, h[:, 1] < -1e-9]/h[None, h[:, 1] < -1e-9, 1], axis=1)
    high = np.min(rhs[:, h[:, 1] > 1e-9]/h[None, h[:, 1] > 1e-9, 1], axis=1)
    flat = np.abs(h[:, 1]) <= 1e-9
    outside = np.max(-rhs[:, flat], axis=1) if flat.any() else np.full(len(points), -1.)
    return low, high, outside


class ContactGoal:
    reference_gap = .020

    def __init__(self, state, hand, criteria=ContactCriteria()):
        self.state, self.hand, self.criteria = state, hand, criteria
        self.points = np.asarray(state['particles_world'])
        faces = np.asarray(state['triangles'], dtype=int).reshape(-1, 3)
        rest = np.asarray(state['source']['rest_positions'])
        self.depth = rest[:, 2]-rest[:, 2].min()
        vertices, triangles = material_band(self.points, faces, self.depth, criteria.band_min_m, criteria.band_max_m)
        self.band = ClothScreen(vertices, triangles)
        self.cloth = ClothScreen(self.points, faces)
        self.skin = SkinBody(state['source']['skin_surface'])
        self.rim = self.points[opening_edges(faces, self.depth)]
        self.opening_center=self.rim.reshape(-1,3).mean(0)
        inward=self.band.points.mean(0)-self.opening_center
        self.opening_inward=inward/max(np.linalg.norm(inward),1e-12)
        self.rim_samples = np.concatenate([a[None, :]+np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(b-a)/.001))+1))[:, None]*(b-a) for a, b in self.rim])
        source = state['source']
        self.basis = foot_basis(source)
        foot = np.asarray(source['surfaces'][0]['vertices'])
        self.center = (foot.min(0)+foot.max(0))/2
        self.lateral = self.basis[:, 2].copy()
        if self.lateral[0] < 0:
            self.lateral *= -1
        # User-confirmed contact region: the ENTIRE actual foot-to-calf skin,
        # without an anatomical ankle cutoff and without artificial skin caps.
        skin_v = np.asarray(source['skin_surface']['vertices'])
        skin_f = np.asarray(source['skin_surface']['triangles'], dtype=int).reshape(-1, 3)
        self.foot = ClothScreen(skin_v, skin_f)
        self.parts = hand.bank(self.reference_gap)
        self.vertices = hand.vertices(self.reference_gap)
        self.hulls = {name: ConvexHull(v).equations for name, v in self.vertices.items() if name.endswith('finger_link')}
        self.distance_request = fcl.DistanceRequest(enable_nearest_points=True)

    @staticmethod
    def coefficient(label):
        return 0. if label.endswith('/gripper_base') else (-.5 if '/left_' in label else .5)

    def place(self, p, R, gap):
        for part in self.parts:
            offset = np.array([0., self.coefficient(part.label)*(gap-self.reference_gap), 0.])
            part.obj.setTransform(fcl.Transform(R, p+R@offset))

    def distance(self, part, obj, witness=False):
        result = fcl.DistanceResult()
        d = float(fcl.distance(part.obj, obj, self.distance_request, result))
        if not np.isfinite(d):
            raise ValueError('Nonfinite contact distance')
        if d < 0:
            contacts = fcl.CollisionResult()
            fcl.collide(part.obj, obj, fcl.CollisionRequest(num_max_contacts=32, enable_contact=True), contacts)
            d = -max([abs(float(c.penetration_depth)) for c in contacts.contacts]+[1e-7])
        if witness:
            return d, [np.asarray(q).tolist() for q in result.nearest_points]
        return d

    def local_fingers(self, gap):
        fingers = []
        for name, points in self.vertices.items():
            if name.endswith('finger_link'):
                fingers.append(points+np.array([0., self.coefficient(name)*(gap-self.reference_gap), 0.]))
        return sorted(fingers, key=lambda v: v[:, 1].mean())

    def rim_proxy(self, p, R, gap):
        local = (self.rim_samples-p)@R
        bounds = []
        for name in sorted(self.hulls):
            h = self.hulls[name].copy()
            h[:, 3] -= h[:, 1]*self.coefficient(name)*(gap-self.reference_gap)
            bounds.append(ray_bounds(h, local))
        negative, positive = bounds
        # Both projected footprints must exist. The facing surfaces are the
        # negative jaw's high bound and positive jaw's low bound.
        violation = np.maximum.reduce([negative[0]-negative[1], positive[0]-positive[1],
                                       negative[2], positive[2], negative[1]-local[:, 1], local[:, 1]-positive[0]])
        return float(np.min(violation))

    def residual(self, p, R, gap, side, cloth_finger):
        self.place(p, R, gap)
        foot_finger = 'right_finger_link' if cloth_finger == 'left_finger_link' else 'left_finger_link'
        target = self.criteria.contact_tolerance_m/2
        values = []
        for part in self.parts:
            skin_d = self.distance(part, self.skin.obj)
            cloth_d = self.distance(part, self.cloth.obj)
            skin_margin = .00001 if part.label.endswith('/'+foot_finger) else self.criteria.other_skin_clearance_m
            values += [4*max(0., skin_margin-skin_d), 4*max(0., .00001-cloth_d)]
            if part.label.endswith('/'+cloth_finger):
                values.append(self.distance(part, self.band.obj)-target)
            if part.label.endswith('/'+foot_finger):
                values.append(self.distance(part, self.foot.obj)-target)
        sign = 1 if side == 'left' else -1
        values += [2*max(0., self.rim_proxy(p, R, gap)+.00005),
                   2*max(0., .002-sign*np.dot(p-self.center, self.lateral))]
        return np.asarray(values)*1000

    def inspect(self, p, R, gap, side, cloth_finger, check_rim=True):
        if side not in ('left', 'right') or cloth_finger not in ('left_finger_link', 'right_finger_link'):
            raise ValueError('Unknown arm or finger')
        if not 0 < gap <= self.hand.maximum_gap_m:
            raise ValueError('Open width outside physical range')
        self.place(p, R, gap)
        foot_finger = 'right_finger_link' if cloth_finger == 'left_finger_link' else 'left_finger_link'
        checks, contacts = [], {}
        for part in self.parts:
            sd, sw = self.distance(part, self.skin.obj, True)
            cd, cw = self.distance(part, self.cloth.obj, True)
            margin = 0. if part.label.endswith('/'+foot_finger) else self.criteria.other_skin_clearance_m
            # Also reject enclosed convex pieces (positive surface distance can
            # otherwise mistake a piece wholly inside the foot for a free one).
            center = R@(part.center+np.array([0., self.coefficient(part.label)*(gap-self.reference_gap), 0.]))+p
            winding = float(self.skin.winding(center[None, :])[0])
            valid = sd >= margin-1e-9 and cd >= -1e-9 and winding < .05
            checks.append(dict(part=part.label, skin_distance_m=sd, cloth_distance_m=cd,
                               skin_margin_m=margin, skin_winding=winding, passed=bool(valid)))
            for suffix, surface, key in [(cloth_finger, self.band.obj, 'cloth'), (foot_finger, self.foot.obj, 'foot')]:
                if part.label.endswith('/'+suffix):
                    d, witnesses = self.distance(part, surface, True)
                    contacts[key] = dict(part=part.label, distance_m=d, witnesses_m=witnesses,
                                         passed=bool(-1e-9 <= d <= self.criteria.contact_tolerance_m))
        rim = []
        if check_rim and self.rim_proxy(p, R, gap) <= .001:
            negative, positive = self.local_fingers(gap)
            # Reject distant edges by the finger bounding box before exact clips.
            local = (self.rim-p)@R
            lower = np.minimum(negative.min(0), positive.min(0))
            upper = np.maximum(negative.max(0), positive.max(0))
            mask = ((local.max(1) >= lower).all(1) & (local.min(1) <= upper).all(1))
            ids = np.flatnonzero(mask)
            rim = rim_intervals(local[mask], negative, positive)
            for segment in rim:
                segment['edge_index'] = int(ids[segment['edge_index']])
        length = sum(r['length_m'] for r in rim)
        signed_side = float((1 if side == 'left' else -1)*np.dot(p-self.center, self.lateral))
        foot_contact_side = min(float((1 if side == 'left' else -1)*np.dot(np.asarray(w)-self.center, self.lateral)) for w in contacts['foot']['witnesses_m'])
        opening_side=float(np.dot(np.mean(contacts['foot']['witnesses_m'],axis=0)-self.opening_center,self.opening_inward))
        passed = all(c['passed'] for c in checks) and all(c['passed'] for c in contacts.values()) and length >= self.criteria.minimum_rim_length_m and signed_side > 0
        return dict(passed=bool(passed), side=side, cloth_finger=cloth_finger, foot_finger=foot_finger,
                    contacts=contacts, part_checks=checks, rim_segments=rim, rim_length_m=length,
                    signed_foot_side_m=signed_side, signed_foot_contact_side_m=foot_contact_side,
                    skin_contact_opening_side_m=opening_side,
                    position_m=np.asarray(p).tolist(), rotation=np.asarray(R).tolist(),
                    opening_gap_m=float(gap), closure_performed=False, criteria=asdict(self.criteria))
