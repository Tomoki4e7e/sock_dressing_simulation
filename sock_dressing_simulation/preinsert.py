"""Deterministic, unwrinkled partial dressing in the current human's world frame.

The Excel's qualitative states are deliberately not interpreted here. E's height
is an explicit assumption. Material rest lengths are never recalibrated.
"""
from __future__ import annotations

import numpy as np

STAGES = ('A', 'B', 'C', 'D', 'E')


def validate_preinsert_config(config):
    settings = config.get('preinsert')
    if not settings:
        return
    runtime = config.get('rcareworld', {})
    if runtime.get('profile') != 'custom_player' or not runtime.get('native_rcareworld'):
        raise ValueError('preinsert requires the native custom Player')
    if not isinstance(settings, dict) or set(settings)-{'stage', 'above_heel_m', 'clearance_m'}:
        raise ValueError('Unknown preinsert settings')
    if settings.get('stage') not in STAGES:
        raise ValueError('preinsert.stage must be A, B, C, D or E')
    for name, default in [('above_heel_m', .05), ('clearance_m', .015)]:
        value = float(settings.get(name, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'preinsert.{name} must be finite and positive')
    sock = config.get('scenario', {}).get('sock', {})
    if float(sock.get('rest_bend_degrees', 0)) != 0 or not sock.get('closed_toe', False):
        raise ValueError('preinsert requires an unbent rest mesh with closed_toe: true')


def _vectors(value, name):
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite Nx3 vectors')
    return result


def _unit(value):
    length = np.linalg.norm(value)
    if length < 1e-8:
        raise ValueError('Degenerate anatomical frame')
    return value / length


def _slab_points(vertices, triangles, center, tangent, half_width):
    """Clip triangle edges to a slab; includes faces crossing an empty slice."""
    edges = vertices[triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)]
    depth = (edges - center) @ tangent
    selected = [vertices[np.abs((vertices-center) @ tangent) <= half_width]]
    for plane in (-half_width, half_width):
        delta = depth[:, 1] - depth[:, 0]
        valid = (np.abs(delta) > 1e-12) & ((depth[:, 0]-plane)*(depth[:, 1]-plane) <= 0)
        t = (plane-depth[valid, 0]) / delta[valid]
        selected.append(edges[valid, 0] + t[:, None]*(edges[valid, 1]-edges[valid, 0]))
    return np.concatenate(selected)


def generate_preinsert(source, stage, *, above_heel_m=0.05, clearance_m=0.015):
    """Return actor-indexed positions and measured placement metadata.

    Source comes from GetSockPreinsertSource, not inferred image landmarks.
    A-C divide the smooth anatomical path from toe tip to the heel/ankle
    junction. D surrounds that junction. E continues up the calf.
    The unused sock length extends past the toes, avoiding artificial bunching.
    """
    if stage not in STAGES:
        raise ValueError('stage must be A, B, C, D or E')
    if not np.isfinite([above_heel_m, clearance_m]).all() or above_heel_m <= 0 or clearance_m <= 0:
        raise ValueError('Height and clearance must be finite and positive')
    rest = _vectors(source['rest_positions'], 'rest_positions')
    surfaces = [(_vectors(s['vertices'], 'surface vertices'), np.asarray(s['triangles'], dtype=int).reshape(-1, 3))
                for s in source['surfaces']]
    if not surfaces or any(len(v) < 4 or len(f) < 4 or f.min() < 0 or f.max() >= len(v) for v, f in surfaces):
        raise ValueError('Invalid anatomical surface mesh')
    foot_vertices = surfaces[0][0]
    ankle, heel, toe, calf = _vectors([source[k] for k in ('ankle', 'heel', 'toe', 'calf')], 'landmarks')
    outward = _unit(toe-heel)
    lateral = _unit(np.cross(outward, calf-ankle))
    inward = -outward
    toe_tip = toe + outward * np.max((foot_vertices-toe) @ outward)
    foot_length = float((toe_tip-heel) @ outward)
    if foot_length <= 0.05:
        raise ValueError('Toe/heel landmarks do not define a usable foot length')
    # A cuff at the heel must already face the shin. A straight tube there
    # would have to enclose the entire shin in its cross-section. Turn smoothly
    # before reaching the ankle and place D just above the foot's dorsal skin.
    calf_axis = _unit(calf-ankle)
    # The bone's heel offset is not on the ankle-to-toe line. Use the ankle
    # line for the sweep so there is no frame discontinuity at the toe or at
    # the start of the bend.
    outward = _unit(toe_tip-ankle)
    inward = -outward
    lateral = _unit(np.cross(outward, calf_axis))
    turn_start = ankle + outward * 0.09
    dorsal_height = max(0., float(np.max((foot_vertices-ankle) @ calf_axis))) + clearance_m
    turn_end = ankle + calf_axis*dorsal_height
    control = ankle
    ts = np.linspace(0, 1, 129)
    curve = ((1-ts[:, None])**2*turn_start + 2*(1-ts[:, None])*ts[:, None]*control + ts[:, None]**2*turn_end)
    # Project start onto the toe axis so the initial straight section joins.
    # Anatomical lateral offsets are blended over this segment as well.
    lead = float(np.linalg.norm(toe_tip-turn_start))
    curve_s = lead + np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    heel_path = float(curve_s[-1])
    depth = {'A': .25*heel_path, 'B': .5*heel_path, 'C': .75*heel_path,
             'D': heel_path, 'E': heel_path+above_heel_m}[stage]
    zmin, zmax = rest[:, 2].min(), rest[:, 2].max()
    length = float(zmax-zmin)
    radius = np.linalg.norm(rest[:, :2], axis=1)
    nominal_radius = float(np.median(radius[radius > 1e-6]))
    axial_scale = max(1., (depth+nominal_radius*.5+clearance_m)/length)
    if axial_scale > 1.5:
        raise ValueError(f'Sock length {length:.3f} m is insufficient for stage {stage} ({depth:.3f} m)')
    levels = np.unique(np.round(rest[:, 2], 6))
    if len(levels) < 3 or nominal_radius <= 0:
        raise ValueError('A straight tubular rest mesh with at least three rings is required')
    radial_counts = [np.count_nonzero(np.isclose(rest[:, 2], z, atol=1e-6, rtol=0) & (radius > 1e-6)) for z in levels]
    segments = min(radial_counts)
    if segments < 8:
        raise ValueError('At least eight particles per ring are required')
    # Straight-rest guard also rejects an ambiguous cap or bent blueprint.
    if np.max(np.abs(np.mean(rest[radius > 1e-6, :2], axis=0))) > 1e-5:
        raise ValueError('Preinsertion requires an unbent, centered rest mesh')

    def frame(s):
        if s <= 0:
            return toe_tip + inward*s, inward
        if s >= heel_path:
            return turn_end + calf_axis*(s-heel_path), calf_axis
        if s < lead:
            tangent = _unit(turn_start-toe_tip)
            return toe_tip + tangent*s, tangent
        center = np.array([np.interp(s, curve_s, curve[:, i]) for i in range(3)])
        i = min(max(int(np.searchsorted(curve_s, s)), 1), len(curve)-1)
        return center, _unit(curve[i]-curve[i-1])

    positions = np.empty_like(rest)
    rings = []
    loft = []
    for z in levels:
        ids = np.flatnonzero(np.isclose(rest[:, 2], z, atol=1e-6, rtol=0))
        s = depth-(z-zmin)*axial_scale
        center, tangent = frame(s)
        base_center = center.copy()
        cross = _unit(np.cross(lateral, tangent))
        # Include a whole inter-ring slab, not just vertex samples at its center.
        half_slab = length*axial_scale/(len(levels)-1)*.55
        sections = [_slab_points(v, f, center, tangent, half_slab) for v, f in surfaces]
        section = np.concatenate(sections)
        if len(section):
            relative = section-center
            u, v = relative @ lateral, relative @ cross
            # Center the ellipse on the anatomical section, rather than forcing
            # it around the bone pivot (which is often near the skin boundary).
            shift_u, shift_v = (u.max()+u.min())*.5, (v.max()+v.min())*.5
            center = center + lateral*shift_u + cross*shift_v
            u, v = u-shift_u, v-shift_v
            a = max(nominal_radius, float(np.max(np.abs(u)))+clearance_m)
            b = max(nominal_radius, float(np.max(np.abs(v)))+clearance_m)
            factor = max(1., float(np.max(np.sqrt((u/a)**2+(v/b)**2))))
            a *= factor / np.cos(np.pi/segments)
            b *= factor / np.cos(np.pi/segments)
        else:
            a = b = nominal_radius
        theta = np.arctan2(rest[ids, 1], rest[ids, 0])
        cap = radius[ids] <= 1e-6
        positions[ids] = center + np.cos(theta)[:, None]*lateral*a + np.sin(theta)[:, None]*cross*b
        positions[ids[cap]] = center
        rings.append(dict(s_m=float(s), center_world=center.tolist(), half_width_m=a, half_height_m=b))
        loft.append((ids, cap, theta, base_center, cross,
                     float((center-base_center) @ lateral), float((center-base_center) @ cross)))
    # Proxy boxes and the toe cap introduce abrupt section changes. Ease those
    # changes over neighboring rings; never shrink the original enclosing
    # section. This avoids stretching a single axial edge across a sharp step.
    shifts = np.array([[row[5], row[6]] for row in loft])
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)/9
    smoothed = np.column_stack([np.convolve(np.pad(shifts[:, i], (2, 2), mode='edge'), kernel, mode='valid') for i in range(2)])
    radii = np.array([[r['half_width_m'], r['half_height_m']] for r in rings])
    original_radii = radii.copy()
    radii += np.abs(smoothed-shifts)
    angles = np.linspace(0, 2*np.pi, 129)
    for i in range(len(rings)):
        du, dv = shifts[i]-smoothed[i]
        factor = max(1., float(np.sqrt(((du+original_radii[i, 0]*np.cos(angles))/radii[i, 0])**2+
                                      ((dv+original_radii[i, 1]*np.sin(angles))/radii[i, 1])**2).max()))
        radii[i] *= factor
    stations = np.array([r['s_m'] for r in rings])
    for i in range(len(rings)):
        radii[i] = np.max(radii-np.abs(stations-stations[i])[:, None]*.25, axis=0)
    for i, (ids, cap, theta, base_center, cross, _, _) in enumerate(loft):
        center = base_center + lateral*smoothed[i, 0] + cross*smoothed[i, 1]
        a, b = radii[i]
        positions[ids] = center + np.cos(theta)[:, None]*lateral*a + np.sin(theta)[:, None]*cross*b
        positions[ids[cap]] = center
        rings[i].update(center_world=center.tolist(), half_width_m=float(a), half_height_m=float(b))
    opening_ids = np.flatnonzero(np.isclose(rest[:, 2], zmin, atol=1e-6, rtol=0))
    _, tangent = frame(depth)
    cuff = np.asarray(rings[0]['center_world'])
    report = dict(stage=stage, opening_shape='normal_ellipse', wrinkles='none_requested', cuff_fold='none',
                  foot_length_m=foot_length, insertion_path_m=depth, above_heel_m=above_heel_m,
                  clearance_m=clearance_m, cuff_center_world=cuff.tolist(),
                  opening_inward_normal_world=(-tangent).tolist(), opening_particle_indices=opening_ids.tolist(),
                  unused_length_beyond_toe_m=length*axial_scale-depth, rings=rings,
                  axial_initial_scale=axial_scale, heel_path_m=heel_path,
                  heel_cuff_above_ankle_m=dorsal_height,
                  material_rest_lengths_modified=False, physical_settling_validated=False,
                  definition='A-C: fractions of toe-to-heel/ankle path; D: heel/ankle junction above dorsal skin; E: D plus calf-axis height')
    return positions, report


def audit_preinsert(positions, source, edges, rest_lengths):
    """Independent convex-volume audit of cloth particles and material edges.

    Checks whole cloth edges, so an outside-to-outside segment crossing the foot
    is rejected as well. This is a geometric gate, not a physics validation.
    """
    points = _vectors(positions, 'particles')
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if len(edges) == 0 or edges.min() < 0 or edges.max() >= len(points):
        raise ValueError('Native cloth edges are required for the geometry audit')
    rest = np.asarray(rest_lengths, dtype=float)
    if rest.shape != (len(edges),) or not np.isfinite(rest).all() or np.any(rest <= 0):
        raise ValueError('Native positive rest edge lengths are required')
    inside = np.zeros(len(points), dtype=bool)
    crossed = np.zeros(len(edges), dtype=bool)
    maximum_penetration = 0.
    for surface in source['surfaces']:
        vertices = _vectors(surface['vertices'], 'surface vertices')
        faces = np.asarray(surface['triangles'], dtype=int).reshape(-1, 3)
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        normal = np.cross(b-a, c-a)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        normal[np.einsum('ij,ij->i', normal, a-vertices.mean(axis=0)) < 0] *= -1
        offset = np.einsum('ij,ij->i', normal, a)
        distances = points @ normal.T-offset
        interior = np.max(distances, axis=1) < -1e-6
        inside |= interior
        if interior.any(): maximum_penetration = max(maximum_penetration, float(-distances[interior].max(axis=1).min()))
        # Clip each complete edge against all halfspaces of the convex solid.
        start = distances[edges[:, 0]]
        delta = distances[edges[:, 1]]-start
        entry, leave = np.zeros(len(edges)), np.ones(len(edges))
        possible = np.ones(len(edges), dtype=bool)
        for i in range(len(normal)):
            parallel = np.abs(delta[:, i]) < 1e-12
            possible &= ~(parallel & (start[:, i] >= -1e-6))
            t = np.divide(-1e-6-start[:, i], delta[:, i], out=np.zeros(len(edges)), where=~parallel)
            entry = np.where(delta[:, i] < -1e-12, np.maximum(entry, t), entry)
            leave = np.where(delta[:, i] > 1e-12, np.minimum(leave, t), leave)
        crossed |= possible & (leave-entry > 1e-6)
    strain = np.linalg.norm(points[edges[:, 0]]-points[edges[:, 1]], axis=1)/rest
    return dict(geometry_ok=not inside.any() and not crossed.any(),
                particles_inside_body=int(inside.sum()), edges_crossing_body=int(crossed.sum()),
                maximum_particle_penetration_m=maximum_penetration,
                maximum_edge_stretch=float(strain.max()),
                p95_edge_stretch=float(np.percentile(strain, 95)),
                physics_settling_validated=False)


def _segment_triangle_hits(starts, ends, vertices, faces, segment_ids=None, face_ids=None):
    """Moller-Trumbore finite segments, excluding shared mesh vertices."""
    hit_segments = np.zeros(len(starts), dtype=bool)
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    e1, e2 = b-a, c-a
    for first in range(0, len(starts), 64):
        last = first+64
        origin = starts[first:last, None, :]
        direction = (ends[first:last]-starts[first:last])[:, None, :]
        h = np.cross(direction, e2)
        det = np.einsum('tij,ij->ti', h, e1)
        inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det) > 1e-12)
        offset = origin-a
        u = np.einsum('tij,tij->ti', offset, h)*inv
        q = np.cross(offset, e1)
        v = np.sum(direction*q, axis=2)*inv
        t = np.einsum('ij,tij->ti', e2, q)*inv
        hits = (np.abs(det) > 1e-12) & (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8) & (t > 1e-6) & (t < 1-1e-6)
        if segment_ids is not None:
            shared = (segment_ids[first:last, :, None, None] == face_ids[None, None, :, :]).any(axis=(1, 3))
            hits &= ~shared
        hit_segments[first:last] = hits.any(axis=1)
    return hit_segments


def audit_preinsert_surface(positions, source, edges, triangles):
    """Check body edges through cloth faces and nonadjacent cloth intersections."""
    points = _vectors(positions, 'particles')
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    faces = np.unique(np.sort(np.asarray(triangles, dtype=int).reshape(-1, 3), axis=1), axis=0)
    body_crossings = 0
    for surface in source['surfaces']:
        vertices = _vectors(surface['vertices'], 'surface vertices')
        body_faces = np.asarray(surface['triangles'], dtype=int).reshape(-1, 3)
        body_edges = np.unique(np.sort(body_faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
        body_crossings += int(_segment_triangle_hits(vertices[body_edges[:, 0]], vertices[body_edges[:, 1]], points, faces).sum())
    self_crossings = int(_segment_triangle_hits(points[edges[:, 0]], points[edges[:, 1]], points, faces,
                                               segment_ids=edges, face_ids=faces).sum())
    return dict(body_edges_crossing_cloth_faces=body_crossings,
                self_intersecting_cloth_edges=self_crossings,
                surface_ok=body_crossings == 0 and self_crossings == 0)
