"""Saved held-A initialization with supported-foot, gravity-driven tail poses."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .preinsert import audit_preinsert, audit_preinsert_surface, _segment_triangle_hits


def foot_basis(source):
    forward = np.asarray(source['toe'], dtype=float)-source['ankle']
    forward /= np.linalg.norm(forward)
    dorsal = np.asarray(source['calf'], dtype=float)-source['ankle']
    dorsal -= forward*np.dot(dorsal, forward)
    if np.linalg.norm(dorsal) < 1e-6:
        dorsal = np.array([0., 1., 0.])-forward*forward[1]
    if dorsal[1] < 0: dorsal *= -1
    dorsal /= np.linalg.norm(dorsal)
    return np.column_stack([forward, dorsal, np.cross(forward, dorsal)])


def load_reference(directory):
    directory = Path(directory)
    raw = (directory/'fitted-geometry.json').read_bytes()
    geometry = json.loads(raw)
    report = json.loads((directory/'report.json').read_text())
    state = json.loads((directory/'fitted-prepared-state.json').read_text())
    if not report.get('passed') or report.get('stage') != 'A':
        raise ValueError('Ankle-only mode requires a validated saved held-A reference')
    points = np.asarray(geometry['particles_world'], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Invalid held reference geometry')
    normal = state['openingNormal']
    normal = np.array([normal[k] for k in ('x', 'y', 'z')]) if isinstance(normal, dict) else np.asarray(normal)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
        raise ValueError('Invalid held reference opening normal')
    if len(points) != len(geometry['source']['rest_positions']):
        raise ValueError('Held reference particle count differs from its source mesh')
    return dict(directory=str(directory.resolve()), geometry=geometry, points=points,
                normal=normal, sha256=hashlib.sha256(raw).hexdigest())


def transport_reference(reference, native_pose):
    """One proper rigid transform: every length and the existing shape survive."""
    source = reference['geometry']['source']
    forward = np.asarray(native_pose['toe_direction'], dtype=float)
    forward /= np.linalg.norm(forward)
    dorsal = np.asarray(native_pose['dorsal_direction'], dtype=float)
    dorsal -= forward*np.dot(dorsal, forward); dorsal /= np.linalg.norm(dorsal)
    target = np.column_stack([forward, dorsal, np.cross(forward, dorsal)])
    rotation = target@foot_basis(source).T
    origin = np.asarray(source['ankle'])
    ankle = np.asarray(native_pose['ankle'])
    points = (reference['points']-origin)@rotation.T+ankle
    normal = rotation@reference['normal']; normal /= np.linalg.norm(normal)
    return points, normal, rotation


def _perimeter(points, rest):
    rest = np.asarray(rest)
    ids = np.flatnonzero(np.isclose(rest[:, 2], rest[:, 2].min(), atol=1e-6, rtol=0))
    ids = ids[np.argsort(np.arctan2(rest[ids, 1], rest[ids, 0]))]
    cuff = points[ids]
    return float(np.linalg.norm(cuff-np.roll(cuff, 1, axis=0), axis=1).sum()), ids


def _skin_context(points, skin, padding=0.):
    vertices = np.asarray(skin['vertices'], dtype=float)
    faces = np.asarray(skin['triangles'], dtype=int).reshape(-1, 3)
    bounds = vertices[faces]
    active = ((bounds.max(axis=1) >= points.min(axis=0)-padding) &
              (bounds.min(axis=1) <= points.max(axis=0)+padding)).all(axis=1)
    faces = faces[active]
    skin_edges = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    return vertices, faces, skin_edges


def _skin_crossings(points, context, edges, triangles):
    vertices, faces, skin_edges = context
    cloth_crossings = int(_segment_triangle_hits(points[edges[:, 0]], points[edges[:, 1]], vertices, faces).sum())
    skin_crossings = int(_segment_triangle_hits(vertices[skin_edges[:, 0]], vertices[skin_edges[:, 1]], points, triangles).sum())
    return cloth_crossings, skin_crossings


def seat_reference(points, skin, edges, triangles, toe_direction, maximum_m=.008):
    """Keep size/shape exact; permit a bounded, millimetre-scale axial slide.

    Foot skin blends with a stationary shin when the ankle rotates. A slight
    rigid slide can clear that skin without inflating or deforming the sock.
    """
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    triangles = np.asarray(triangles, dtype=int).reshape(-1, 3)
    direction = np.asarray(toe_direction, dtype=float); direction /= np.linalg.norm(direction)
    context = _skin_context(points, skin, padding=maximum_m+.0001)
    if sum(_skin_crossings(points, context, edges, triangles)) == 0:
        return points, np.zeros(3)
    for distance in np.arange(.0005, maximum_m+.00001, .0005):
        correction = direction*min(float(distance)+.0001, maximum_m)
        candidate = points+correction
        if sum(_skin_crossings(candidate, context, edges, triangles)) == 0:
            return candidate, correction
    raise RuntimeError(f'Held-A geometry cannot clear actual skin within {maximum_m*1000:g} mm; resizing is prohibited')


def audit_actual_skin(points, skin, edges, triangles):
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    triangles = np.asarray(triangles, dtype=int).reshape(-1, 3)
    cloth_crossings, skin_crossings = _skin_crossings(points, _skin_context(points, skin), edges, triangles)
    surface = audit_preinsert_surface(points, dict(surfaces=[]), edges, triangles)
    surface.update(body_edges_crossing_cloth_faces=skin_crossings, cloth_edges_crossing_skin_faces=cloth_crossings,
        surface_ok=cloth_crossings == 0 and skin_crossings == 0 and surface['self_intersecting_cloth_edges'] == 0,
        method='actual_baked_skin_triangles')
    body = dict(geometry_ok=cloth_crossings == 0 and skin_crossings == 0,
                cloth_edges_crossing_skin_faces=cloth_crossings, skin_edges_crossing_cloth_faces=skin_crossings,
                method='actual_baked_skin_triangles')
    return body, surface


def initialize_held_ankle_pattern(environment, pattern, held_reference, signs=None, gravity_drape=None):
    from .circumference_fit import current_pose_hull
    from .config import resolve_package_path
    attr = environment.sock_cloth
    if attr is None or environment._cloth_contact_rollout_started:
        raise RuntimeError('Initialize the held reference before a contact rollout')
    reference = load_reference(resolve_package_path(held_reference))
    signs = signs or dict(shift=1, diagonal=1, roll=1)
    attr.set_pattern_held(False)
    attr.set_pattern_foot_pose(pattern.pitch_deg, signs['roll']*pattern.roll_deg)
    environment._env.step(simulate=False)
    attr.request_preinsert_source(); attr.request_fit_skin(); attr.request_particles()
    attr._send_data("GetSockPatternSkin")
    environment._env.step(simulate=False)
    source = copy.deepcopy(attr.data['sock_preinsert_source'])
    rest = np.asarray(source['rest_positions'])
    saved_rest = np.asarray(reference['geometry']['source']['rest_positions'])
    if rest.shape != saved_rest.shape or not np.allclose(rest, saved_rest, atol=1e-6, rtol=0):
        raise ValueError('Actor particle ordering/rest mesh differs from held reference')
    source['skin_surface'] = copy.deepcopy(attr.data['sock_pattern_skin'])
    source['surfaces'] = []
    for key in ('sock_fit_skin', 'sock_fit_calf_skin'):
        vertices, faces = current_pose_hull(attr.data[key])
        source['surfaces'].append(dict(vertices=vertices.tolist(), triangles=faces.tolist()))
    pose = copy.deepcopy(attr.data['sock_pattern_pose'])
    points, normal, rotation = transport_reference(reference, pose)
    edges = np.asarray(attr.data['particle_edges'], dtype=int).reshape(-1, 2)
    triangles = np.asarray(attr.data['particle_surface_triangles'], dtype=int).reshape(-1, 3)
    reference_edges = np.unique(np.sort(np.asarray(reference['geometry']['edges']).reshape(-1, 2), axis=1), axis=0)
    if not np.array_equal(reference_edges, np.unique(np.sort(edges, axis=1), axis=0)):
        raise ValueError('Held reference topology does not match this cloth')
    points, contact_translation = seat_reference(points, source['skin_surface'], edges, triangles, pose['toe_direction'])
    convex_audit = audit_preinsert(points, source, edges, attr.data['particle_rest_edge_lengths'])
    body, surface = audit_actual_skin(points, source['skin_surface'], edges, triangles)
    if not body['geometry_ok'] or not surface['surface_ok']:
        environment._held_reference_collision = dict(points=points.tolist(), source=source, edges=edges.tolist(), triangles=triangles.tolist(), pose=pose)
        raise RuntimeError(f'Rigid held-A pose fails geometry checks: {body}; {surface}')
    attr.initialize_preinsert(points, normal)
    environment._env.step(simulate=False)
    attr.set_pattern_held(True); attr.request_particles()
    environment._env.step(simulate=False)
    readback = attr.particles()
    error = float(np.linalg.norm(readback-points, axis=1).max())
    lengths = np.linalg.norm(readback[edges[:, 0]]-readback[edges[:, 1]], axis=1)
    original = reference['points']
    reference_lengths = np.linalg.norm(original[edges[:, 0]]-original[edges[:, 1]], axis=1)
    length_error = float(np.max(np.abs(lengths-reference_lengths)))
    circumference, cuff_ids = _perimeter(readback, rest)
    reference_circumference, _ = _perimeter(original, saved_rest)
    measured_pitch = float(np.degrees(np.arcsin(np.clip(pose['toe_direction'][1], -1, 1))))
    if (error > 5e-6 or length_error > 2e-6 or abs(circumference-reference_circumference) > 5e-6
            or abs(measured_pitch-pattern.pitch_deg) > .05
            or not attr.data.get('sock_pattern_hold', {}).get('enabled')):
        raise RuntimeError('Held reference geometry/pose changed during native readback')
    body, surface = audit_actual_skin(readback, source['skin_surface'], edges, triangles)
    if not body['geometry_ok'] or not surface['surface_ok']:
        raise RuntimeError('Native held reference readback intersects the actual skin or itself')
    report = dict(stage='A', pattern=asdict(pattern), motion_mode='held_reference_ankle_only',
        applied_conditions=dict(stage='A', pitch_deg=pattern.pitch_deg, roll_deg=signs['roll']*pattern.roll_deg,
            shift_cm=0, diagonal_deg=0, added_wrinkles=False, added_collapse=False, added_twist=False, added_fold=False),
        ignored_workbook_conditions=['sock_id', 'stage', 'shift_cm', 'diagonal_deg', 'opening', 'wrinkles', 'fold'],
        reference_directory=reference['directory'], reference_geometry_sha256=reference['sha256'],
        rigid_rotation=rotation.tolist(), scale=1., maximum_edge_length_error_m=length_error,
        contact_translation_world=contact_translation.tolist(), contact_translation_m=float(np.linalg.norm(contact_translation)),
        maximum_allowed_contact_translation_m=.008,
        baseline_cuff_perimeter_m=reference_circumference, cuff_perimeter_m=circumference,
        cuff_center_world=readback[cuff_ids].mean(axis=0).tolist(), opening_particle_indices=cuff_ids.tolist(),
        opening_inward_normal_world=normal.tolist(), geometry_audit=body, surface_audit=surface,
        conservative_convex_hull_diagnostic=convex_audit,
        local_features=[], local_feature_count=0, changed_particle_fraction=0.,
        clearance_m=0., clearance_mode='preserve_saved_held_shape',
        maximum_readback_error_m=error, native_pose=pose, measured_pitch_deg=measured_pitch,
        native_hold=copy.deepcopy(attr.data['sock_pattern_hold']), physical_settling_validated=False)
    if gravity_drape and gravity_drape.get('enabled', True):
        report = apply_gravity_drape(environment, reference, source, report, gravity_drape)
    environment._preinsert_observation_pending = True
    environment._pose_pattern_report = report; environment._pose_pattern_source = source
    return report


def apply_gravity_drape(environment, reference, source, report, settings):
    attr = environment.sock_cloth
    state = (Path(reference['directory'])/'fitted-prepared-state.json').read_text()
    attr._send_data('BeginSockPatternGravityDrape', state, float(settings.get('pinned_length_m', .075)))
    environment._env.step(simulate=False)
    initial = copy.deepcopy(attr.data['sock_pattern_gravity_drape'])
    samples = []
    steps = int(settings.get('settle_steps', 300))
    for first in range(0, steps, 10):
        environment.advance_physics(min(10, steps-first))
        attr._send_data('GetSockPatternGravityDrape'); environment._env.step(simulate=False)
        samples.append(copy.deepcopy(attr.data['sock_pattern_gravity_drape']))
        if (first+10) % 50 == 0:
            latest=samples[-1]
            print(f"drape steps={first+10}/{steps} drop={latest['tip_drop_m']:.3f}m alignment={latest['downward_alignment']:.3f} speed={latest['maximum_free_speed_m_s']:.3f}m/s", flush=True)
    final = samples[-1]
    checks, tip_motion = assess_gravity_drape(initial, samples, len(reference['points']))
    attr.request_particles(); environment._env.step(simulate=False)
    points = attr.particles()
    body, surface = audit_actual_skin(points, source['skin_surface'], attr.data['particle_edges'], attr.data['particle_surface_triangles'])
    if body['geometry_ok'] and not surface['surface_ok']:
        for blend in (3.,4.,6.,8.):
            attr._send_data('SetSockPatternDrapeBoundaryBlend',blend);environment._env.step(simulate=False)
            environment.advance_physics(1)
            attr._send_data('GetSockPatternGravityDrape');attr.request_particles();environment._env.step(simulate=False)
            final=copy.deepcopy(attr.data['sock_pattern_gravity_drape']);samples.append(final)
            points=attr.particles()
            body,surface=audit_actual_skin(points,source['skin_surface'],attr.data['particle_edges'],attr.data['particle_surface_triangles'])
            if body['geometry_ok'] and surface['surface_ok']:break
        checks,tip_motion=assess_gravity_drape(initial,samples,len(reference['points']))
    circumference, _ = _perimeter(points, source['rest_positions'])
    checks.update(no_skin_intersection=body['geometry_ok'], no_self_intersection=surface['surface_ok'],
                  cuff_size_preserved=abs(circumference-report['baseline_cuff_perimeter_m']) < 5e-6)
    result = dict(passed=all(bool(x) for x in checks.values()), checks={k:bool(v) for k,v in checks.items()},
                  initial=initial, final=final, samples=samples, last_second_tip_motion_m=tip_motion)
    environment._gravity_drape_debug = dict(result=result, points=points.tolist(), source=source,
        edges=attr.data['particle_edges'], triangles=attr.data['particle_surface_triangles'])
    if not result['passed']:
        raise RuntimeError(f'Gravity drape failed: {result["checks"]}; final={final}; motion={tip_motion}')
    report.update(motion_mode='held_reference_gravity_drape', gravity_drape=result,
        native_hold=dict(mode='fixed_wrapped_region_dynamic_tail', enabled=True,
                         fixed_particles=final['fixed_particles'], free_particles=final['free_particles']),
        geometry_audit=body, surface_audit=surface, cuff_perimeter_m=circumference,
        physical_settling_validated=True, validated_physics_model=final['physics_model'],
        reference_rigid_checks_apply_to='before_gravity_drape')
    return report


def assess_gravity_drape(initial, samples, particle_count):
    if len(samples) < 6:
        return dict(sufficient_samples=False), float('inf')
    final=samples[-1]
    tips=np.array([x['tip_center_world'] for x in samples[-6:]])
    if tips.shape != (6,3) or not np.isfinite(tips).all():
        return dict(finite_tip_samples=False), float('inf')
    tip_motion=float(np.linalg.norm(tips-tips[0],axis=1).max())
    keys=('physics_steps','physics_time_s','free_particles','free_mass_kg','fixed_particles',
          'maximum_fixed_error_m','tip_drop_m','downward_alignment','maximum_free_speed_m_s')
    if not all(k in final and np.isfinite(final[k]) for k in keys):
        return dict(finite_physics_data=False), tip_motion
    checks=dict(gravity=np.allclose(final['gravity_world'],[0,-9.81,0]),
        time_advanced=final['physics_steps']>initial['physics_steps'] and final['physics_time_s']>initial['physics_time_s'],
        dynamic_tail=0<final['free_particles']<particle_count and .005<final['free_mass_kg']<.05
            and final.get('physics_model')=='gravity_pbd_section_chain' and final.get('dynamic_sections',0)>0,
        supported_region=0<final['fixed_particles']<particle_count and final['maximum_fixed_error_m']<5e-6
            and final['free_particles']+final['fixed_particles']==particle_count,
        tip_dropped=final['tip_drop_m']>.02,
        hangs_down=final['downward_alignment']>.80,
        settled_tip=tip_motion<.005)
    return checks,tip_motion
