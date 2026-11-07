"""Stage C--E references prepared with A's native fit and gravity-tail method.

The final bulged surface is frozen for geometric reaching, as it is for A.
Native material fitting and toe-supported settling precede that deformation.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .cuff_bulge import refine_sections, radial_bulge
from .held_ankle_poses import audit_actual_skin, _skin_context, _skin_crossings, _perimeter

PROFILE = 'CDE_skin_section_tight_fit_supported_gravity_cuff_bulge_v3'
STAGES = ('C', 'D', 'E')
DEFAULT_ROOT = Path('artifacts/cde-tight-fit')
BASELINE_ROOT = Path('artifacts/cde-a-method')


def tighten_supported_surface(state, *, strength=.75, clearance_m=.0025, support_material_length_m=None):
    """Reduce excess sleeve circumference against the actual posed skin.

    This adjusts the supported, frozen geometry after native preparation. The
    final two supported rings and the entire gravity tail stay exactly fixed,
    preserving the measured gravity-tail geometry after the supported fit.
    Each local move is tested against skin before it is accepted.
    """
    from scipy.spatial import ConvexHull
    from .preinsert import _slab_points
    from .toe_contact import select_toe_support
    import trimesh
    if not np.isfinite([strength, clearance_m]).all() or not 0 < strength <= 1 or clearance_m <= 0:
        raise ValueError('Finite positive skin clearance and strength within (0, 1] required')
    result = copy.deepcopy(state)
    rest = np.asarray(state['source']['rest_positions'])
    original = np.asarray(state['particles_world'])
    points, target = original.copy(), original.copy()
    depth = rest[:, 2]-rest[:, 2].min()
    levels = np.unique(np.round(depth, 6))
    spacing = float(np.median(np.diff(levels)))
    boundary = (select_toe_support(original, state['source'])['material_max_m']
                if support_material_length_m is None else float(support_material_length_m))
    fit_end = boundary-2*spacing
    skin = state['source']['skin_surface']
    vertices = np.asarray(skin['vertices'])
    skin_faces = np.asarray(skin['triangles']).reshape(-1, 3)
    faces = np.asarray(state['triangles']).reshape(-1, 3)
    edges = np.asarray(state['edges']).reshape(-1, 2)
    for level in levels:
        if level >= fit_end-1e-6:
            continue
        ids = np.flatnonzero(np.isclose(depth, level, atol=1e-6, rtol=0))
        theta = np.arctan2(rest[ids, 1], rest[ids, 0])
        center = original[ids].mean(0)
        q = original[ids]-center
        lateral = (q*np.cos(theta)[:, None]).sum(0)
        cross = (q*np.sin(theta)[:, None]).sum(0)
        if np.linalg.norm(lateral) < 1e-9:
            continue
        lateral /= np.linalg.norm(lateral)
        cross -= lateral*(cross@lateral)
        if np.linalg.norm(cross) < 1e-9:
            continue
        cross /= np.linalg.norm(cross)
        samples = _slab_points(vertices, skin_faces, center, np.cross(lateral, cross), spacing*.85)
        if len(samples) < 3:
            continue
        local = np.c_[(samples-center)@lateral, (samples-center)@cross]
        if np.linalg.matrix_rank(local-local.mean(0), tol=1e-8) < 2:
            continue
        hull = ConvexHull(local)
        shift = local[hull.vertices].mean(0)
        bounds = -(hull.equations[:, 2]+hull.equations[:, :2]@shift)+clearance_m
        directions = np.c_[np.cos(theta), np.sin(theta)]
        den = directions@hull.equations[:, :2].T
        radii = np.min(np.divide(bounds[None, :], den, out=np.full_like(den, np.inf), where=den>1e-10), axis=1)
        radii = radii/np.cos(np.pi/len(ids))+.0005
        fitted = center+lateral*shift[0]+cross*shift[1]+radii[:, None]*(directions[:, :1]*lateral+directions[:, 1:]*cross)
        amount = strength*np.clip((fit_end-level)/(3*spacing), 0., 1.)
        target[ids] = (1-amount)*original[ids]+amount*fitted
    moves = []
    for pass_index in range(2):
        for level in levels:
            ids = np.flatnonzero(np.isclose(depth, level, atol=1e-6, rtol=0))
            delta = target[ids]-points[ids]
            if np.linalg.norm(delta, axis=1).max() < 1e-8:
                continue
            active = np.zeros(len(points), dtype=bool)
            active[ids] = True
            local_edges = edges[active[edges].any(1)]
            local_faces = faces[active[faces].any(1)]
            context = _skin_context(np.vstack([points[local_faces.ravel()], target[ids]]), skin, padding=.001)
            for fraction in (1., .75, .5, .25, .125, .0625):
                trial = points.copy()
                trial[ids] += fraction*delta
                if sum(_skin_crossings(trial, context, local_edges, local_faces)) == 0:
                    points = trial
                    moves.append(dict(pass_index=pass_index, depth_m=float(level), fraction=fraction))
                    break
    body, surface = audit_actual_skin(points, skin, edges, faces)
    unchanged = depth >= fit_end-1e-6
    mesh = trimesh.Trimesh(vertices, skin_faces, process=False)
    measured = depth <= fit_end+1e-6
    def gaps(p):
        distances = trimesh.proximity.closest_point_naive(mesh, p[measured])[1]
        return dict(minimum_m=float(distances.min()), median_m=float(np.median(distances)),
                    percentile95_m=float(np.quantile(distances, .95)))
    before_gap, after_gap = gaps(original), gaps(points)
    before_perimeter = _perimeter(original, rest)[0]
    after_perimeter = _perimeter(points, rest)[0]
    report = dict(passed=bool(body['geometry_ok'] and surface['surface_ok']
                             and after_perimeter < before_perimeter
                             and after_gap['median_m'] < before_gap['median_m']
                             and np.array_equal(points[unchanged], original[unchanged])),
        method='posed_skin_section_envelope_with_local_crossing_gate', strength=strength,
        target_clearance_m=clearance_m, support_boundary_m=boundary, unchanged_from_depth_m=fit_end,
        before_perimeter_m=before_perimeter, after_perimeter_m=after_perimeter,
        before_skin_gap=before_gap, after_skin_gap=after_gap, accepted_moves=moves,
        body=body, surface=surface, adjusted_shape_unpinned_retention_validated=False)
    if not report['passed']:
        raise ValueError(f'Skin fitting failed or did not improve fit: {report}')
    result.update(particles_world=points.tolist(), surface_fit=report)
    return result


def state_directory(root, number, stage):
    if stage not in STAGES:
        raise ValueError('A-method stage preparation requires C, D or E')
    return Path(root) / f'pose-{number:02d}-{stage}'


def qualify_native_stage_fit(report, source, points, edges, triangles):
    """Use A's actual-skin surface gate; keep convex-hull crossings diagnostic.

    A coarse convex hull can protrude through a cloth triangle even though the
    actual skin is separated. Native contact, finite mass, gravity, tension and
    retention still must pass, and no particle may be inside the contact hull.
    """
    result = copy.deepcopy(report)
    body, surface = audit_actual_skin(np.asarray(points), source['skin_surface'], edges, triangles)
    result['conservative_convex_geometry_audit'] = result['geometry_audit']
    result['conservative_convex_surface_audit'] = result['surface_audit']
    result['geometry_audit'] = body
    result['surface_audit'] = surface
    result['passed'] = bool(result['assessment']['passed']
                            and result['conservative_convex_geometry_audit']['particles_inside_body'] == 0
                            and body['geometry_ok'] and surface['surface_ok'])
    result['physical_retention_validated'] = result['passed']
    return result


def assess_stage_gravity(initial, final, particle_count, excursion, window_s, gravity_scale=1.):
    return dict(gravity=bool(0 < gravity_scale <= 1 and np.allclose(final['gravity_world'], [0., -9.81*gravity_scale, 0.])),
        time_advanced=final['physics_steps'] > initial['physics_steps'] and final['physics_time_s'] > initial['physics_time_s'],
        dynamic_tail=final['physics_model'] == 'gravity_pbd_section_chain' and final['dynamic_sections'] > 0
            and 0 < final['free_mass_kg'] < .05,
        supported_region=0 < final['fixed_particles'] < particle_count
            and final['fixed_particles']+final['free_particles'] == particle_count
            and final['maximum_fixed_error_m'] < 5e-6,
        settled=final['maximum_free_speed_m_s'] < .05 and excursion < .002 and window_s >= 1.-1e-5)


def apply_stage_gravity_drape(environment, reference, source, report, settings):
    """A's supported-region section chain with this stage's toe boundary.

    C--E can have only one or two free sections. Their mass and possible drop
    follow their remaining material length, rather than A's long-tail minima.
    """
    from .toe_contact import select_toe_support
    from .held_ankle_poses import _perimeter
    attr = environment.sock_cloth
    points = attr.particles().copy()
    support = select_toe_support(points, source)
    depth = np.asarray(source['rest_positions'])[:, 2]
    depth = depth-depth.min()
    pinned_length = support['material_max_m']
    if pinned_length >= depth.max()-1e-6:
        raise ValueError('The stage has no free material beyond the selected toe support')
    state = (Path(reference['directory'])/'fitted-prepared-state.json').read_text()
    attr._send_data('BeginSockHeldStageGravityDrape', state, float(pinned_length))
    gravity_scale = float(settings.get('gravity_scale', 1.))
    if not np.isfinite(gravity_scale) or not 0 < gravity_scale <= 1:
        raise ValueError('Drape gravity scale must be in (0, 1]')
    attr._send_data('SetSockHeldStageGravityScale', gravity_scale)
    attr._send_data('GetSockPatternGravityDrape')
    environment._env.step(simulate=False)
    initial = copy.deepcopy(attr.data['sock_pattern_gravity_drape'])
    samples, recent = [], []
    steps = int(settings.get('settle_steps', 150))
    for first in range(0, steps, 10):
        environment.advance_physics(min(10, steps-first))
        attr._send_data('GetSockPatternGravityDrape')
        attr.request_particles()
        environment._env.step(simulate=False)
        row = copy.deepcopy(attr.data['sock_pattern_gravity_drape'])
        samples.append(row)
        recent.append((float(row['physics_time_s']), attr.particles().copy()))
        recent = recent[-6:]
        if (first+10) % 50 == 0:
            print(f'stage gravity {first+10}/{steps}: free_speed={row["maximum_free_speed_m_s"]:.4g} m/s', flush=True)
    final = samples[-1]
    points = attr.particles()
    body, surface = audit_actual_skin(points, source['skin_surface'], attr.data['particle_edges'], attr.data['particle_surface_triangles'])
    boundary_trials = [dict(blend_intervals=4, body=body, surface=surface)]
    if not body['geometry_ok'] or not surface['surface_ok']:
        # The same boundary-blend exploration used by A's toe-tail preparation.
        for blend in (2., 3., 6., 8.):
            attr._send_data('SetSockPatternDrapeBoundaryBlend', blend)
            environment._env.step(simulate=False)
            environment.advance_physics(5)
            attr.request_particles()
            environment._env.step(simulate=False)
            body, surface = audit_actual_skin(attr.particles(), source['skin_surface'],
                attr.data['particle_edges'], attr.data['particle_surface_triangles'])
            boundary_trials.append(dict(blend_intervals=blend, body=body, surface=surface))
            if body['geometry_ok'] and surface['surface_ok']:
                # Measure a fresh full second after changing the surface map.
                recent = []
                for _ in range(6):
                    environment.advance_physics(10)
                    attr._send_data('GetSockPatternGravityDrape')
                    attr.request_particles()
                    environment._env.step(simulate=False)
                    final = copy.deepcopy(attr.data['sock_pattern_gravity_drape'])
                    samples.append(final)
                    recent.append((float(final['physics_time_s']), attr.particles().copy()))
                points = attr.particles()
                body, surface = audit_actual_skin(points, source['skin_surface'],
                    attr.data['particle_edges'], attr.data['particle_surface_triangles'])
                if body['geometry_ok'] and surface['surface_ok']:
                    break
    excursion = float(max(np.linalg.norm(p-recent[0][1], axis=1).max() for _, p in recent))
    checks = assess_stage_gravity(initial, final, len(points), excursion, recent[-1][0]-recent[0][0], gravity_scale)
    checks.update(no_skin_intersection=body['geometry_ok'], no_self_intersection=surface['surface_ok'])
    result = dict(passed=all(checks.values()), checks=checks, initial=initial, final=final,
                  samples=samples, support=support, whole_cloth_last_second_excursion_m=excursion,
                  supported_material_length_m=pinned_length,
                  boundary_trials=boundary_trials,
                  gravity_scale=gravity_scale,
                  model='A supported-region gravity section chain; short free length retained for C--E')
    environment._gravity_drape_debug = dict(result=result, points=points.tolist(), source=source,
        edges=attr.data['particle_edges'], triangles=attr.data['particle_surface_triangles'])
    if not result['passed']:
        raise RuntimeError(f'Stage gravity preparation failed: {checks}')
    report.update(motion_mode='held_reference_gravity_drape', gravity_drape=result,
        native_hold=dict(mode='fixed_wrapped_region_dynamic_tail', enabled=True,
                         fixed_particles=final['fixed_particles'], free_particles=final['free_particles']),
        geometry_audit=body, surface_audit=surface, cuff_perimeter_m=_perimeter(points, source['rest_positions'])[0],
        physical_settling_validated=True, validated_physics_model=final['physics_model'])
    return report


def build_bulged_stage_state(state, fit_report, settling_report, stage):
    """Apply precisely A's refinement and radial profile to a settled surface."""
    if stage not in STAGES or fit_report.get('stage') != stage or not fit_report.get('passed'):
        raise ValueError('A passing native circumference fit for this stage is required')
    if (settling_report.get('stage') != stage or not settling_report.get('physical_settling_validated')
            or settling_report.get('validated_physics_model') != 'gravity_pbd_section_chain'
            or not settling_report.get('gravity_drape', {}).get('passed')):
        raise ValueError('A passing native toe-supported gravity preparation is required')
    if not state.get('surface_fit', {}).get('passed'):
        raise ValueError('A passing supported skin-section fit is required')
    result = copy.deepcopy(state)
    original_rest = np.asarray(state['source']['rest_positions'])
    original = np.asarray(state['particles_world'])
    rest, before, faces = refine_sections(original_rest, original, state['triangles'])
    after, profile = radial_bulge(original_rest, original, rest, before, width=.030, peak=.005)
    edges = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    body, surface = audit_actual_skin(after, state['source']['skin_surface'], edges, faces)
    if not body['geometry_ok'] or not surface['surface_ok']:
        raise ValueError(f'{stage} bulged shape intersects skin or itself: {body}; {surface}')
    result['source']['rest_positions'] = rest.tolist()
    result.update(mode='frozen_A_method_stage_cuff_bulge', shape_profile=PROFILE,
                  applied_opening_stage=stage, particles_world=after.tolist(),
                  triangles=faces.ravel().tolist(), edges=edges.ravel().tolist(),
                  rest_edge_lengths=np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1).tolist(),
                  bulge=dict(width_m=.030, center_m=.015, peak_radius_increment_m=.005,
                             profile='sine_squared', fixed_geometry_only=True),
                  preparation=dict(method='A_native_fit_and_supported_region_gravity',
                                   stage=stage, fit_passed=True, settling_passed=True,
                                   rest_circumference_scale=fit_report['rest_circumference_scale'],
                                   physics_model=settling_report['validated_physics_model'],
                                   surface_fit_passed=True,
                                   adjusted_shape_unpinned_retention_validated=False,
                                   final_bulge_is_frozen_geometry=True),
                  geometry_audit=body, surface_audit=surface)
    return result, before, profile


def load_stage_state(root, pattern, stage):
    """Never silently reuse the old ellipse or a different ankle/stage."""
    path = state_directory(root, pattern.number, stage) / 'state.json'
    state = json.loads(path.read_text())
    saved = state.get('pattern', {})
    if (state.get('shape_profile') != PROFILE or state.get('applied_opening_stage') != stage
            or any(saved.get(k) != asdict(pattern)[k] for k in ('pitch_deg', 'roll_deg'))
            or not state.get('preparation', {}).get('fit_passed')
            or not state.get('preparation', {}).get('settling_passed')):
        raise ValueError(f'Stale or mismatched A-method stage shape: {path}')
    if not state.get('surface_fit', {}).get('passed'):
        raise ValueError(f'Stale or mismatched skin-section fit: {path}')
    state['pattern'] = asdict(pattern)
    return state


def ensure_stage_state(root, pattern, stage):
    """Prepare missing native references instead of falling back to an ellipse."""
    path = state_directory(root, pattern.number, stage) / 'state.json'
    if not path.exists():
        import subprocess
        import sys
        package_root = Path(__file__).resolve().parents[1]
        subprocess.run([sys.executable, '-u', str(package_root / 'scripts/prepare_held_stage_shapes.py'),
                        '--numbers', str(pattern.number), '--stages', stage,
                        '--output', str(Path(root).resolve())], cwd=package_root, check=True)
    return load_stage_state(root, pattern, stage)


def current_shape_profile(stage):
    if stage == 'B':
        from .fitted_b_cuff import PROFILE as b_profile
        return b_profile
    return PROFILE if stage in STAGES else None


def current_shape_qualified(stage, state):
    expected = current_shape_profile(stage)
    return expected is None or state.get('shape_profile') == expected
