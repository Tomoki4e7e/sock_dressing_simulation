import copy
from dataclasses import replace, asdict
import json

import numpy as np
import pytest

from sock_dressing_simulation.held_stage_shapes import (
    PROFILE, build_bulged_stage_state, load_stage_state, state_directory, current_shape_qualified,
    qualify_native_stage_fit,
    assess_stage_gravity,
    tighten_supported_surface,
)
from sock_dressing_simulation.held_ankle_poses import load_reference
from sock_dressing_simulation.pose_patterns import read_patterns
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settled_surface():
    theta = np.arange(16) * 2*np.pi/16
    rest = np.array([[.04*np.cos(a), .04*np.sin(a), z]
                     for z in (0., .0125, .025, .0375, .05) for a in theta])
    faces = []
    for section in range(4):
        for j in range(16):
            a = section*16+j
            b = section*16+(j+1)%16
            faces.extend([[a,b,a+16], [b,b+16,a+16]])
    skin = dict(vertices=[[10.,0.,0.],[11.,0.,0.],[10.,1.,0.]], triangles=[0,1,2])
    return dict(source=dict(rest_positions=rest.tolist(), skin_surface=skin),
                particles_world=rest.tolist(), triangles=np.array(faces).ravel().tolist(), surface_fit=dict(passed=True))


@pytest.mark.parametrize('stage', list('CDE'))
def test_each_stage_uses_A_bulge_and_preserves_the_other_material(settled_surface, stage):
    fit = dict(stage=stage, passed=True, rest_circumference_scale=.63)
    settling = dict(stage=stage, physical_settling_validated=True,
                    validated_physics_model='gravity_pbd_section_chain', gravity_drape=dict(passed=True))
    old = copy.deepcopy(settled_surface)
    state, before, profile = build_bulged_stage_state(settled_surface, fit, settling, stage)
    assert old == settled_surface
    rest = np.array(state['source']['rest_positions'])
    points = np.array(state['particles_world'])
    depth = rest[:,2]-rest[:,2].min()
    outside = (depth <= 1e-10) | (depth >= .03-1e-10)
    assert np.array_equal(points[outside], before[outside])
    assert np.linalg.norm(points-before, axis=1).max() == pytest.approx(.005)
    assert np.allclose(np.linalg.norm((points-before)[np.isclose(depth,.015)], axis=1), .005)
    assert state['applied_opening_stage'] == stage
    assert current_shape_qualified(stage, state)
    assert state['geometry_audit']['geometry_ok'] and state['surface_audit']['surface_ok']


def test_unverified_physics_or_wrong_stage_cannot_become_a_current_shape(settled_surface):
    settling = dict(stage='C', physical_settling_validated=True, validated_physics_model='native')
    with pytest.raises(ValueError, match='circumference fit'):
        build_bulged_stage_state(settled_surface, dict(stage='D',passed=True), settling, 'C')
    with pytest.raises(ValueError, match='gravity preparation'):
        build_bulged_stage_state(settled_surface, dict(stage='C',passed=True), {}, 'C')


def test_stage_cache_rejects_old_ellipses_other_stages_and_other_ankles(tmp_path):
    pattern = replace(read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')[0], stage='C')
    path = state_directory(tmp_path, pattern.number, 'C')/'state.json'
    path.parent.mkdir()
    state = dict(shape_profile=PROFILE, applied_opening_stage='C', pattern=asdict(pattern),
                 surface_fit=dict(passed=True),
                 preparation=dict(fit_passed=True,settling_passed=True))
    path.write_text(json.dumps(state))
    assert load_stage_state(tmp_path, pattern, 'C')['applied_opening_stage'] == 'C'
    for field, value in [('shape_profile','old_ellipse'), ('applied_opening_stage','D')]:
        wrong = dict(state, **{field:value})
        path.write_text(json.dumps(wrong))
        with pytest.raises(ValueError, match='mismatched'):
            load_stage_state(tmp_path,pattern,'C')
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='mismatched'):
        load_stage_state(tmp_path,replace(pattern,roll_deg=pattern.roll_deg+5),'C')


@pytest.mark.parametrize('stage', list('CDE'))
def test_validated_saved_references_require_the_requested_stage(stage):
    directory = ROOT/'artifacts/sock-pose-pattern/preinsert-fitted'/stage
    assert len(load_reference(directory,stage=stage)['points']) == 801
    with pytest.raises(ValueError,match='held-A'):
        load_reference(directory)


def test_CDE_old_shapes_are_no_longer_current_but_A_is_unchanged():
    assert all(not current_shape_qualified(s,dict(mode='frozen_normal_preinsert')) for s in 'CDE')
    assert current_shape_qualified('A',dict(mode='frozen_cuff_bulge_5mm_center15mm'))


def test_skin_requalification_does_not_bypass_physics_or_inside_particles(settled_surface):
    points = np.array(settled_surface['particles_world'])
    faces = np.array(settled_surface['triangles']).reshape(-1,3)
    edges = np.unique(np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0)
    original = dict(assessment=dict(passed=True),
                    geometry_audit=dict(geometry_ok=False,particles_inside_body=0,edges_crossing_body=1),
                    surface_audit=dict(surface_ok=False,body_edges_crossing_cloth_faces=1))
    def qualify(report):
        return qualify_native_stage_fit(report,settled_surface['source'],points,edges,faces)
    assert qualify(original)['passed']
    assert not qualify(dict(original,assessment=dict(passed=False)))['passed']
    assert not qualify(dict(original,geometry_audit=dict(particles_inside_body=1)))['passed']
    assert original['geometry_audit']['edges_crossing_body'] == 1


@pytest.mark.parametrize('stage',list('CDE'))
def test_endpoint_cache_cannot_skip_preparation_by_reusing_an_old_stage(tmp_path,stage):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from search_diverse_left_endpoints import run
    from sock_dressing_simulation.reaching_dataset import file_sha256
    initial = ROOT/'config/diverse_left_reaching_shared.yaml'
    (tmp_path/f'pose-10-{stage}.json').write_text(json.dumps(dict(initial_sha256=file_sha256(initial))))
    with pytest.raises(ValueError,match='previous circumference'):
        run([10],[stage],tmp_path,initial,seconds=1e-9)


def test_short_free_end_is_physical_without_A_long_tail_minima():
    initial = dict(physics_steps=0,physics_time_s=0.)
    final = dict(gravity_world=[0.,-9.81,0.],physics_steps=150,physics_time_s=3.,
                 physics_model='gravity_pbd_section_chain',dynamic_sections=2,free_mass_kg=.002,
                 fixed_particles=736,free_particles=65,maximum_fixed_error_m=0.,
                 maximum_free_speed_m_s=.001,tip_drop_m=.003)
    assert all(assess_stage_gravity(initial,final,801,.0001,1.).values())
    for field,value in [('gravity_world',[0.,0.,0.]),('physics_steps',0),('free_mass_kg',0.),
                        ('maximum_fixed_error_m',.001),('maximum_free_speed_m_s',.2)]:
        assert not all(assess_stage_gravity(initial,dict(final,**{field:value}),801,.0001,1.).values())
    assert not assess_stage_gravity(initial,final,801,.01,1.)['settled']


def test_tighter_skin_sections_reduce_clearance_and_keep_tail_boundary_exact():
    import trimesh
    theta = np.arange(16)*2*np.pi/16
    levels = np.arange(17)*.0125
    rest = np.array([[.04*np.cos(t),.04*np.sin(t),z] for z in levels for t in theta])
    faces = []
    for k in range(len(levels)-1):
        for j in range(16):
            a,b = k*16+j,k*16+(j+1)%16
            faces += [[a,b,a+16],[b,b+16,a+16]]
    faces = np.array(faces)
    edges = np.unique(np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0)
    skin = trimesh.creation.cylinder(radius=.025,height=.14,sections=32)
    skin.apply_translation([0.,0.,.04])
    state = dict(source=dict(rest_positions=rest.tolist(),skin_surface=dict(vertices=skin.vertices.tolist(),
        triangles=skin.faces.ravel().tolist())),particles_world=rest.tolist(),triangles=faces.ravel().tolist(),
        edges=edges.ravel().tolist())
    original = copy.deepcopy(state)
    fitted = tighten_supported_surface(state,support_material_length_m=.15)
    report = fitted['surface_fit']
    assert state == original
    assert report['passed'] and report['body']['geometry_ok'] and report['surface']['surface_ok']
    assert report['after_perimeter_m'] < .9*report['before_perimeter_m']
    assert report['after_skin_gap']['median_m'] < report['before_skin_gap']['median_m']
    fixed = rest[:,2] >= .125-1e-6
    assert np.array_equal(np.array(fitted['particles_world'])[fixed],rest[fixed])
    assert not report['adjusted_shape_unpinned_retention_validated']
    with pytest.raises(ValueError,match='strength'):
        tighten_supported_surface(state,strength=1.1,support_material_length_m=.15)


def test_reduced_drape_gravity_must_match_the_recorded_scale():
    initial = dict(physics_steps=0,physics_time_s=0.)
    final = dict(gravity_world=[0.,-9.81*.5,0.],physics_steps=150,physics_time_s=3.,
                 physics_model='gravity_pbd_section_chain',dynamic_sections=2,free_mass_kg=.002,
                 fixed_particles=736,free_particles=65,maximum_fixed_error_m=0.,maximum_free_speed_m_s=.001)
    assert all(assess_stage_gravity(initial,final,801,.0001,1.,gravity_scale=.5).values())
    assert not assess_stage_gravity(initial,final,801,.0001,1.)['gravity']
    assert not assess_stage_gravity(initial,final,801,.0001,1.,gravity_scale=0)['gravity']


@pytest.mark.parametrize('hull_ok,skin_ok,self_ok', [
    (True, True, True), (False, True, True), (True, False, True), (True, True, False),
])
def test_stage_initial_hull_surface_fallback_still_requires_containment_and_actual_skin(
        monkeypatch, tmp_path, hull_ok, skin_ok, self_ok):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / 'scripts'))
    import prepare_held_stage_shapes as script
    from sock_dressing_simulation import preinsert
    points = np.zeros((4, 3))
    initialized = []
    attr = SimpleNamespace(
        data=dict(sock_preinsert_source={}, sock_pattern_skin={}, particle_edges=[],
                  particle_surface_triangles=[], particle_rest_edge_lengths=[],
                  sock_preinsert_report=dict(ok=True)),
        request_preinsert_source=lambda: None, _send_data=lambda *args: None,
        release=lambda *args: None, request_particles=lambda: None,
        initialize_preinsert=lambda *args: initialized.append(args), particles=lambda: points)
    def conservative_rejection(stage):
        raise RuntimeError('Preinsert geometry intersects the body or itself')
    env = SimpleNamespace(sock_cloth=attr, initialize_preinsert=conservative_rejection,
                          _env=SimpleNamespace(step=lambda **kw: None))
    monkeypatch.setattr(preinsert, 'generate_preinsert', lambda *a, **kw:
                        (points, dict(opening_inward_normal_world=[0., 0., 1.])))
    monkeypatch.setattr(preinsert, 'audit_preinsert', lambda *a: dict(geometry_ok=hull_ok))
    monkeypatch.setattr(preinsert, 'audit_preinsert_surface', lambda *a: dict(surface_ok=False))
    monkeypatch.setattr(script, 'audit_actual_skin', lambda *a:
                        (dict(geometry_ok=skin_ok), dict(surface_ok=skin_ok and self_ok)))
    if hull_ok and skin_ok and self_ok:
        report = script.initialize_stage_preinsert(env, 'E', tmp_path)
        assert report['passed'] and len(initialized) == 1
        assert not report['conservative_convex_surface_audit']['surface_ok']
        assert report['maximum_readback_error_m'] == 0
    else:
        with pytest.raises(RuntimeError, match='actual skin or self-intersection'):
            script.initialize_stage_preinsert(env, 'E', tmp_path)
        assert not initialized


def test_stage_initial_clearance_search_stops_at_first_safe_envelope(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / 'scripts'))
    import prepare_held_stage_shapes as script
    from sock_dressing_simulation import preinsert
    attempted, initialized = [], []
    points = np.zeros((4, 3))
    attr = SimpleNamespace(
        data=dict(sock_preinsert_source={}, sock_pattern_skin={}, particle_edges=[],
                  particle_surface_triangles=[], particle_rest_edge_lengths=[], sock_preinsert_report=dict(ok=True)),
        request_preinsert_source=lambda: None, _send_data=lambda *a: None, release=lambda *a: None,
        request_particles=lambda: None, initialize_preinsert=lambda *a: initialized.append(a),
        particles=lambda: points)
    def reject(stage):
        raise RuntimeError('Preinsert geometry intersects the body or itself')
    def generate(source, stage, clearance_m):
        attempted.append(clearance_m)
        return points, dict(clearance_m=clearance_m, opening_inward_normal_world=[0., 0., 1.])
    env = SimpleNamespace(sock_cloth=attr, initialize_preinsert=reject,
                          _env=SimpleNamespace(step=lambda **kw: None))
    monkeypatch.setattr(preinsert, 'generate_preinsert', generate)
    monkeypatch.setattr(preinsert, 'audit_preinsert', lambda *a: dict(geometry_ok=attempted[-1] >= .025))
    monkeypatch.setattr(preinsert, 'audit_preinsert_surface', lambda *a: dict(surface_ok=False))
    monkeypatch.setattr(script, 'audit_actual_skin', lambda *a: (dict(geometry_ok=True), dict(surface_ok=True)))
    report = script.initialize_stage_preinsert(env, 'E', tmp_path)
    assert attempted == [.015, .0175, .020, .025]
    assert report['passed'] and len(initialized) == 1
    assert [r['passed'] for r in report['initial_clearance_trials']] == [False, False, False, True]
