import json
from dataclasses import replace

import numpy as np
import pytest

from sock_dressing_simulation.held_ankle_poses import foot_basis, transport_reference, audit_actual_skin, load_reference
from sock_dressing_simulation.pose_patterns import anatomical_frame, read_patterns, validate_pattern_config
from test_preinsert import source


@pytest.fixture
def reference(source):
    rng = np.random.default_rng(7)
    points = rng.normal(size=(30, 3))*.05+source['toe']
    return dict(points=points, normal=np.array([0., 0., 1.]), geometry=dict(source=source))


def test_identity_reproduces_every_saved_particle(reference):
    frame = foot_basis(reference['geometry']['source'])
    pose = dict(toe_direction=frame[:, 0], dorsal_direction=frame[:, 1], ankle=reference['geometry']['source']['ankle'])
    points, normal, rotation = transport_reference(reference, pose)
    assert np.allclose(points, reference['points'], atol=1e-14)
    assert np.allclose(rotation, np.eye(3), atol=1e-14)
    assert np.allclose(normal, reference['normal'])


@pytest.mark.parametrize('pitch,roll', [(0,0), (30,-30), (40,15), (50,30)])
def test_ankle_rotation_preserves_all_lengths_without_resizing(reference, pitch, roll):
    source = reference['geometry']['source']
    forward, dorsal, _ = anatomical_frame(source, pitch, roll)
    pose = dict(toe_direction=forward, dorsal_direction=dorsal, ankle=np.array([.1,.5,.6]))
    points, normal, rotation = transport_reference(reference, pose)
    before = np.linalg.norm(reference['points'][:, None]-reference['points'][None], axis=2)
    after = np.linalg.norm(points[:, None]-points[None], axis=2)
    assert np.allclose(before, after, atol=1e-14)
    assert np.isclose(np.linalg.det(rotation), 1)
    assert np.isclose(np.linalg.norm(normal), 1)
    assert np.isclose(np.linalg.norm(points.mean(0)-pose['ankle']),
                      np.linalg.norm(reference['points'].mean(0)-source['ankle']))


def test_actual_skin_audit_detects_skin_crossing_cloth_face(source):
    skin = source['surfaces'][0]
    points = np.array([[-1.,0.,-1.],[1.,0.,-1.],[0.,0.,1.]])
    body, surface = audit_actual_skin(points, skin, [[0,1],[1,2],[2,0]], [[0,1,2]])
    assert not body['geometry_ok']
    assert surface['body_edges_crossing_cloth_faces'] > 0
    assert not surface['surface_ok']
    assert surface['method'] == 'actual_baked_skin_triangles'


def test_actual_skin_audit_accepts_separated_meshes(source):
    points = np.array([[-.1,.1,.3],[.1,.1,.3],[0.,.2,.3]])
    body, surface = audit_actual_skin(points, source['surfaces'][0], [[0,1],[1,2],[2,0]], [[0,1,2]])
    assert body['geometry_ok'] and surface['surface_ok']


def test_failed_held_reference_cannot_be_used(tmp_path):
    (tmp_path/'fitted-geometry.json').write_text('{}')
    (tmp_path/'report.json').write_text(json.dumps(dict(passed=False, stage='A')))
    (tmp_path/'fitted-prepared-state.json').write_text('{}')
    with pytest.raises(ValueError, match='validated'):
        load_reference(tmp_path)


def test_reference_is_an_explicit_config_option():
    validate_pattern_config(dict(pose_pattern=dict(workbook='patterns.xlsx', held_reference='held/A')))


def test_small_contact_slide_keeps_shape_and_size_exact():
    from sock_dressing_simulation.held_ankle_poses import seat_reference
    skin = dict(vertices=[[-.01,-.01,0],[.01,-.01,0],[.01,.01,0],[-.01,.01,0]], triangles=[0,1,2,0,2,3])
    points = np.array([[-.003,0,-.001],[.003,0,.002],[0,.005,.002]])
    original = points.copy(); edges = np.array([[0,1],[1,2],[2,0]]); faces = [[0,1,2]]
    moved, correction = seat_reference(points, skin, edges, faces, [0,0,1])
    assert np.array_equal(points, original)
    assert .001 <= np.linalg.norm(correction) < .002
    assert np.allclose(moved-points, correction)
    assert np.allclose(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]], axis=1),
                       np.linalg.norm(moved[edges[:,0]]-moved[edges[:,1]], axis=1))
    assert audit_actual_skin(moved, skin, edges, faces)[0]['geometry_ok']
    with pytest.raises(RuntimeError, match='resizing is prohibited'):
        seat_reference(points, skin, edges, faces, [0,0,1], maximum_m=.0005)


def test_empty_held_reference_config_is_rejected():
    with pytest.raises(ValueError, match='held_reference'):
        validate_pattern_config(dict(pose_pattern=dict(workbook='patterns.xlsx', held_reference='')))


def test_held_mode_cannot_fall_back_to_procedural_sock_generation(monkeypatch):
    from sock_dressing_simulation import pose_patterns, held_ankle_poses
    marker = object()
    def forbidden(*args, **kwargs):
        raise AssertionError('Held reference must not be resized or regenerated')
    monkeypatch.setattr(pose_patterns, 'generate_pattern', forbidden)
    monkeypatch.setattr(held_ankle_poses, 'initialize_held_ankle_pattern', lambda *args, **kwargs: marker)
    assert pose_patterns.initialize_pattern(object(), object(), held_reference='saved/A') is marker


def test_gravity_drape_config_requires_supported_region_and_real_settling():
    base=dict(workbook='patterns.xlsx', held_reference='saved/A')
    for invalid in [dict(settle_steps=0), dict(pinned_length_m=.3), dict(unknown=True)]:
        with pytest.raises(ValueError, match='gravity|Gravity|drape'):
            validate_pattern_config(dict(pose_pattern=dict(base, gravity_drape=invalid)))
    validate_pattern_config(dict(pose_pattern=dict(base, gravity_drape=dict(enabled=True, settle_steps=300, pinned_length_m=.075))))


@pytest.fixture
def gravity_samples():
    initial=dict(physics_steps=0,physics_time_s=0.)
    rows=[dict(physics_steps=10*(i+1),physics_time_s=.2*(i+1),gravity_world=[0,-9.81,0],
        physics_model='gravity_pbd_section_chain',dynamic_sections=18,
        free_particles=577,fixed_particles=224,free_mass_kg=.036,
        maximum_fixed_error_m=0.,tip_drop_m=.2,downward_alignment=.95,
        maximum_free_speed_m_s=.001,tip_center_world=[0,.5,0]) for i in range(6)]
    return initial,rows


def test_gravity_acceptance_requires_mass_time_drop_and_support(gravity_samples):
    from sock_dressing_simulation.held_ankle_poses import assess_gravity_drape
    initial,rows=gravity_samples
    checks,motion=assess_gravity_drape(initial,rows,801)
    assert all(checks.values()) and motion==0


@pytest.mark.parametrize('change,expected', [
    (dict(gravity_world=[0,0,0]),'gravity'),
    (dict(dynamic_sections=0),'dynamic_tail'),
    (dict(free_particles=0,fixed_particles=801,free_mass_kg=0),'dynamic_tail'),
    (dict(physics_time_s=0),'time_advanced'),
    (dict(tip_drop_m=0),'tip_dropped'),
    (dict(downward_alignment=0),'hangs_down'),
    (dict(maximum_fixed_error_m=.01),'supported_region'),
    (dict(tip_center_world=[0,.6,0]),'settled_tip'),
])
def test_floating_or_fixed_tail_cannot_pass_as_gravity_drape(gravity_samples,change,expected):
    from sock_dressing_simulation.held_ankle_poses import assess_gravity_drape
    initial,rows=gravity_samples
    rows[-1].update(change)
    checks,_=assess_gravity_drape(initial,rows,801)
    assert not checks[expected]
