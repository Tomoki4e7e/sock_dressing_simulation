"""Geometric counterexamples for the open-contact task, independent of search."""
import itertools
import csv
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('fcl')
from sock_dressing_simulation.contact_pregrasp import (
    ContactCriteria, ContactGoal, aperture_planes, clip_segment, material_band, opening_edges,
    ray_bounds, rim_intervals,
)


def box(lower, upper):
    return np.array(list(itertools.product(*zip(lower, upper))))


def test_material_band_clips_a_face_crossing_both_limits():
    points = np.array([[0., 0., 0.], [1., 0., .04], [0., 1., .04]])
    vertices, triangles = material_band(points, [[0, 1, 2]], np.array([0., .04, .04]), .015, .030)
    assert vertices[:, 2].min() == pytest.approx(.015)
    assert vertices[:, 2].max() == pytest.approx(.030)
    # Cross sections scale linearly with depth; exact trapezoidal area.
    area_xy = sum(abs(np.cross(vertices[b]-vertices[a], vertices[c]-vertices[a])[2])/2 for a, b, c in triangles)
    assert area_xy == pytest.approx(((.030/.04)**2-(.015/.04)**2)/2)


def test_opening_uses_actual_boundary_edges_and_excludes_internal_chord():
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    edges = opening_edges(faces, np.zeros(4))
    assert {tuple(e) for e in edges} == {(0, 1), (1, 2), (2, 3), (0, 3)}
    with pytest.raises(ValueError, match='boundary'):
        opening_edges(faces, np.ones(4))


def test_rim_clipping_respects_both_finite_jaw_footprints_and_aperture():
    negative = box([-2., -2., -1.], [1., -1., 1.])
    positive = box([-1., 1., -.5], [2., 2., .5])
    segments = np.array([
        [[-3., 0., 0.], [3., 0., 0.]],  # Only the shared x interval [-1, 1].
        [[0., 0., .75], [.5, 0., .75]],  # Outside positive jaw footprint.
        [[0., 1.5, 0.], [.5, 1.5, 0.]],  # Inside positive jaw solid.
        [[0., -1.5, 0.], [.5, -1.5, 0.]],  # Inside negative jaw solid.
    ])
    result = rim_intervals(segments, negative, positive)
    assert len(result) == 1
    assert result[0]['edge_index'] == 0
    assert result[0]['length_m'] == pytest.approx(2.)
    np.testing.assert_allclose(result[0]['endpoints_local_m'], [[-1., 0., 0.], [1., 0., 0.]], atol=1e-10)


def test_two_disconnected_rim_segments_are_not_joined_across_an_outside_gap():
    negative = box([-1., -2., -1.], [1., -1., 1.])
    positive = box([-1., 1., -1.], [1., 2., 1.])
    segments = np.array([[[-.8, 0., 0.], [-.6, 0., 0.]], [[.6, 0., 0.], [.8, 0., 0.]]])
    result = rim_intervals(segments, negative, positive)
    assert len(result) == 2
    assert sum(r['length_m'] for r in result) == pytest.approx(.4)


def test_degenerate_segment_at_a_corner_is_not_a_boundary_interval():
    assert clip_segment(np.array([-1., 1., 0.]), np.array([1., -1., 0.]),
                        np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]])) is None


def test_contact_tolerance_is_half_a_millimetre_and_invalid_settings_rejected():
    assert ContactCriteria().contact_tolerance_m == .0005
    with pytest.raises(ValueError):
        ContactCriteria(contact_tolerance_m=float('nan'))
    with pytest.raises(ValueError):
        ContactCriteria(band_min_m=.04)


def test_skin_contact_region_keeps_calf_triangles_beyond_anatomical_foot():
    import trimesh
    mesh = trimesh.creation.box(extents=[1., 1., 3.])
    skin = dict(vertices=mesh.vertices.tolist(), triangles=mesh.faces.ravel().tolist())
    rest = np.array([[-1., -1., 0.], [1., -1., 0.], [-1., 1., .04], [1., 1., .04]])
    foot_vertices = box([-.5, -.5, -.5], [.5, .5, .5])
    state = dict(particles_world=rest.tolist(), triangles=[0, 1, 2, 1, 3, 2],
                 source=dict(rest_positions=rest.tolist(), ankle=[0., 0., 0.], toe=[0., 0., 1.],
                             calf=[0., 1., 0.], surfaces=[dict(vertices=foot_vertices.tolist())], skin_surface=skin))
    hand = SimpleNamespace(bank=lambda gap: [], vertices=lambda gap: {
        'right_gripper/left_finger_link': box([-1., -2., -1.], [1., -1., 1.]),
        'right_gripper/right_finger_link': box([-1., 1., -1.], [1., 2., 1.]),
    })
    goal = ContactGoal(state, hand)
    np.testing.assert_array_equal(goal.foot.faces, mesh.faces)


def test_contact_policy_only_changes_the_named_foot_finger_skin_pairs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import contact_scene
    pairs = [
        ('left_gripper/left_finger_link', 'human_full_skin', .001),
        ('left_gripper/right_finger_link', 'human_full_skin', .001),
        ('left_gripper/left_finger_link', 'right_leg_actual_skin_rigid', .001),
        ('left_gripper/left_finger_link', 'frozen_cloth', .0001),
        ('right_gripper/left_finger_link', 'human_full_skin', .001),
        ('left_arm/link_5', 'human_full_skin', .005),
    ]
    scene = SimpleNamespace(pairs=pairs)
    contact_scene(scene, {'left': 'left_finger_link'})
    assert [p[2] for p in scene.pairs] == [0., .001, 0., .0001, .001, .005]


def test_export_prepares_jaws_at_start_and_keeps_widths_fixed_during_both_approaches(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import export_route_csv
    initial = dict(left_q_rad=[0.]*7, right_q_rad=[0.]*7, left_gap_m=.04, right_gap_m=.04)
    result = dict(terminal=dict(left=dict(opening_gap_m=.02), right=dict(opening_gap_m=.03)),
                  segments=[dict(side='left', times_s=[0., 1.], q_rad=[[0.]*7, [.1]*7], duration_s=1.),
                            dict(side='right', times_s=[0., 1.], q_rad=[[0.]*7, [.2]*7], duration_s=1.)])
    path = tmp_path/'trajectory.csv'
    export_route_csv(path, initial, result)
    rows = list(csv.DictReader(path.open()))
    assert np.all(np.diff([float(r['time_s']) for r in rows]) > 0)
    for row in rows:
        if row['stage'].startswith('prepare'):
            assert all(float(row[f'{side}_joint_{i}_rad']) == 0 for side in ('left', 'right') for i in range(1, 8))
        if row['stage'].startswith('approach'):
            assert float(row['left_gap_m']) == pytest.approx(.02)
            assert float(row['right_gap_m']) == pytest.approx(.03)
    assert float(rows[-1]['left_joint_1_rad']) == .1
    assert float(rows[-1]['right_joint_1_rad']) == .2


def test_shared_initial_transition_happens_open_before_width_preparation(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import export_route_csv
    initial = dict(left_q_rad=[0.]*7, right_q_rad=[0.]*7, left_gap_m=.04, right_gap_m=.04)
    result = dict(terminal=dict(left=dict(opening_gap_m=.02), right=dict(opening_gap_m=.03)),
                  initial_transition_segments=[dict(side='left', times_s=[0., 1.], q_rad=[[0.]*7, [-.1]*7], duration_s=1.)],
                  segments=[dict(side='left', times_s=[0., 1.], q_rad=[[-.1]*7, [.1]*7], duration_s=1.),
                            dict(side='right', times_s=[0., 1.], q_rad=[[0.]*7, [.2]*7], duration_s=1.)])
    path = tmp_path/'bridge.csv'
    export_route_csv(path, initial, result)
    rows = list(csv.DictReader(path.open()))
    assert rows[1]['stage'] == 'initial_transition_left'
    assert float(rows[1]['left_gap_m']) == .04
    assert float(rows[1]['right_gap_m']) == .04
    prepared = next(r for r in rows if r['stage'].startswith('prepare'))
    assert float(prepared['left_joint_1_rad']) == -.1
    assert np.all(np.diff([float(r['time_s']) for r in rows]) > 0)


def test_rrt_can_use_smaller_explicit_margin_without_crossing_physical_limits():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from plan_fixed_torso_route import rrt
    model = SimpleNamespace(lower=-np.ones(7), upper=np.ones(7))
    scene = SimpleNamespace(is_clear=lambda *args: True)
    start = np.zeros(7); start[0] = .99
    target = start.copy(); target[0] = .98
    assert not rrt(model, scene, start, target, .04, np.zeros(3), iterations=1)['passed']
    result = rrt(model, scene, start, target, .04, np.zeros(3), iterations=1,joint_margin_deg=.5)
    assert result['passed']
    assert np.max(np.asarray(result['raw_q'])) < 1.
    with pytest.raises(ValueError):
        rrt(model, scene, start, target, .04, np.zeros(3),joint_margin_deg=-1.)


def test_pair_subset_preserves_actual_shoulder_triangle_guard(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import retain_blocking_pairs
    from sock_dressing_simulation.fixed_torso_reaching import FixedTorsoScene
    from sock_dressing_simulation.compound_reaching import CompoundScene
    full = [('arm', 'cloth', 0.), ('other', 'body', .005), ('right_arm/link_2', 'torso/link_3', .005)]
    scene = FixedTorsoScene.__new__(FixedTorsoScene)
    scene.pairs = full
    scene.shoulder_pairs = {'right':2}
    scene.assert_fixed = lambda:None
    scene.shoulder_distance = lambda q,side:.00498
    monkeypatch.setattr(CompoundScene,'slacks',lambda *args,**kw:dict(passed=True,minimum_slack_m=.001,issues=[]))
    retain_blocking_pairs(scene, full, {'right':2}, {('right_arm/link_2','torso/link_3')})
    checked = scene.slacks(np.zeros(7),.04,np.zeros(3),details=True)
    assert not checked['passed']
    assert checked['issues'][0]['distance_method']=='triangle_surface_guard'
    assert checked['issues'][0]['slack_m']==pytest.approx(-.00002)


def test_contact_route_changes_redundant_joints_in_a_narrow_corridor():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import plan_contact_route
    def pose(q):return np.asarray(q[:3]),np.eye(3)
    def fast(q):return (*pose(q),np.eye(3,7),np.zeros((3,7)))
    def solve(p,R,seed):
        q=np.array(seed);q[:3]=p
        return dict(q=q.tolist(),position_error_m=0.,orientation_error_deg=0.)
    model=SimpleNamespace(lower=-np.ones(7),upper=np.ones(7),pose=pose,fast_pose=fast,solve=solve)
    def slack(q):return q[3]-max(0.,.01-abs(q[0]+.01))
    def slacks(q,*args,**kw):
        g=np.array([slack(q)]);J=np.zeros((1,10));J[0,3]=1.
        if abs(q[0]+.01)<.01:J[0,0]=np.sign(q[0]+.01)
        return (g,J) if kw.get('with_jacobian') else g
    checker=SimpleNamespace(is_clear=lambda q,*args:slack(q)>=-1e-8,slacks=slacks,
                            base=SimpleNamespace(pairs=[('a','b',.005)],floor_minima={}))
    start=np.zeros(7);start[0]=-.1;target=np.zeros(7)
    result=plan_contact_route(model,checker,start,target,.04,np.zeros(3),dict(id='synthetic-corridor'),
                              'left',np.array([0.,1.,0.]),5.,.5)
    assert result['passed']
    raw=np.asarray(result['raw_q'])
    assert raw[:,3].max()>=.009
    for a,b in zip(raw[:-1],raw[1:]):
        assert all(slack(q)>=-1e-8 for q in np.linspace(a,b,101))


def test_retreat_hint_uses_the_nearest_point_on_the_real_rim_segment():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import opening_retreat_hint
    rim=np.array([[[-.1,0.,0.],[.1,0.,0.]],[[-.1,.2,0.],[.1,.2,0.]]])
    np.testing.assert_allclose(opening_retreat_hint(rim,np.array([.02,.03,0.])),[0.,-1.,0.],atol=1e-12)
    assert opening_retreat_hint(rim,np.array([.02,0.,0.])) is None


def test_forward_approach_can_reach_contact_in_a_different_ik_branch():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import plan_contact_route
    def solve(p,R,seed):
        q=np.array(seed);q[:3]=p
        return dict(q=q,position_error_m=0.,orientation_error_deg=0.)
    model=SimpleNamespace(lower=-np.ones(7),upper=np.ones(7),
                          pose=lambda q:(np.asarray(q[:3]),np.eye(3)),solve=solve)
    checker=SimpleNamespace(is_clear=lambda q,*args:q[3]>=0.)
    start=np.zeros(7);start[0]=-.15;start[3]=.02
    target=np.zeros(7);target[3]=-.02
    result=plan_contact_route(model,checker,start,target,.04,np.zeros(3),dict(id='alternate-ik'),
                              'left',np.array([0.,1.,0.]),5.,.5)
    assert result['passed']
    raw=np.asarray(result['raw_q'])
    np.testing.assert_allclose(raw[-1,:3],target[:3])
    assert np.all(raw[:,3]>.0)
    assert all(checker.is_clear(q)for a,b in zip(raw[:-1],raw[1:])for q in np.linspace(a,b,101))


def test_adaptive_rrt_can_turn_inside_a_narrow_joint_corridor():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from plan_fixed_torso_route import rrt
    model=SimpleNamespace(lower=-np.ones(7),upper=np.ones(7))
    def free(q,*args):return abs(q[1])<=.03 and not(.18<q[0]<.22 and q[1]<.02)
    checker=SimpleNamespace(is_clear=free)
    start=np.zeros(7);target=np.zeros(7);target[0]=.4
    result=rrt(model,checker,start,target,.04,np.zeros(3),iterations=1500,
               adaptive_extension=True,shortcut_iterations=0)
    assert result['passed']
    raw=np.asarray(result['raw_q'])
    np.testing.assert_allclose(raw[0],start)
    np.testing.assert_allclose(raw[-1],target)
    assert raw[:,1].max()>=.02
    assert all(free(q)for a,b in zip(raw[:-1],raw[1:])for q in np.linspace(a,b,41))


def test_prior_wrist_turn_from_another_opening_discards_a_colliding_tail(tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_contact_pregrasp import plan_contact_route
    import json
    def solve(p,R,seed):
        q=np.array(seed);q[:3]=p
        return dict(q=q,position_error_m=0.,orientation_error_deg=0.)
    model=SimpleNamespace(lower=-np.ones(7),upper=np.ones(7),
                          pose=lambda q:(np.asarray(q[:3]),np.eye(3)),solve=solve)
    def free(q,*args):return not(.18<q[0]<.22 and q[3]<.01)
    checker=SimpleNamespace(is_clear=free)
    start=np.zeros(7);start[4]=-.2
    target=np.zeros(7);target[0]=.4;target[4]=.2
    waypoint=target.copy();waypoint[0]=.25;waypoint[3]=.02
    blocked=waypoint.copy();blocked[0]=.21;blocked[3]=-.01
    (tmp_path/'routes').mkdir()
    previous=dict(passed=True,condition=dict(id='B-source',stage='B',pitch_deg=0.,roll_deg=0.),
        initial_right_q_rad=start.tolist(),terminal=dict(right=dict(q_rad=blocked.tolist())),
        segments=[dict(side='right',knots_q_rad=[start.tolist(),waypoint.tolist(),blocked.tolist()])])
    (tmp_path/'routes/B-source.json').write_text(json.dumps(previous))
    result=plan_contact_route(model,checker,start,target,.04,np.zeros(3),
        dict(id='E-target',stage='E',pitch_deg=10.,roll_deg=0.),'right',np.array([0.,1.,0.]),5.,.5,route_root=tmp_path)
    assert result['passed']and result['stage']=='prior_approach_contact_extension'
    raw=np.array(result['raw_q'])
    np.testing.assert_allclose(raw[-1],target)
    assert raw[:,3].max()>=.02
    assert all(free(q)for a,b in zip(raw[:-1],raw[1:])for q in np.linspace(a,b,101))
