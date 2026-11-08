import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sock_dressing_simulation.common_opening_approach import (
    reflected_pose, piecewise_knots, approach_targets, edge_samples, linear_quintic_profile,
)


def test_reflection_preserves_approach_and_closing_and_restores_tool_handedness():
    R = Rotation.from_euler('xyz', [23., -41., 12.], degrees=True).as_matrix()
    center = np.array([.2, .3, -.4])
    normal = np.array([1., .3, -.2]); normal /= np.linalg.norm(normal)
    p = np.array([-.1, .5, .6])
    mirrored, attitude = reflected_pose(p, R, center, normal)
    S = np.eye(3)-2*np.outer(normal, normal)
    np.testing.assert_allclose(attitude.T@attitude, np.eye(3), atol=1e-14)
    assert np.linalg.det(attitude) == pytest.approx(1.)
    np.testing.assert_allclose(attitude[:, :2], S@R[:, :2], atol=1e-14)
    assert (mirrored-center)@normal == pytest.approx(-(p-center)@normal)
    twice, twice_R = reflected_pose(mirrored, attitude, center, normal)
    np.testing.assert_allclose(twice, p, atol=1e-14)
    np.testing.assert_allclose(twice_R, R, atol=1e-14)


def test_penultimate_waypoint_is_a_direction_change_not_the_last_dense_sample():
    q = np.array([[0., 0.], [.1, 0.], [.2, 0.], [.2, .1], [.2, .2], [.3, .2], [.4, .2]])
    assert piecewise_knots(q) == [0, 2, 4, 6]
    with pytest.raises(ValueError, match='repeated'):
        piecewise_knots(np.vstack([q[:1], q]))


@pytest.mark.parametrize('side,sign', [('left', 1.), ('right', -1.)])
def test_opening_target_compensates_tip_offset_for_every_roll_and_tilt(monkeypatch, side, sign):
    pytest.importorskip('fcl')
    from sock_dressing_simulation import grasp_geometry
    sample = dict(position=[.1, .2, .3], normal=[sign, 0., 0.], along=[0., 1., 0.], depth_m=0.)
    monkeypatch.setattr(grasp_geometry, 'candidate_samples', lambda *a, **kw: [sample])
    delta = np.array([-.03, -.12, -.3])
    targets = list(approach_targets({}, side, np.eye(3), delta))
    assert targets
    for target in targets:
        R = np.array(target['rotation'])
        tcp = np.array(target['tcp_target_m'])
        tip = np.array(target['tip_midplane_target_m'])
        np.testing.assert_allclose(tcp+.010*R[:, 0], tip, atol=1e-14)
        np.testing.assert_allclose(tip-np.array(sample['position'])-delta,
                                   np.array(sample['normal'])*.002, atol=1e-14)
        assert np.linalg.det(R) == pytest.approx(1.)


def test_joint_edge_sampling_includes_endpoints_and_bounds_every_step():
    a = np.deg2rad([0., 20., 35.])
    b = np.deg2rad([3.2, 19., 34.])
    q = edge_samples(a, b, .25)
    np.testing.assert_array_equal(q[0], a)
    np.testing.assert_array_equal(q[-1], b)
    assert np.max(abs(np.diff(q, axis=0))) <= np.deg2rad(.25)+1e-14


def test_independent_trajectory_validation_rejects_collision_between_valid_endpoints():
    pytest.importorskip('fcl')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_common_opening_starts import validate_route
    model = SimpleNamespace(velocity=np.ones(7), acceleration=np.ones(7), lower=-np.ones(7),
        upper=np.ones(7), arm_names=[str(i) for i in range(7)],
        pose=lambda q: (np.array([q[0], 0., 0.]), np.eye(3)))
    checker = SimpleNamespace(is_clear=lambda q, *a: not .049 < q[0] < .051,
        slacks=lambda *a, **kw: dict(passed=False, issues=['intermediate obstacle']))
    raw = np.zeros((2, 7)); raw[-1, 0] = .1
    target = dict(tcp_target_m=[.1, 0., 0.], tip_midplane_target_m=[.11, 0., 0.], rotation=np.eye(3).tolist())
    result = validate_route(model, checker, raw, .04, np.zeros(3), target)
    assert not result['validation']['passed']
    assert result['validation']['endpoint_position_error_m'] < 1e-12
    assert result['validation']['failures']


def test_rrt_margin_is_five_degrees_and_timeout_is_not_a_collision_claim(monkeypatch):
    pytest.importorskip('fcl')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    import plan_fixed_torso_route as planner
    k = SimpleNamespace(lower=-np.ones(7), upper=np.ones(7))
    checker = SimpleNamespace(is_clear=lambda *a: True)
    start = np.zeros(7); start[0] = .9  # 5.73 degrees from the hard limit.
    goal = start.copy(); goal[0] = .8
    result = planner.rrt(k, checker, start, goal, .04, np.zeros(3), iterations=5)
    assert result['passed']
    times = iter(range(100))
    monkeypatch.setattr(planner.time, 'monotonic', lambda: next(times))
    result = planner.rrt(k, checker, start, goal, .04, np.zeros(3), timeout_s=.1)
    assert not result['passed'] and result['stage'] == 'rrt_timeout'


def test_fixed_frame_audit_detects_motion_in_the_other_arm():
    pytest.importorskip('fcl')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from report_common_opening_starts import fixed_frame_audit
    def fk(q, gap, only):
        result = {n: np.eye(4) for n in only}
        result['left_arm/link_7'][0, 3] = q[0]  # Deliberate unintended coupling.
        return result
    initial = dict(right_q_rad=[0.]*7, left_q_rad=[0.]*7, right_gap_m=.04, translation_m=[0.]*3)
    result = dict(side='right', other_arm_q_rad=[0.]*7, translation_m=[0.]*3,
                  route=dict(q_rad=[[0.]*7, [.1]+[0.]*6], opening_gap_m=.04))
    audit = fixed_frame_audit(result, initial, SimpleNamespace(fk=fk))
    assert not audit['passed']
    assert audit['maximum_fixed_frame_position_error_m'] == pytest.approx(.1)


def test_spatial_broadphase_matches_registered_robot_collision_results():
    pytest.importorskip('fcl')
    import json
    import yaml
    root = Path(__file__).resolve().parents[1]
    initial_path = root/'artifacts/common-opening-approach/initial.yaml'
    if not initial_path.exists():
        pytest.skip('Shared-start search artifact is not present')
    sys.path.insert(0, str(root/'scripts'))
    from search_fixed_torso_starts import context
    from search_centered_reaching import REGISTERED_MANIFEST
    from sock_dressing_simulation.controlled_arm_reaching import ControlledCollisionScene
    initial = yaml.safe_load(initial_path.read_text())
    state = json.loads((root/'artifacts/bulged-cuff/pose-10/state.json').read_text())
    k, base, *_ = context(10, state, REGISTERED_MANIFEST, left_foot_floor_allowance=.15)
    base.park_left(initial['left_q_rad'], .04)
    checker = ControlledCollisionScene(base, 'right', initial['right_q_rad'], .04)
    rng = np.random.default_rng(501)
    poses = [np.asarray(initial['right_q_rad']), np.zeros(7)]
    poses += [np.clip(poses[0]+rng.normal(0, .5, 7), k.lower+1e-6, k.upper-1e-6) for _ in range(5)]
    answers = []
    for enabled in (False, True):
        base.use_spatial_aabb = enabled
        base.fixed_pair_valid_cache = {}
        answers.append([checker.is_clear(q, .04, initial['translation_m']) for q in poses])
    assert answers[0] == answers[1]
    assert any(answers[0]) and not all(answers[0])


def test_exact_polyline_fallback_avoids_an_obstacle_reached_only_by_smoothing():
    pytest.importorskip('fcl')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
    from search_common_opening_starts import validate_route
    model = SimpleNamespace(velocity=np.ones(7), acceleration=np.ones(7), lower=-np.ones(7),
        upper=np.ones(7), arm_names=[str(i) for i in range(7)],
        pose=lambda q: (np.array([q[0], q[1], 0.]), np.eye(3)))
    def clear(q, *a):
        return np.linalg.norm(q[:2]-[.05, .0375]) > .0025
    checker = SimpleNamespace(is_clear=clear,
        slacks=lambda *a, **kw: dict(passed=False, issues=['smoothing cuts into obstacle']))
    raw = np.zeros((3, 7)); raw[1, :2] = [.1, .05]; raw[2, 0] = .2
    target = dict(tcp_target_m=[.2, 0., 0.], tip_midplane_target_m=[.21, 0., 0.], rotation=np.eye(3).tolist())
    result = validate_route(model, checker, raw, .04, np.zeros(3), target)
    assert result['validation']['passed']
    assert result['validation']['rejected_interpolations']
    assert 'Exact joint polyline' in result['validation']['interpolation']
    assert 'segment_durations_s' in result


def test_piecewise_quintic_is_continuous_and_resting_at_its_turns():
    q = np.zeros((3, 7)); q[1, :2] = [.1, .05]; q[2, 0] = .2
    profile = linear_quintic_profile(q, np.ones(7), np.ones(7))
    assert np.all(np.diff(profile['t']) > 0.)
    for time in np.r_[0., np.cumsum(profile['segment_durations_s'])]:
        ids = np.flatnonzero(np.isclose(profile['t'], time, atol=1e-10))
        assert len(ids) == 1
        np.testing.assert_allclose(profile['velocity'][ids], 0., atol=1e-12)
        np.testing.assert_allclose(profile['acceleration'][ids], 0., atol=1e-12)
    assert np.max(profile['velocity_bound']) <= .1+1e-12
    assert np.max(profile['acceleration_bound']) <= .1+1e-12
