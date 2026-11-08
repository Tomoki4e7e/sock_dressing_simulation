"""Either-arm FK, shared hand geometry, and side-specific closure exclusions."""
import json
from pathlib import Path
import copy
import sys
import numpy as np
import pytest
pytest.importorskip('fcl')
import fcl
from scipy.spatial.transform import Rotation
from sock_dressing_simulation.compound_reaching import CompoundScene
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,shared_gripper_geometry_audit
from sock_dressing_simulation.grasp_geometry import HANDEDNESS

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def native_model():
    path=ROOT/'artifacts/bulged-cuff/pose-10/state.json'
    if not path.exists():pytest.skip('Native calibration snapshot not present')
    sys.path.insert(0,str(ROOT/'scripts'))
    from search_fixed_torso_starts import context
    return context(10,json.loads(path.read_text()))[:2]


def test_left_view_matches_full_fk_and_restores_the_held_arm(native_model):
    k,checker=native_model;right=np.deg2rad([150,-50,-100,100,140,-20,45]);left=np.deg2rad([120,-40,-70,95,150,-60,65]);old=dict(k.q0);v=ControlledArmKinematics(k,'left',right,.04)
    frames=v.fk(left,.03);assert k.q0==old
    T=frames['left_gripper/gripper_base'];p=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b
    np.testing.assert_allclose(v.pose(left)[0],p,atol=1e-12,rtol=0)
    np.testing.assert_allclose(v.pose(left)[1],k.A@T[:3,:3]@HANDEDNESS,atol=1e-12,rtol=0)
    reference=k.fk(right,.04)
    for name in ('right_arm/link_3','right_gripper/gripper_base','torso/link_3','head/link_3'):
        np.testing.assert_array_equal(frames[name],reference[name])


def test_left_angular_jacobian_accounts_for_unity_handedness(native_model):
    k,_=native_model;v=ControlledArmKinematics(k,'left');q=np.deg2rad([120,-40,-70,95,150,-60,65]);R=v.pose(q)[1];h=1e-6
    columns=[]
    for j in range(7):
        plus=q.copy();minus=q.copy();plus[j]+=h;minus[j]-=h
        columns.append((Rotation.from_matrix(v.pose(plus)[1]@R.T).as_rotvec()-Rotation.from_matrix(v.pose(minus)[1]@R.T).as_rotvec())/(2*h))
    np.testing.assert_allclose(v.fast_pose(q)[3],np.array(columns).T,atol=2e-9,rtol=0)


def test_shared_hand_audit_rejects_changed_joint_geometry(native_model):
    k,_=native_model;assert shared_gripper_geometry_audit(k)['passed']
    altered=copy.copy(k);altered.tree=copy.deepcopy(k.tree);altered.joints={j.get('name'):j for j in altered.tree.findall('joint')}
    altered.joints['left_gripper/finger_joint'].find('axis').set('xyz','1 0 0')
    assert not shared_gripper_geometry_audit(altered)['passed']


def test_left_closure_skips_only_left_cloth_and_keeps_skin_and_right_cloth():
    c=CompoundScene.__new__(CompoundScene);c.update=lambda *args:None;c.floor_minima={};c.groups={};c.group_bounds={}
    bounds=(np.full((1,3),-.5),np.full((1,3),.5))
    for side in ('left','right'):
        name=side+'_gripper/finger';c.groups[name]=[dict(obj=fcl.CollisionObject(fcl.Box(1,1,1)))];c.group_bounds[name]=bounds
    c.obstacle_objects={n:fcl.CollisionObject(fcl.Box(1,1,1)) for n in ('frozen_cloth','human_full_skin')};c.obstacle_bounds={n:(bounds[0][0],bounds[1][0]) for n in c.obstacle_objects}
    c.pairs=[('left_gripper/finger','frozen_cloth',0.),('right_gripper/finger','frozen_cloth',0.),('left_gripper/finger','human_full_skin',.001)]
    result=c.slacks(np.zeros(7),.001,np.zeros(3),allow_cloth_overlap=True,cloth_overlap_side='left',details=True)
    assert [r['pair'] for r in result['issues']]==[['right_gripper/finger','frozen_cloth'],['left_gripper/finger','human_full_skin']]
    default=c.slacks(np.zeros(7),.001,np.zeros(3),allow_cloth_overlap=True,details=True)
    assert [r['pair'] for r in default['issues']]==[['left_gripper/finger','frozen_cloth'],['left_gripper/finger','human_full_skin']]


def test_jaw_profile_closes_smoothly_within_finger_velocity_limit():
    sys.path.insert(0,str(ROOT/'scripts'))
    from replay_controlled_photo_grasp import jaw_profile
    t=np.linspace(0,10,1001);positions,rates=jaw_profile(t,10,.05,.001)
    np.testing.assert_allclose(positions[[0,-1]],[.05,.001],atol=1e-15,rtol=0);np.testing.assert_allclose(rates[[0,-1]],0,atol=1e-15,rtol=0)
    assert np.all(np.diff(positions)<=1e-12) and np.max(abs(rates))<.02


def test_fixed_pair_cache_matches_uncached_checks_when_both_arms_and_scene_change(native_model):
    from sock_dressing_simulation.controlled_arm_reaching import ControlledCollisionScene
    k, checker = native_model
    original = dict(k.q0)
    view = ControlledCollisionScene(checker, 'left', k.start)
    left = np.array([k.q0[f'left_arm/joint_{i}'] for i in range(1, 8)])
    try:
        for offset in (0., .015):
            q = left.copy(); q[0] += offset
            delta = np.array([offset, .12, -.3])
            cached = view.slacks(q, .04, delta)
            checker.fixed_pair_cache_key = None
            checker.park_left(q, .04)
            direct = checker.slacks(view.fixed_right_q, view.fixed_right_gap, delta)
            np.testing.assert_allclose(cached, direct, atol=1e-12, rtol=0)
            view.fixed_right_q = view.fixed_right_q.copy()
            view.fixed_right_q[0] += .01
    finally:
        k.q0.clear(); k.q0.update(original)
        checker.fixed_pair_cache_key = None


def test_right_arm_fixed_pair_cache_tracks_held_joints_width_and_translation(native_model):
    from sock_dressing_simulation.controlled_arm_reaching import ControlledCollisionScene
    from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM
    k,checker=native_model;original=dict(k.q0)
    view=ControlledCollisionScene(checker,'right',k.start)
    left=np.array([k.q0[n]for n in LEFT_ARM])
    try:
        for left_offset,left_gap,translation in ((0.,.04,0.),(.015,.04,0.),(.015,.03,0.),(.015,.03,.01)):
            held=left.copy();held[0]+=left_offset;checker.park_left(held,left_gap)
            delta=np.array([translation,.12,-.3])
            for offset,gap in ((0.,.04),(.01,.03)):
                moving=k.start.copy();moving[0]+=offset
                cached=view.slacks(moving,gap,delta)
                checker.fixed_pair_cache_key=None
                direct=checker.slacks(moving,gap,delta)
                np.testing.assert_allclose(cached,direct,atol=1e-12,rtol=0)
                valid=view.is_clear(moving,gap,delta)
                checker.fixed_pair_cache_key=None
                assert valid==checker.is_clear(moving,gap,delta)
    finally:
        k.q0.clear();k.q0.update(original);checker.fixed_pair_cache_key=None
