"""Frozen visual frame and complete parked-left collision policy checks."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('fcl')
from sock_dressing_simulation.fixed_torso_reaching import FixedTorsoScene


def test_fixed_joint_guard_allows_arm_changes_but_rejects_torso_change():
    checker=FixedTorsoScene.__new__(FixedTorsoScene)
    checker.reference_fixed_joints={'torso/joint_1':-.785,'head/joint_1':0.,'base_link_rot_yaw':0.}
    checker.k=SimpleNamespace(q0=dict(checker.reference_fixed_joints,**{'left_arm/joint_1':1.,'right_arm/joint_1':2.}))
    checker.assert_fixed()
    checker.k.q0['left_arm/joint_1']=2.5;checker.assert_fixed()
    checker.k.q0['torso/joint_1']+=.001
    with pytest.raises(ValueError,match='torso, head or base'):checker.assert_fixed()


def test_left_policy_retains_opposing_fingers_and_cross_arm_collision():
    checker=FixedTorsoScene.__new__(FixedTorsoScene)
    checker.group={n:n for n in ['left_gripper/gripper_base','right_gripper/gripper_base']};checker.neighbours=set();checker.model_exclusions=set()
    assert checker.excluded('left_arm/link_5','left_arm/link_7')
    assert checker.excluded('left_gripper/gripper_base','left_gripper/left_finger_link')
    assert not checker.excluded('left_gripper/left_finger_link','left_gripper/right_finger_link')
    assert not checker.excluded('left_gripper/gripper_base','right_gripper/gripper_base')


def test_explicit_grounded_foot_exception_keeps_sock_support_and_raised_leg_floors():
    checker=FixedTorsoScene.__new__(FixedTorsoScene)
    checker.floor_minima={'human_full_skin':.023,'frozen_cloth':.62,'right_leg_actual_skin_rigid':.54,'support':.43}
    before=dict(checker.floor_minima)
    checker.allow_grounded_left_foot_penetration(.05)
    assert checker.floor_minima['human_full_skin']==pytest.approx(.073)
    for name in ('frozen_cloth','right_leg_actual_skin_rigid','support'):
        assert checker.floor_minima[name]==before[name]
    with pytest.raises(ValueError,match='already been configured'):
        checker.allow_grounded_left_foot_penetration(.05)
    other=FixedTorsoScene.__new__(FixedTorsoScene)
    with pytest.raises(ValueError,match='0–250 mm'):
        other.allow_grounded_left_foot_penetration(.3)


def test_native_calibrated_left_parking_preserves_torso_head_and_right_tcp():
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/fixed-torso-start-search/left-parks.json').exists():pytest.skip('Native scene artifacts not present')
    import sys
    sys.path.insert(0,str(root/'scripts'))
    from search_fixed_torso_starts import context
    state=json.loads((root/'artifacts/bulged-cuff/pose-10/state.json').read_text())
    k,checker,dl,du,contract,scene=context(10,state)
    before=k.fk(k.start);tcp=k.pose(k.start)[0].copy();span=contract['gripper_x_interval_m'].copy()
    park=json.loads((root/'artifacts/fixed-torso-start-search/left-parks.json').read_text())['candidates'][0]
    checker.park_left(park['q_rad']);after=k.fk(k.start)
    for name in ['base_link','torso/link_3','head/link_3']:
        np.testing.assert_array_equal(after[name],before[name])
    np.testing.assert_array_equal(k.pose(k.start)[0],tcp)
    assert contract['gripper_x_interval_m']==span
    assert not np.allclose(after['left_gripper/grasping_frame'],before['left_gripper/grasping_frame'])
    assert checker.is_clear(k.start,.04,[0,0,-3])
    # The previously untested parked-left self pairs are now present.
    assert any(set((a,b))=={'left_gripper/gripper_base','left_arm/link_5'} for a,b,m in checker.pairs)
    assert any(set((a,b))=={'left_arm/link_2','torso/link_3'} for a,b,m in checker.pairs)


def test_left_chain_position_and_jacobian_match_full_tree_fk():
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/bulged-cuff/pose-10/state.json').exists():pytest.skip('Native scene artifacts not present')
    import sys
    sys.path.insert(0,str(root/'scripts'))
    from search_fixed_torso_starts import context
    from search_photo_reference_reaching import LeftChain
    state=json.loads((root/'artifacts/bulged-cuff/pose-10/state.json').read_text());k,checker,*_=context(10,state)
    chain=LeftChain(k);q=np.deg2rad([150,-40,-80,100,120,-60,60]);checker.park_left(q)
    T=k.fk(k.start,only=['left_gripper/gripper_base'])['left_gripper/gripper_base']
    expected=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b
    np.testing.assert_allclose(chain.pose(q)[0],expected,atol=1e-12,rtol=0)
    columns=[];h=1e-6
    for j in range(7):
        plus=q.copy();minus=q.copy();plus[j]+=h;minus[j]-=h
        checker.park_left(plus);Tp=k.fk(k.start,only=['left_gripper/gripper_base'])['left_gripper/gripper_base']
        checker.park_left(minus);Tm=k.fk(k.start,only=['left_gripper/gripper_base'])['left_gripper/gripper_base']
        pp=k.A@(Tp[:3,3]+Tp[:3,:3]@k.hand.grasp_center_urdf)+k.b
        pm=k.A@(Tm[:3,3]+Tm[:3,:3]@k.hand.grasp_center_urdf)+k.b
        columns.append((pp-pm)/(2*h))
    np.testing.assert_allclose(chain.pose(q)[2],np.array(columns).T,atol=2e-9,rtol=0)


def test_saved_photo_start_has_lower_wrists_inward_hands_and_foot_between_grippers():
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/bulged-cuff/pose-10/state.json').exists():pytest.skip('Native scene artifacts not present')
    import sys
    import yaml
    sys.path.insert(0,str(root/'scripts'))
    from search_fixed_torso_starts import context
    from sock_dressing_simulation.grasp_geometry import HANDEDNESS
    config=yaml.safe_load((root/'config/photo_reference_initial_pose.yaml').read_text())
    state=json.loads((root/config['reference_state']).read_text());k,c,*_=context(config['number'],state,left_foot_floor_allowance=config['left_foot_floor_allowance_m']);c.park_left(config['left_q_rad'],config['left_gap_m'])
    frames=k.fk(config['right_q_rad'],config['right_gap_m']);span=[]
    samples=json.loads((root/f"artifacts/bulged-cuff/pose-{config['number']:02d}/samples.json").read_text())
    minimum_depth=min(s['depth_m'] for s in samples)
    cuff=np.mean([s['position'] for s in samples if s['depth_m']==minimum_depth],axis=0)+config['translation_m']
    for side in ('right','left'):
        elbow=frames[side+'_arm/link_3']
        for i in (5,7):
            wrist=frames[side+'_arm/link_'+str(i)]
            assert (k.A@(elbow[:3,3]-wrist[:3,3]))[1]>=.02-1e-6
        T=frames[side+'_gripper/gripper_base'];tcp=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b;direction=cuff-tcp;direction/=np.linalg.norm(direction)
        assert (k.A@T[:3,:3]@HANDEDNESS)[:,0]@direction>=np.cos(np.deg2rad(40.001))
        span.append((k.A@frames[side+'_gripper/grasping_frame'][:3,3]+k.b)[0])
    foot=np.array(state['source']['surfaces'][0]['vertices'])+config['translation_m']
    assert foot[:,0].min()>=min(span) and foot[:,0].max()<=max(span)
