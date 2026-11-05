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
