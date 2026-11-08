"""The native driver must follow the exported curve, including safe corner stops."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common_opening_replay_support import make_profile,motion_metrics
from sock_dressing_simulation.common_opening_approach import linear_quintic_profile
from validate_compound_trajectory import smooth_profile

@pytest.mark.parametrize('piecewise',[False,True])
def test_replay_reproduces_export_and_stops(piecewise):
    model=SimpleNamespace(velocity=np.ones(7),acceleration=np.ones(7))
    q=np.array([[0.]*7,[.1,.2,0,0,0,0,0],[.2,.1,0,0,0,0,0]])
    p=(linear_quintic_profile if piecewise else smooth_profile)(q,model.velocity,model.acceleration)
    route=dict(raw_q_rad=q.tolist(),q_rad=p['q'].tolist(),times_s=p['t'].tolist(),duration_s=p['duration'])
    if piecewise:route.update(knots_q_rad=p['knots_q_rad'],segment_durations_s=p['segment_durations_s'])
    actual=make_profile(route,model)
    for t,expected in [(0,q[0]),(p['duration'],q[-1])]:
        value,speed=actual['evaluate'](t);np.testing.assert_allclose(value,expected,atol=1e-12);np.testing.assert_allclose(speed,0,atol=1e-12)
    if piecewise:
        value,speed=actual['evaluate'](p['segment_durations_s'][0]);np.testing.assert_allclose(value,q[1]);np.testing.assert_allclose(speed,0,atol=1e-12)
    route['q_rad'][1][0]+=.001
    with pytest.raises(AssertionError):make_profile(route,model)


def test_missing_physics_step_cannot_pass():
    model=SimpleNamespace(lower=-np.ones(7),upper=np.ones(7),velocity=np.ones(7),acceleration=np.ones(7))
    trace=[dict(actual_q_rad=[0.]*7,actual_gap_m=.04) for _ in range(3)]
    native=dict(cloth_time_s=.06,cloth_fixed_steps=2)
    last=(native,np.zeros(3),np.eye(3),np.zeros(7),.04)
    route=dict(target=dict(tcp_target_m=[0.]*3,rotation=np.eye(3).tolist()))
    result=dict(unexpected_contacts=[],actual_configuration_failures=[])
    metrics=motion_metrics(trace,dict(cloth_time_s=0,cloth_fixed_steps=0),last,model,route,.02,0,result,True)
    assert not metrics['passed']
    native['cloth_fixed_steps']=3
    assert motion_metrics(trace,dict(cloth_time_s=0,cloth_fixed_steps=0),last,model,route,.02,0,result,True)['passed']


def test_closure_fast_gate_keeps_skin_and_other_hand_cloth_pairs():
    from common_opening_replay_support import threshold_check
    original=[('right_gripper/finger','frozen_cloth',0),('left_gripper/finger','frozen_cloth',0),('right_gripper/finger','human_skin',.001)]
    class Checker:
        pairs=original
        closure_side='right'
        shoulder_pairs={}
        def is_clear(self,*args):
            assert self.pairs==original[1:]
            return True
        def slacks(self,*args,**kwargs):raise AssertionError('Unexpected full query')
    checker=Checker();result=threshold_check(checker,np.zeros(7),.001,np.zeros(3),True)
    assert result['passed'] and checker.pairs is original


def test_fast_gate_rejection_uses_original_A_tolerance():
    from common_opening_replay_support import threshold_check
    class Checker:
        pairs=[];closure_side='left';shoulder_pairs={}
        def is_clear(self,*args):return False
        def slacks(self,*args,**kwargs):
            assert kwargs['allow_cloth_overlap'] is False
            return dict(passed=True,minimum_slack_m=-.5e-9,issues=[])
    assert threshold_check(Checker(),np.zeros(7),.04,np.zeros(3),False)['passed']
