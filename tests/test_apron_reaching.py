import importlib.util
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('fcl')
spec=importlib.util.spec_from_file_location('apron_common',Path(__file__).resolve().parents[1]/'scripts/apron_reaching_common.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_overlay_replaces_both_robot_arms_and_preserves_environment_without_aliases():
 old={key:{'pose':'old'} for key in module.ROBOT_KEYS};old['human_skin']={'vertices':[[0,1,2]]};new={key:{'pose':'apron'} for key in module.ROBOT_KEYS}
 scene=module.overlay_robot(old,new)
 for key in module.ROBOT_KEYS:assert scene[key]==new[key]
 assert scene['human_skin']==old['human_skin']
 scene['links']['pose']='changed';scene['human_skin']['vertices'][0][0]=99
 assert old['human_skin']['vertices'][0][0]==0 and new['links']['pose']=='apron'


def test_shoulder_surface_guard_rejects_optimistic_convex_distance(monkeypatch):
 from sock_dressing_simulation.compound_reaching import CompoundScene
 checker=module.AuditedScene.__new__(module.AuditedScene);checker.shoulder_index=0;checker.pairs=[('right_arm/link_2','torso/link_3',.005)];checker.shoulder_distance=lambda q:.0048
 def parent(self,*args,**kwargs):
  if kwargs.get('details'):return dict(passed=True,minimum_slack_m=.0005,issues=[])
  return (np.array([.0005]),np.zeros((1,10))) if kwargs.get('with_jacobian') else np.array([.0005])
 monkeypatch.setattr(CompoundScene,'slacks',parent)
 report=checker.slacks(np.zeros(7),.04,np.zeros(3),details=True)
 assert not report['passed'];assert report['minimum_slack_m']==pytest.approx(-.0002)
 v,J=checker.slacks(np.zeros(7),.04,np.zeros(3),with_jacobian=True)
 assert v[0]==pytest.approx(-.0002);np.testing.assert_allclose(J,0.)


def test_explicit_apron_start_moves_inward_before_uniform_margin_is_required():
 lower=np.deg2rad(np.full(7,-135.));upper=np.deg2rad(np.full(7,135.));start=np.zeros(7);start[3]=np.deg2rad(131.);q=np.tile(start,(4,1));q[:,3]=np.deg2rad([131,130.5,130,120])
 r=module.start_margin_transition(q,lower,upper,start)
 assert r['passed'] and not r['uniform_5deg_passed']
 assert r['first_full_margin_index']==2
 q[1,3]=np.deg2rad(131.2)
 assert not module.start_margin_transition(q,lower,upper,start)['passed']


def test_start_margin_exception_does_not_allow_return_to_limit_or_other_joint_exception():
 lower=np.deg2rad(np.full(7,-135.));upper=np.deg2rad(np.full(7,135.));start=np.zeros(7);start[3]=np.deg2rad(131.);q=np.tile(start,(4,1));q[:,3]=np.deg2rad([131,130,130.2,120])
 assert not module.start_margin_transition(q,lower,upper,start)['passed']
 q[:,3]=np.deg2rad([131,130.5,130,120]);q[2,4]=np.deg2rad(131.)
 assert not module.start_margin_transition(q,lower,upper,start)['passed']
