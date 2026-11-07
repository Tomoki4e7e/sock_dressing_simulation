import importlib.util
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('fcl')
spec=importlib.util.spec_from_file_location('measure_closure',Path(__file__).resolve().parents[1]/'scripts/measure_grasp_closure.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def measurement(active=True,abort=None):
 rows=[]
 for i in range(4):
  pieces=[dict(link=f'right_gripper/{name}_finger_link',active_events=int(active),active_particle_ids=[j] if active else []) for j,name in enumerate(('left','right'))]
  rows.append(dict(phase='hold',pieces=pieces,gap_m=.005,time_s=.1*i,target_relative_tcp_world=[.001*i,0,0],tcp_world=[0,0,0],
    maximum_band_speed_m_s=.01,band_positions_world=[[.001*i,-.002,0],[.001*i,.002,0]]))
 return dict(samples=rows,abort_reason=abort,band_particle_indices=[0,1])


def test_no_contact_is_not_zero_slip_or_retention():
 r=module.analyze(measurement(False));assert r['target_material_relative_drift_m'] is None
 assert r['tangential_contact_material_drift']['left']['maximum_tangential_drift_m'] is None
 assert not r['provisional_contact_persistence_pass'] and not r['capture_or_retention_success_validated']


def test_tracked_material_drift_is_measured_in_metres():
 r=module.analyze(measurement());assert r['bilateral_contact_frames']==4
 assert r['target_material_relative_drift_m']==pytest.approx(.003)
 assert r['tangential_contact_material_drift']['left']['maximum_tangential_drift_m']==pytest.approx(.003)
 assert not r['capture_or_retention_success_validated']


def test_safety_abort_invalidates_persistence_pass():
 r=module.analyze(measurement(abort='foot_intersection'));assert not r['provisional_contact_persistence_pass']


def test_both_contacts_in_a_bin_are_not_necessarily_simultaneous():
 data=measurement()
 for row in data['samples']:row.update(bilateral_physics_steps=0,observation_physics_steps=5)
 r=module.analyze(data);assert r['bilateral_contact_frames']==0
 assert r['hold_bilateral_fraction']==0 and not r['provisional_contact_persistence_pass']


def test_retention_proxy_rejects_missing_nonfinite_and_stationary_target_during_pull():
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
 from report_grasp_closure import retention_status
 assert retention_status(None,'bilateral_contact_persisted',1,1,.001,.001,.01)=='provisional_retention_proxy'
 assert retention_status(None,'bilateral_contact_persisted',1,1,.005,.001,.01)=='material_drift_over_threshold'
 assert retention_status(None,'bilateral_contact_persisted',1,1,None,.001,.01)=='material_drift_not_evaluable'
 assert retention_status(None,'bilateral_contact_persisted',1,1,.001,.001,float('nan'))=='invalid_numeric_measurement'
 assert retention_status('foot_intersection','bilateral_contact_persisted',1,1,.001,.001,.01)=='safety_abort'


def test_local_pad_proximity_distinguishes_wide_gap_and_true_local_proximity():
 from sock_dressing_simulation.closure_pad_metrics import sections,local_pad_proximity
 import trimesh
 class Hand:
  def vertices(self,gap):
   box=trimesh.creation.box(extents=[.02,.004,.02]).vertices
   return {'right_gripper/left_finger_link':box+[0,-gap/2-.002,0],
           'right_gripper/right_finger_link':box+[0,gap/2+.002,0]}
 v=np.array([[-.01,0,-.01],[.01,0,-.01],[.01,0,.01],[-.01,0,.01]])
 faces=np.array([[0,1,2],[0,2,3]])
 near=local_pad_proximity(v,faces,sections(Hand(),.004));assert near['negative_near'] and near['positive_near']
 far=local_pad_proximity(v,faces,sections(Hand(),.020));assert not far['negative_near'] and not far['positive_near']


def test_pull_follow_ratio_distinguishes_world_static_and_hand_following_material():
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
 from report_grasp_closure import material_motion
 for follow,ratio,drift in [(False,0.,.005),(True,1.,0.)]:
  samples=[dict(phase='hold',time_s=0,pull_m=0,target_material_world=[0,0,0],target_relative_tcp_world=[0,0,0]),
   dict(phase='post_hold',time_s=1,pull_m=.005,target_material_world=[.005 if follow else 0,0,0],target_relative_tcp_world=[0 if follow else -.005,0,0])]
  result=material_motion(samples,np.eye(3),[1,0,0]);assert result['pull_follow_ratio']==pytest.approx(ratio)
  assert result['target_tangential_motion_all_hold_m']==pytest.approx(drift)


def test_postprocessing_threshold_can_be_tuned_without_replaying_physics():
 import sys
 sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
 from report_grasp_closure import DEFAULT_CRITERIA,retention_status
 cfg=dict(DEFAULT_CRITERIA,target_drift_m=.006)
 assert retention_status(None,'bilateral_contact_persisted',1,1,.005,.001,.01,criteria=cfg)=='provisional_retention_proxy'
 assert retention_status(None,'bilateral_contact_persisted',1,1,.005,.001,.01)=='material_drift_over_threshold'
