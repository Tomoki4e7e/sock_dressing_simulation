import json
from pathlib import Path
import runpy
import numpy as np
import pytest
from types import SimpleNamespace
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.sock_cloth import validate_prepared_state_readback


def test_equal_particles_do_not_hide_different_preparation_velocities():
    snapshot=dict(positions=[dict(x=0,y=0,z=0,w=0)],previousPositions=[],velocities=[dict(x=0,y=0,z=0,w=0)],
        angularVelocities=[],orientations=[],previousOrientations=[],restKeys=[1],restValues=[.1],
        rimKeys=[0],rimValues=[dict(x=0,y=0,z=0)],barrierDepthKeys=[],barrierDepthValues=[],
        barrierInside=[],barrierContacts=[],predictiveContacts=[])
    captured=json.dumps(snapshot)
    assert validate_prepared_state_readback(captured,captured)['ok']
    snapshot['velocities'][0]['y']=.1
    validation=validate_prepared_state_readback(captured,json.dumps(snapshot))
    assert not validation['ok'] and not validation['checks']['velocities']


def test_material_request_matches_values_actually_sent_to_native_player():
    config=load_config(Path('config/opening_widen_w95.yaml'))
    assert config['obi']['requested']==config['obi']['expected']
    assert config['obi']['requested']['stretch_compliance']==1e-5


@pytest.fixture
def conforming(tmp_path):
    module=runpy.run_path('scripts/tune_foot_conforming.py');assess=module['assess']
    commands=np.loadtxt(module['BASELINE']/'applied_action.csv',delimiter=',')
    episode=tmp_path/'condition/data_sock_sim_smoke/train/episode'
    episode.mkdir(parents=True)
    np.savetxt(episode/'applied_action.csv',commands,delimiter=',')
    (tmp_path/'condition/config.yaml').write_text('scene:\n  grasp_anchors:\n    max_distance_m: 0.04\nobi:\n  expected:\n    slip_constraint_error_m: 0.2\n')
    (tmp_path/'condition/applied_material.json').write_text(json.dumps(dict(
        validation=dict(ok=True),prepared_state_readback_validation=dict(ok=True),actual=dict(particle_radius_m=.008,effective_constraint_iterations={
            key:20 for key in ['distance','bending','collision','particle_collision','pin']}))))
    safety=dict(contact_safety_armed=True,remaining_swept_intersections=0,remaining_surface_intersections=0,
                continuous_surface_motion_checked=True,remaining_swept_surface_intersections=0,
                native_particle_ccd=1,native_surface_collision_tolerance_m=.0001,
                particle_contact_shell_policy='particle_radius_with_numerical_separation',
                particle_contact_shell_m=.00801,particle_contact_numerical_separation_m=.00001,
                surface_reference_policy="per_triangle_accepted_segments",surface_bounds_policy="collider_geometry",
                maximum_render_position_error_m=0,maximum_end_position_error_m=0,
                foot_skin_vertex_count=326,uncovered_foot_skin_vertices=0,
                cloth_surface_triangle_count=1500,
                surface_collisions_enabled=True,native_collision_triangles=1500,native_collision_edges=32,
                maximum_original_grasp_anchor_error_m=0,maximum_grasp_offset_drift_m=0)
    safety['native_grasp_binding']=dict(all_original_targets_match=True,expected_pin_count=10,matched_pin_count=10)
    rows=[dict(foot_collision_safety={**safety,'cloth_physics_clock':dict(externally_stepped=True,fixed_steps=1000+i*18,simulated_time_s=(1000+i*18)*.02,last_step_time_s=.02)},geometric_maximum_cloth_foot_penetration_m=0,
               surface_containment_ratio=.3,sections=[dict(name='toes',containment_ratio=.75),
               dict(name='forefoot',containment_ratio=.5)]) for i in range(100)]
    material=[dict(stretch=dict(available=True,edge_classes=dict(body_body=dict(maximum_stretch=1.5)))) for _ in range(100)]
    metadata=dict(frames=100,stop_reason='max_steps',dressing_quality_by_frame=rows,
                  rollout_initial_cloth_physics_clock=dict(externally_stepped=True,fixed_steps=982,simulated_time_s=982*.02,last_step_time_s=.02),
                  cloth_quality_by_frame=material,task_success=dict(continuous_grasp_ok=True,success=False))
    path=episode/'metadata.json'
    def check():path.write_text(json.dumps(metadata));return assess(path,100)
    return metadata,check


def test_conforming_and_original_dressing_success_are_separate(conforming):
    _,check=conforming;row=check()
    assert row['conforming_passed']
    assert not row['original_task_success']


@pytest.mark.parametrize('binding', [None,
    dict(all_original_targets_match=False,expected_pin_count=10,matched_pin_count=9),
    dict(all_original_targets_match=True,expected_pin_count=0,matched_pin_count=0),
    dict(all_original_targets_match=True,expected_pin_count=10,matched_pin_count=9)])
def test_stale_or_unmeasured_native_gripper_references_cannot_qualify(conforming,binding):
    metadata,check=conforming
    contact=metadata['dressing_quality_by_frame'][4]['foot_collision_safety']
    if binding is None:contact.pop('native_grasp_binding')
    else:contact['native_grasp_binding']=binding
    result=check()
    assert not result['safe']
    assert not result['checks']['native_pins_reference_original_grippers']


def add_anatomical_metrics(metadata, surface=.3, toes=.75, forefoot=.5):
    for row in metadata['dressing_quality_by_frame']:
        row['foot_collision_safety'].update(foot_collider_shape_mode='skin-hull',skin_face_coverage_verified=True,native_foot_mesh_registration_verified=True,
            substep_velocity_projection_enabled=True,velocity_sync_preserves_native_damping_and_friction=True,
            native_substep_velocity_synchronizations=144,
            maximum_velocity_position_error_m_s=0)
        row['anatomical_foot_conformity']=dict(valid=True,surface_containment_ratio=surface,sections=[
            dict(name='toes',containment_ratio=toes,total_samples=8,sample_source='skin_envelope'),
            dict(name='forefoot',containment_ratio=forefoot,total_samples=8,sample_source='skin_envelope')])


def test_actual_foot_coverage_keeps_legacy_metrics_and_original_success(conforming):
    metadata,check=conforming
    for row in metadata['dressing_quality_by_frame']:row['surface_containment_ratio']=0
    add_anatomical_metrics(metadata)
    result=check()
    assert result['conforming_passed'] and result['mean_tail_surface_containment']==pytest.approx(.3)
    assert result['legacy_mean_tail_surface_containment']==0 and not result['original_task_success']
    assert result['containment_metric']=='anatomical_skin_envelope'


def test_good_box_samples_do_not_hide_bad_actual_foot_coverage(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata,toes=.5)
    assert not check()['conforming_passed']


def test_missing_anatomical_frame_fails_closed_instead_of_using_boxes(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata)
    del metadata['dressing_quality_by_frame'][4]['anatomical_foot_conformity']
    assert not check()['safe']


def test_proxy_samples_cannot_establish_actual_forefoot_expansion(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata)
    metadata['dressing_quality_by_frame'][4]['anatomical_foot_conformity']['sections'][1]['sample_source']='box_fallback'
    assert not check()['safe']


def test_vertex_coverage_does_not_hide_gaps_between_collider_faces(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata)
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['skin_face_coverage_verified']=False
    assert not check()['safe']


def test_geometric_skin_coverage_cannot_hide_missing_native_mesh_registration(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata)
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['native_foot_mesh_registration_verified']=False
    result=check()
    assert not result['safe'] and not result['checks']['native_foot_mesh_registration_verified']


def test_new_mesh_inherits_robot_collision_policy_before_first_physics_step():
    calls=[]
    class Cloth:
        def configure_foot_collider_shape(self,mode):calls.append(('shape',mode))
        def configure_foot_conforming(self,**kwargs):calls.append(('conforming',kwargs))
        def arm_foot_collision_safety(self,armed):calls.append(('arm',armed))
        def ignore_non_gripper_robot_human_rigid_collisions(self,robot):calls.append(('policy',robot))
    env=SockDressingEnv.__new__(SockDressingEnv)
    env.sock_cloth=Cloth()
    env.config=dict(foot_conforming_rollout=dict(collider_shape='skin-hull'),
                   scene=dict(ignore_robot_human_rigid_collisions=False,ignore_non_gripper_robot_human_rigid_collisions=True),
                   assets=dict(robot_id=123))
    env._env=SimpleNamespace(step=lambda **kwargs:calls.append(('flush',kwargs)))
    env.begin_cloth_contact_rollout()
    assert [call[0] for call in calls]==['shape','conforming','arm','policy','flush']
    assert calls[-1][1]=={'simulate':False}
    assert env._cloth_contact_rollout_started


@pytest.mark.parametrize('key',['maximum_original_grasp_anchor_error_m','maximum_grasp_offset_drift_m'])
def test_attached_grippers_do_not_hide_moving_grasp_offsets(conforming,key):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][23]['foot_collision_safety'][key]=.201 if key=='maximum_original_grasp_anchor_error_m' else .003
    row=check();assert not row['safe'] and not row['conforming_passed']


def test_particle_selection_radius_is_not_a_mechanical_slip_limit(conforming):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][23]['foot_collision_safety']['maximum_original_grasp_anchor_error_m']=.05
    row=check()
    assert row['safe']
    assert row['configured_grasp_slip_threshold_m']==.2
    assert row['maximum_anchor_error_m']==.05


def test_single_final_coverage_frame_does_not_establish_expansion(conforming):
    metadata,check=conforming
    for row in metadata['dressing_quality_by_frame'][-10:-1]:row['surface_containment_ratio']=0
    assert not check()['conforming_passed']


def test_coverage_does_not_hide_excessive_stretch(conforming):
    metadata,check=conforming
    metadata['cloth_quality_by_frame'][-1]['stretch']['edge_classes']['body_body']['maximum_stretch']=1.52
    row=check();assert row['safe'] and not row['conforming_passed']


def test_missing_anchor_diagnostics_fail_closed(conforming):
    metadata,check=conforming
    del metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['maximum_original_grasp_anchor_error_m']
    assert not check()['safe']


def test_extra_query_physics_step_cannot_pass_fixed_motion(conforming):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][23]['foot_collision_safety']['cloth_physics_clock']['fixed_steps']+=1
    assert not check()['checks']['fixed_cloth_physics_time']


@pytest.mark.parametrize('field',['maximum_render_position_error_m','maximum_end_position_error_m'])
def test_corrected_particles_cannot_hide_stale_render_geometry(conforming,field):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][3]['foot_collision_safety'][field]=.004
    assert not check()['checks']['accepted_geometry_matches_render']


def test_render_driven_cloth_clock_cannot_pass_fixed_motion(conforming):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][23]['foot_collision_safety']['cloth_physics_clock']['externally_stepped']=False
    assert not check()['safe']


def test_missing_material_frames_cannot_pass_conformance(conforming):
    metadata,check=conforming
    metadata['cloth_quality_by_frame']=[]
    assert not check()['conforming_passed']


def test_zero_cloth_triangles_cannot_establish_no_face_crossings(conforming):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][0]['foot_collision_safety']['cloth_surface_triangle_count']=0
    assert not check()['safe']


def test_static_face_checks_do_not_replace_native_surface_impulses(conforming):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][2]['foot_collision_safety']['native_collision_triangles']=0
    assert not check()['checks']['native_surface_contact_applied']


def test_diagnostic_continuation_cannot_be_selected_as_best(conforming):
    metadata,check=conforming
    metadata['pre_inference_drape']={'passed':False,'continued_for_diagnostics':True}
    row=check()
    assert not row['safe'] and not row['conforming_passed']


def test_distance_iterations_do_not_hide_single_iteration_grasp(conforming,tmp_path):
    _,check=conforming
    path=tmp_path/'condition/applied_material.json'
    values=json.loads(path.read_text())
    values['actual']['effective_constraint_iterations']['pin']=1
    path.write_text(json.dumps(values))
    assert not check()['checks']['coupled_solver_iterations_applied']


def test_rollout_readback_matches_last_camera_without_changing_physics_time():
    class Backend:
        def __init__(self):
            self.frame=0;self.pending=False;self.steps=[]
        def step(self,simulate=True):
            self.steps.append(simulate)
            self.frame+=int(simulate)
            if self.pending:
                environment.sock_cloth.data={'sample_frame':self.frame}
                self.pending=False
    for rollout in (False,True):
        backend=Backend()
        environment=SockDressingEnv({'scene':{}},backend=backend)
        environment.sock_cloth=SimpleNamespace(id=10,data={})
        environment.robot=SimpleNamespace(data={})
        environment.human=SimpleNamespace(data={})
        environment.camera=environment.recording_camera=environment.side_recording_camera=object()
        environment._cloth_contact_rollout_started=rollout
        environment._request_native_cloth_observation=lambda: setattr(backend,'pending',True)
        environment.diagnostics=lambda: {}
        environment.robot_signals=lambda data: {}
        def capture(simulate=True):
            backend.step(simulate=simulate)
            return {'sample_frame':backend.frame}
        environment._capture_camera=environment._capture_recording_camera=environment._capture_side_recording_camera=capture
        observation=environment._observe_native_custom_player()
        assert backend.frame==4
        assert observation['side_recording_camera']['sample_frame']==4
        assert observation['cloth']['sample_frame']==(4 if rollout else 1)
        assert backend.steps.count(False)==4*int(rollout)
        if rollout:
            assert observation['camera']['sample_frame']==4
            assert observation['recording_camera']['sample_frame']==4


def test_native_registration_does_not_hide_unsynchronized_substep_velocity(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata)
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['native_substep_velocity_synchronizations']=143
    result=check()
    assert not result['safe'] and not result['checks']['accepted_substep_velocity_synchronized']


def test_nonfinite_velocity_sync_diagnostic_fails_closed(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata)
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['maximum_velocity_position_error_m_s']=float('nan')
    assert not check()['safe']


def test_rigid_stop_tolerates_only_physx_boundary_roundoff_and_preserves_raw_value():
    from sock_dressing_simulation.quality import rigid_collision_stop_required
    measured=.005000021308660507
    assert not rigid_collision_stop_required(measured,.005)
    assert measured>.005  # Original task-success comparison stays unchanged.
    assert rigid_collision_stop_required(.005001,.005)
    assert rigid_collision_stop_required(.006,.005)
    assert rigid_collision_stop_required(float('nan'),.005)


def test_final_solver_conditions_use_candidate_values_after_common_preparation(conforming):
    import yaml
    metadata,check=conforming;add_anatomical_metrics(metadata)
    path=Path(check()['metadata']).parents[3]
    config=yaml.safe_load((path/'config.yaml').read_text())
    config['foot_conforming_rollout']=dict(parameters=dict(substeps=16,solver_iterations=40))
    (path/'config.yaml').write_text(yaml.safe_dump(config))
    material=json.loads((path/'applied_material.json').read_text())
    material['actual']['effective_constraint_iterations']={key:40 for key in material['actual']['effective_constraint_iterations']}
    (path/'applied_material.json').write_text(json.dumps(material))
    for row in metadata['dressing_quality_by_frame']:
        row['foot_collision_safety']['native_substep_velocity_synchronizations']=288
    result=check()
    assert result['safe'] and result['checks']['coupled_solver_iterations_applied']
    assert result['checks']['accepted_substep_velocity_synchronized']


def test_projection_cannot_bypass_native_friction_or_damping(conforming):
    metadata,check=conforming;add_anatomical_metrics(metadata)
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['velocity_sync_preserves_native_damping_and_friction']=False
    assert not check()['safe']


def test_clear_vertices_and_endpoint_faces_cannot_hide_crossing_face_motion(conforming):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['remaining_swept_surface_intersections']=1
    result=check()
    assert not result['safe'] and not result['checks']['no_remaining_surface_motion']


def test_native_surface_contact_without_prediction_is_not_accepted(conforming):
    metadata,check=conforming
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety']['native_particle_ccd']=0
    assert not check()['safe']


@pytest.mark.parametrize('key,value,check', [
    ('surface_reference_policy', 'whole_solve_frozen', 'local_surface_history_applied'),
    ('surface_bounds_policy', 'section_template', 'surface_bounds_match_collider_geometry'),
])
def test_safe_endpoints_cannot_hide_wrong_surface_history_or_bounds(conforming, key, value, check):
    metadata, assess_record = conforming
    metadata['dressing_quality_by_frame'][4]['foot_collision_safety'][key] = value
    result = assess_record()
    assert not result['checks'][check]
    assert not result['conforming_passed']


@pytest.mark.parametrize('frames,stop,port,expected', [
    (40,'max_steps',6480,6480), (8,'max_steps',6480,None),
    (40,'rigid_collision',6480,None), (40,'max_steps',True,None),
    (40,'max_steps',-1,None), (40,'max_steps',65535,None),
])
def test_recorded_port_requires_a_complete_matching_record(tmp_path,frames,stop,port,expected):
    from scripts.tune_foot_conforming import recorded_transport_port
    import yaml
    metadata=tmp_path/'metadata.json'
    metadata.write_text(json.dumps(dict(frames=frames,stop_reason=stop)))
    (tmp_path/'result.json').write_text(json.dumps(dict(metadata=str(metadata))))
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(dict(rcareworld=dict(port=port))))
    assert recorded_transport_port(tmp_path,40)==expected


def test_cache_port_grouping_preserves_ports_and_serializes_shared_ports():
    from scripts.tune_foot_conforming import transport_groups
    pending=[('cached_a',6440),('new_a',None),('cached_b',6440),('cached_c',6444),('new_b',None)]
    groups=list(transport_groups(pending,6440,4))
    assert groups[0]==[('cached_a',6440),('new_a',6448),('cached_c',6444),('new_b',6452)]
    assert groups[1]==[('cached_b',6440)]
    assert pending[1][1] is None


def test_eight_independent_recorders_keep_historical_ports_exclusive():
    from scripts.tune_foot_conforming import transport_groups
    pending=[('cached_a',6620),('cached_b',6624),('shared',6620)]
    pending += [('new_'+str(i),None) for i in range(10)]
    groups=list(transport_groups(pending,6620,8))
    assert [len(group) for group in groups]==[8,5]
    assert groups[0][:2]==[('cached_a',6620),('cached_b',6624)]
    assert groups[1][0]==('shared',6620)
    assert {condition for group in groups for condition,_ in group}=={condition for condition,_ in pending}
    for index,group in enumerate(groups):
        assert len({port for _,port in group})==len(group)
        assert all(port%4==0 for _,port in group)
        # A completed group's port can be reused after the batch barrier.
        reserved=(6620,6624) if index==0 else (6620,)
        assert all(port not in reserved for condition,port in group if condition.startswith('new_'))


def test_condition_identity_retains_existing_baseline_record_name():
    from scripts.tune_foot_conforming import condition_identity,condition_name
    base=dict(friction=.5,damping=.8,stretch_compliance=1e-5,bend_compliance=.1,
              particle_radius_m=.008,collision_margin_m=.002)
    assert condition_name(condition_identity(base),'local_transport')=='local_transport_26ba670fdd43'


def test_transport_choice_uses_safe_matched_controls_and_keeps_local_on_a_tie():
    from scripts.tune_foot_conforming import choose_grasp_transport
    local=dict(safe=True,mode='local',fixed=True,collider_shape='skin-hull',transport='local',
               mean_tail_surface_containment=.4,tail_stretch_violation_frames=0,maximum_contact_correction_m=.002)
    bulk={**local,'transport':'bulk'}
    assert choose_grasp_transport([bulk,local])[0]=='local'
    assert choose_grasp_transport([local,{**bulk,'mean_tail_surface_containment':.5}])[0]=='bulk'
    assert choose_grasp_transport([local,{**bulk,'safe':False,'mean_tail_surface_containment':1}])[0]=='local'
    assert choose_grasp_transport([local,{**bulk,'fixed':False,'mean_tail_surface_containment':1}])[0]=='local'
    assert choose_grasp_transport([local,{**bulk,'collider_shape':'boxes','mean_tail_surface_containment':1}])[0]=='local'
    assert choose_grasp_transport([{**bulk,'safe':False}])==('local',None)


def test_active_port_waits_for_lock_owner_without_accepting_partial_results(tmp_path):
    import fcntl
    import yaml
    from scripts.tune_foot_conforming import active_transport_port,recorded_transport_port
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(dict(rcareworld=dict(port=6480))))
    with (tmp_path/'replay.lock').open('a') as recorder:
        fcntl.flock(recorder.fileno(),fcntl.LOCK_EX)
        assert active_transport_port(tmp_path)==6480
        assert recorded_transport_port(tmp_path,40) is None
        fcntl.flock(recorder.fileno(),fcntl.LOCK_UN)
    assert active_transport_port(tmp_path) is None


@pytest.mark.parametrize('field,value', [
    ('particle_contact_shell_m', .01001),
    ('particle_contact_shell_policy', 'legacy_radius_plus_margin'),
    ('particle_contact_numerical_separation_m', .002),
    ('particle_contact_numerical_separation_m', float('nan')),
])
def test_predictive_margin_cannot_masquerade_as_physical_particle_radius(conforming, field, value):
    metadata, check = conforming
    metadata['dressing_quality_by_frame'][17]['foot_collision_safety'][field] = value
    result = check()
    assert not result['checks']['particle_contact_shell_matches_native_radius']
    assert not result['safe'] and not result['conforming_passed']


def test_missing_physical_contact_shell_diagnostics_fail_closed(conforming):
    metadata, check = conforming
    del metadata['dressing_quality_by_frame'][17]['foot_collision_safety']['particle_contact_shell_policy']
    assert not check()['checks']['particle_contact_shell_matches_native_radius']
