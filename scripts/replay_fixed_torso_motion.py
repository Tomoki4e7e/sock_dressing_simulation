#!/usr/bin/env python3
"""Drive a validated fixed-torso route after its native initial-state check."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,REGISTERED_MANIFEST
from validate_compound_trajectory import smooth_profile
from replay_registered_apron_routes import native_tcp
from sock_dressing_simulation.arm_reaching import ARM
from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM


def replay(env,mapping,action,reference,k,checker,saved,source_path,delta,expected_cloth,out,dt=.02):
    gate=json.loads(source_path.with_name(source_path.name.replace('trajectory','validation')).read_text())
    if not gate['offline_passed'] or gate['trajectory_sha256']!=hashlib.sha256(source_path.read_bytes()).hexdigest() or gate['geometry_manifest_sha256']!=hashlib.sha256(REGISTERED_MANIFEST.read_bytes()).hexdigest():raise ValueError('Trajectory validation is missing or stale')
    profile=smooth_profile(np.array(saved['raw_q_rad']),k.velocity,k.acceleration);profile['duration']*=saved.get('time_scale',1.);np.testing.assert_allclose(profile['q'],saved['q_rad'],atol=1e-12,rtol=0)
    attr=env.sock_cloth;indices=mapping.index();right_ids=[indices[n] for n in ARM];left_ids=[indices[n] for n in LEFT_ARM];finger=indices['right_gripper/finger_joint'];left_finger=indices['left_gripper/finger_joint']
    native_ids=mapping.simulator_indices[right_ids];kp=np.array(env.robot.data['joint_stiffness'])[native_ids];kd=np.array(env.robot.data['joint_damping'])[native_ids];phase=dt*kd/(kd+kp*dt)
    attr._send_data('BeginSockCompoundReplay',env.robot.id);env._env.step(2,simulate=False);initial=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda v:v.tolist()));initial_links={r['link']:r for r in initial['links']}
    result=dict(number=saved['number'],source=str(source_path.relative_to(ROOT)),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),passed=False,executed=False,torso_fixed=True,reference_cloth_held=True,dynamic_grasp_validated=False,actual_configuration_failures=[],unexpected_contacts=[],fixed_step_s=dt,drive_phase_compensation_s=phase.tolist())
    result.update(endpoint_task=saved.get('endpoint_task','geometric_grasp_target'),virtual_closure_passed=saved.get('virtual_closure_passed'),left_foot_floor_allowance_m=saved.get('left_foot_floor_allowance_m',0.))
    trace=[];maximum_tracking=0.;maximum_fixed_drift=0.;fixed_angle=0.;last=None
    with (out/'motion-trace.jsonl').open('w') as f:
        def advance(q,qd,stage,t):
            nonlocal maximum_tracking,maximum_fixed_drift,fixed_angle,last
            action[right_ids]=np.asarray(q)-phase*np.asarray(qd);action[finger]=k.hand.joint_for_gap(saved['opening_gap_m'])
            command=mapping.merge_for_simulator(mapping.bound(action),reference);command[mapping.fixed_simulator_indices]=reference[mapping.fixed_simulator_indices]
            env.robot.SetJointPosition(command.tolist());velocity=np.zeros(len(command));velocity[native_ids]=np.rad2deg(qd)
            attr._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=velocity.tolist())));env._env.Simulate(dt,1);attr._send_data('GetSockCompoundReplay',env.robot.id);env._env.step(simulate=False)
            native=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda v:v.tolist()));feedback=mapping.from_simulator(env.robot.data['joint_positions']);actual=feedback[right_ids];left=feedback[left_ids];gap=float(feedback[finger]+k.hand.gap_offset_m);left_gap=float(feedback[left_finger]+k.hand.gap_offset_m);checker.park_left(left,left_gap)
            tcp,rot=native_tcp(native,k);desired=k.pose(q)[0];tracking=float(np.linalg.norm(tcp-desired));maximum_tracking=max(maximum_tracking,tracking)
            for link in native['links']:
                name=link['link']
                if name.startswith(('right_arm/','right_gripper/')):continue
                old=initial_links[name];maximum_fixed_drift=max(maximum_fixed_drift,float(np.linalg.norm(np.array(link['position'])-old['position'])));fixed_angle=max(fixed_angle,float(np.rad2deg(Rotation.from_matrix(Rotation.from_quat(link['rotation']).as_matrix()@Rotation.from_quat(old['rotation']).as_matrix().T).magnitude())))
            unexpected=[]
            for contact in native['contacts']:
                a,b=contact['a_link'],contact['b_link'];moving=a.startswith(('right_arm/','right_gripper/','left_arm/','left_gripper/')) or b.startswith(('right_arm/','right_gripper/','left_arm/','left_gripper/'))
                excluded=bool(a and b and a in checker.group and b in checker.group and checker.excluded(a,b))
                if moving and not excluded and contact['separation_m']< -1e-5:unexpected.append(contact)
            if native['dropped']:raise RuntimeError('Native contact recorder overflow')
            if unexpected:result['unexpected_contacts'].append(dict(step=len(trace),contacts=unexpected))
            if len(trace)%5==0 or stage=='settle':
                check=checker.slacks(actual,gap,delta,details=True)
                if not check['passed']:result['actual_configuration_failures'].append(dict(step=len(trace),check=check))
            record=dict(step=len(trace),stage=stage,stage_time_s=t,requested_q_rad=np.asarray(q).tolist(),actual_q_rad=actual.tolist(),actual_left_q_rad=left.tolist(),actual_gap_m=gap,tcp_world_m=tcp.tolist(),tracking_error_m=tracking,contacts=native['contacts'],cloth_time_s=native['cloth_time_s'],cloth_fixed_steps=native['cloth_fixed_steps'])
            trace.append(record);f.write(json.dumps(record)+'\n');last=(native,tcp,rot,actual)
            if len(trace)%50==0:f.flush();print('FIXED TORSO REPLAY',stage,len(trace),'tracking mm',tracking*1000,'fixed drift um',maximum_fixed_drift*1e6,flush=True)
            return not unexpected and not result['actual_configuration_failures']
        okay=True
        for t in np.r_[np.arange(dt,profile['duration'],dt),profile['duration']]:
            tau=t/profile['duration'];u=10*tau**3-15*tau**4+6*tau**5;qd=profile['curve'](u,1)*(30*tau**2-60*tau**3+30*tau**4)/profile['duration']
            if not advance(profile['curve'](u),qd,'reach',float(t)):okay=False;break
        if okay:
            for t in np.arange(dt,1.+dt/2,dt):
                if not advance(np.array(saved['q_rad'][-1]),np.zeros(7),'settle',float(t)):okay=False;break
    native,tcp,rotation,actual=last;candidate=saved['geometry_record']['candidate'];target=np.array(candidate['grasp_center_unity_world_m'])+delta
    actuals=np.array([v['actual_q_rad'] for v in trace]);vel=np.diff(actuals,axis=0)/dt;acc=np.diff(vel,axis=0)/dt
    result.update(executed=True,steps=len(trace),planned_reach_duration_s=profile['duration'],endpoint_position_error_m=float(np.linalg.norm(tcp-target)),endpoint_orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(rotation@np.array(candidate['rotation_matrix']).T).magnitude())),
        maximum_tracking_error_m=maximum_tracking,maximum_fixed_drift_m=maximum_fixed_drift,maximum_fixed_orientation_drift_deg=fixed_angle,
        actual_hard_joint_limits_passed=bool(np.all(actuals>=k.lower-1e-8) and np.all(actuals<=k.upper+1e-8)),minimum_actual_joint_margin_deg=float(np.rad2deg(np.minimum(actuals-k.lower,k.upper-actuals)).min()),
        maximum_velocity_fraction=float(np.max(abs(vel)/k.velocity)),maximum_acceleration_fraction=float(np.max(abs(acc)/k.acceleration)),initial_clock=initial['cloth_time_s'],final_clock=native['cloth_time_s'],initial_steps=initial['cloth_fixed_steps'],final_steps=native['cloth_fixed_steps'])
    attr.request_particles();env._env.step(simulate=False);result['cloth_error_m']=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max())
    result['passed']=bool(okay and result['endpoint_position_error_m']<=1e-4 and result['endpoint_orientation_error_deg']<=.5 and maximum_fixed_drift<2e-5 and result['cloth_error_m']<5e-6 and result['actual_hard_joint_limits_passed'] and result['maximum_velocity_fraction']<=1 and result['maximum_acceleration_fraction']<=1 and result['final_steps']-result['initial_steps']==len(trace))
    write(out/'replay-result.json',result);print('REPLAY RESULT',result['passed'],'endpoint mm',result['endpoint_position_error_m']*1000,'orientation deg',result['endpoint_orientation_error_deg'],'fixed drift um',maximum_fixed_drift*1e6,flush=True)
    return result
