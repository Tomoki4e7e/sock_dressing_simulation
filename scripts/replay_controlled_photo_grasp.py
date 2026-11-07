#!/usr/bin/env python3
"""Drive either arm through opening, reaching, and skin-safe jaw closure."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,REGISTERED_MANIFEST
from validate_compound_trajectory import smooth_profile
from sock_dressing_simulation.arm_reaching import ARM,ROS_UNITY
from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM
from sock_dressing_simulation.grasp_geometry import HANDEDNESS
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,shared_gripper_geometry_audit
from sock_dressing_simulation.reaching_dataset import verify_dataset_inputs


def jaw_profile(t,duration,start,end):
    tau=np.clip(t/duration,0.,1.);u=10*tau**3-15*tau**4+6*tau**5;du=(30*tau**2-60*tau**3+30*tau**4)/duration
    return start+(end-start)*u,(end-start)*du


def replay(env,mapping,action,reference,k,checker,saved,source_path,delta,expected_cloth,out,dt=.02,collector=None):
    verify_dataset_inputs(saved,ROOT)
    gate=json.loads(source_path.with_name(source_path.name.replace('trajectory','validation')).read_text())
    if not gate['offline_passed'] or not gate['virtual_closure_passed'] or not gate['actual_grasp_geometry_passed'] or gate['trajectory_sha256']!=hashlib.sha256(source_path.read_bytes()).hexdigest() or gate['geometry_manifest_sha256']!=hashlib.sha256(REGISTERED_MANIFEST.read_bytes()).hexdigest():raise ValueError('Approach/capture/closure validation is missing or stale')
    side=saved['moving_side'];model=ControlledArmKinematics(k,side,saved['right_q_rad'],saved['right_gap_m']);checker.closure_side=side
    if side=='left' and not shared_gripper_geometry_audit(k)['passed']:raise ValueError('Left gripper geometry differs from the validated hand')
    profile=smooth_profile(np.array(saved['raw_q_rad']),model.velocity,model.acceleration);profile['duration']*=saved['time_scale'];np.testing.assert_allclose(profile['q'],saved['q_rad'],atol=1e-12,rtol=0)
    indices=mapping.index();right_ids=[indices[n] for n in ARM];left_ids=[indices[n] for n in LEFT_ARM];controlled=left_ids if side=='left' else right_ids;finger=indices[side+'_gripper/finger_joint'];right_finger=indices['right_gripper/finger_joint'];left_finger=indices['left_gripper/finger_joint'];native_ids=mapping.simulator_indices[controlled];native_finger=int(mapping.simulator_indices[finger])
    kp=np.array(env.robot.data['joint_stiffness']);kd=np.array(env.robot.data['joint_damping']);phase=dt*kd[native_ids]/(kd[native_ids]+kp[native_ids]*dt);finger_phase=float(dt*kd[native_finger]/(kd[native_finger]+kp[native_finger]*dt))
    initial_gap=saved['initial_opening_gap_m'];opening=saved['opening_gap_m'];closing=saved['closing_gap_m'];attr=env.sock_cloth;attr._send_data('BeginSockCompoundReplay',env.robot.id);env._env.step(2,simulate=False)
    initial=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda v:v.tolist()));initial_links={r['link']:r for r in initial['links']}
    result=dict(number=saved['number'],moving_side=side,source=str(source_path.relative_to(ROOT)),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),passed=False,executed=False,torso_fixed=True,human_position_unchanged=True,
        reference_cloth_held=True,dynamic_grasp_validated=False,virtual_closure_passed=True,actual_configuration_failures=[],unexpected_contacts=[],fixed_step_s=dt,drive_phase_compensation_s=phase.tolist(),finger_phase_compensation_s=finger_phase,stages=[],endpoint_task='geometric_grasp_target')
    trace=[];maximum_tracking=0.;maximum_fixed_drift=0.;fixed_angle=0.;last=None;closure_gaps=[];all_checks=[]
    def tcp_pose(native):
        link=next(r for r in native['links'] if r['link']==side+'_gripper/gripper_base');R=Rotation.from_quat(link['rotation']).as_matrix()
        return np.array(link['position'])+R@ROS_UNITY@model.hand.grasp_center_urdf,R@ROS_UNITY@HANDEDNESS
    with (out/'motion-trace.jsonl').open('w') as stream:
        def advance(q,qd,gap,gap_rate,stage,t):
            nonlocal maximum_tracking,maximum_fixed_drift,fixed_angle,last
            action[controlled]=np.array(q)-phase*np.asarray(qd);action[finger]=model.hand.joint_for_gap(gap-finger_phase*gap_rate)
            command=mapping.merge_for_simulator(mapping.bound(action),reference);command[mapping.fixed_simulator_indices]=reference[mapping.fixed_simulator_indices];env.robot.SetJointPosition(command.tolist());velocity=np.zeros(len(command));velocity[native_ids]=np.rad2deg(qd);velocity[native_finger]=gap_rate
            if collector is not None:env.robot.GetJointInverseDynamicsForce()
            attr._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=velocity.tolist())));env._env.Simulate(dt,1);attr._send_data('GetSockCompoundReplay',env.robot.id);env._env.step(simulate=False)
            native=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda v:v.tolist()));feedback=mapping.from_simulator(env.robot.data['joint_positions']);right=feedback[right_ids];left=feedback[left_ids];actual=feedback[controlled];right_gap=float(feedback[right_finger]+k.hand.gap_offset_m);left_gap=float(feedback[left_finger]+k.hand.gap_offset_m);actual_gap=left_gap if side=='left' else right_gap
            checker.park_left(left,left_gap);tcp,R=tcp_pose(native);tracking=float(np.linalg.norm(tcp-model.pose(q)[0]));maximum_tracking=max(maximum_tracking,tracking)
            for link in native['links']:
                name=link['link']
                if name.startswith((side+'_arm/',side+'_gripper/')):continue
                old=initial_links[name];maximum_fixed_drift=max(maximum_fixed_drift,float(np.linalg.norm(np.array(link['position'])-old['position'])));fixed_angle=max(fixed_angle,float(np.rad2deg(Rotation.from_matrix(Rotation.from_quat(link['rotation']).as_matrix()@Rotation.from_quat(old['rotation']).as_matrix().T).magnitude())))
            unexpected=[]
            for contact in native['contacts']:
                a,b=contact['a_link'],contact['b_link'];moving=a.startswith(('left_arm/','left_gripper/','right_arm/','right_gripper/')) or b.startswith(('left_arm/','left_gripper/','right_arm/','right_gripper/'));excluded=bool(a and b and a in checker.group and b in checker.group and checker.excluded(a,b))
                if moving and not excluded and contact['separation_m']< -1e-5:unexpected.append(contact)
            if native['dropped']:raise RuntimeError('Native contact recorder overflow')
            if unexpected:result['unexpected_contacts'].append(dict(step=len(trace),stage=stage,contacts=unexpected))
            # During closure, cloth overlap is the intended frozen-cloth capture
            # surrogate. Skin, robot, other arm, and all other pairs stay active.
            closing_stage=stage in ('closure','closed_settle')
            if len(trace)%5==0 or closing_stage or stage=='settle':
                check=checker.slacks(right,right_gap,delta,allow_cloth_overlap=closing_stage,details=True);all_checks.append(dict(step=len(trace),stage=stage,minimum_slack_m=check['minimum_slack_m']))
                if not check['passed']:result['actual_configuration_failures'].append(dict(step=len(trace),stage=stage,check=check))
            if closing_stage:closure_gaps.append(actual_gap)
            record=dict(step=len(trace),stage=stage,stage_time_s=t,requested_q_rad=np.asarray(q).tolist(),actual_q_rad=actual.tolist(),actual_right_q_rad=right.tolist(),actual_left_q_rad=left.tolist(),actual_right_gap_m=right_gap,actual_left_gap_m=left_gap,requested_gap_m=gap,actual_gap_m=actual_gap,tcp_world_m=tcp.tolist(),tracking_error_m=tracking,cloth_time_s=native['cloth_time_s'],cloth_fixed_steps=native['cloth_fixed_steps'],contacts=native['contacts'])
            trace.append(record);stream.write(json.dumps(record)+'\n');last=(native,tcp,R,actual,actual_gap)
            if collector is not None:collector.append(env,mapping,record)
            if len(trace)%100==0:stream.flush();print('CONTROLLED REPLAY',side,stage,len(trace),'tracking mm',tracking*1000,'fixed drift um',maximum_fixed_drift*1e6,flush=True)
            return not unexpected and not result['actual_configuration_failures']
        okay=True;zero=np.zeros(7);qstart=np.array(saved['start_q_rad']);qgoal=np.array(saved['q_rad'][-1])
        if abs(opening-initial_gap)>1e-9:
            duration=saved['preopen_duration_s'];result['stages'].append(dict(stage='opening',duration_s=duration))
            for t in np.r_[np.arange(dt,duration,dt),duration]:
                g,gd=jaw_profile(t,duration,initial_gap,opening)
                if not advance(qstart,zero,g,gd,'opening',float(t)):okay=False;break
        if okay:
            result['stages'].append(dict(stage='reach',duration_s=profile['duration']))
            for t in np.r_[np.arange(dt,profile['duration'],dt),profile['duration']]:
                tau=t/profile['duration'];u=10*tau**3-15*tau**4+6*tau**5;qd=profile['curve'](u,1)*(30*tau**2-60*tau**3+30*tau**4)/profile['duration']
                if not advance(profile['curve'](u),qd,opening,0.,'reach',float(t)):okay=False;break
        if okay:
            for t in np.arange(dt,1.+dt/2,dt):
                if not advance(qgoal,zero,opening,0.,'settle',float(t)):okay=False;break
        if okay:
            duration=saved['closure_duration_s'];result['stages'].append(dict(stage='closure',duration_s=duration))
            for t in np.r_[np.arange(dt,duration,dt),duration]:
                g,gd=jaw_profile(t,duration,opening,closing)
                if not advance(qgoal,zero,g,gd,'closure',float(t)):okay=False;break
        if okay:
            for t in np.arange(dt,1.+dt/2,dt):
                if not advance(qgoal,zero,closing,0.,'closed_settle',float(t)):okay=False;break
    native,tcp,R,actual,actual_gap=last;target=np.array(saved['geometry_record']['candidate']['grasp_center_unity_world_m'])+delta
    actuals=np.array([v['actual_q_rad'] for v in trace]);vel=np.diff(actuals,axis=0)/dt;acc=np.diff(vel,axis=0)/dt;gaps=np.array([v['actual_gap_m'] for v in trace]);gap_velocity=np.diff(gaps)/dt
    result.update(executed=True,steps=len(trace),planned_reach_duration_s=profile['duration'],endpoint_position_error_m=float(np.linalg.norm(tcp-target)),endpoint_orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(R@np.array(saved['geometry_record']['candidate']['rotation_matrix']).T).magnitude())),
        final_actual_gap_m=actual_gap,closing_gap_error_m=abs(actual_gap-closing),closure_completed=bool(okay and len(closure_gaps)>0 and abs(actual_gap-closing)<2e-4),maximum_tracking_error_m=maximum_tracking,maximum_fixed_drift_m=maximum_fixed_drift,maximum_fixed_orientation_drift_deg=fixed_angle,
        actual_hard_joint_limits_passed=bool(np.all(actuals>=model.lower-1e-8) and np.all(actuals<=model.upper+1e-8)),minimum_actual_joint_margin_deg=float(np.rad2deg(np.minimum(actuals-model.lower,model.upper-actuals)).min()),
        maximum_velocity_fraction=float(np.max(abs(vel)/model.velocity)),maximum_acceleration_fraction=float(np.max(abs(acc)/model.acceleration)),maximum_gap_velocity_m_s=float(abs(gap_velocity).max()),initial_clock=initial['cloth_time_s'],final_clock=native['cloth_time_s'],initial_steps=initial['cloth_fixed_steps'],final_steps=native['cloth_fixed_steps'],actual_distance_checks=all_checks)
    attr.request_particles();env._env.step(simulate=False);result['cloth_error_m']=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max())
    result['passed']=bool(okay and result['closure_completed'] and result['endpoint_position_error_m']<=1e-4 and result['endpoint_orientation_error_deg']<=.5 and maximum_fixed_drift<2e-5 and result['cloth_error_m']<5e-6 and result['actual_hard_joint_limits_passed'] and result['maximum_velocity_fraction']<=1 and result['maximum_acceleration_fraction']<=1 and result['maximum_gap_velocity_m_s']<=.02 and result['final_steps']-result['initial_steps']==len(trace))
    write(out/'replay-result.json',result);print('CONTROLLED REPLAY RESULT',side,result['passed'],'closure',result['closure_completed'],'end mm',result['endpoint_position_error_m']*1000,'gap',actual_gap,'fixed drift um',maximum_fixed_drift*1e6,flush=True)
    return result
