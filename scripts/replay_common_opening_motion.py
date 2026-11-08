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
from common_opening_replay_support import verify_source, make_profile, evaluate_grasp, motion_metrics, threshold_check


def jaw_profile(t,duration,start,end):
    tau=np.clip(t/duration,0.,1.);u=10*tau**3-15*tau**4+6*tau**5;du=(30*tau**2-60*tau**3+30*tau**4)/duration
    return start+(end-start)*u,(end-start)*du


def replay(env,mapping,action,reference,k,checker,saved,source_path,delta,expected_cloth,out,dt=.02,collector=None):
    side=saved['side'];route=saved['route'];model=ControlledArmKinematics(k,side,action[[mapping.index()[n] for n in ARM]],.04);checker.closure_side=side
    if not shared_gripper_geometry_audit(k)['passed']:raise ValueError('Gripper geometry audit failed')
    profile=make_profile(route,model)
    indices=mapping.index();right_ids=[indices[n] for n in ARM];left_ids=[indices[n] for n in LEFT_ARM];controlled=left_ids if side=='left' else right_ids;finger=indices[side+'_gripper/finger_joint'];right_finger=indices['right_gripper/finger_joint'];left_finger=indices['left_gripper/finger_joint'];native_ids=mapping.simulator_indices[controlled];native_finger=int(mapping.simulator_indices[finger])
    kp=np.array(env.robot.data['joint_stiffness']);kd=np.array(env.robot.data['joint_damping']);phase=dt*kd[native_ids]/(kd[native_ids]+kp[native_ids]*dt);finger_phase=float(dt*kd[native_finger]/(kd[native_finger]+kp[native_finger]*dt))
    initial_gap=.04;opening=route['opening_gap_m'];closing=.001;attr=env.sock_cloth;attr._send_data('BeginSockCompoundReplay',env.robot.id);env._env.step(2,simulate=False)
    initial=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda v:v.tolist()));initial_links={r['link']:r for r in initial['links']}
    result=dict(number=saved['condition']['number'],condition=saved['condition'],moving_side=side,source=str(source_path.relative_to(ROOT)),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),passed=False,executed=False,torso_fixed=True,human_position_unchanged=True,
        reference_cloth_held=True,dynamic_grasp_validated=False,virtual_closure_passed=None,actual_configuration_failures=[],unexpected_contacts=[],fixed_step_s=dt,drive_phase_compensation_s=phase.tolist(),finger_phase_compensation_s=finger_phase,stages=[],endpoint_task='unchanged_common_opening_approach_target')
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
                check=threshold_check(checker,right,right_gap,delta,closing_stage);all_checks.append(dict(step=len(trace),stage=stage,minimum_slack_m=check['minimum_slack_m']))
                if not check['passed']:result['actual_configuration_failures'].append(dict(step=len(trace),stage=stage,check=check))
            if closing_stage:closure_gaps.append(actual_gap)
            record=dict(step=len(trace),stage=stage,stage_time_s=t,requested_q_rad=np.asarray(q).tolist(),actual_q_rad=actual.tolist(),actual_right_q_rad=right.tolist(),actual_left_q_rad=left.tolist(),actual_right_gap_m=right_gap,actual_left_gap_m=left_gap,requested_gap_m=gap,actual_gap_m=actual_gap,tcp_world_m=tcp.tolist(),tracking_error_m=tracking,cloth_time_s=native['cloth_time_s'],cloth_fixed_steps=native['cloth_fixed_steps'],contacts=native['contacts'])
            trace.append(record);stream.write(json.dumps(record)+'\n');last=(native,tcp,R,actual,actual_gap)
            if collector is not None:collector.append(env,mapping,record)
            if len(trace)%100==0:stream.flush();print('CONTROLLED REPLAY',side,stage,len(trace),'tracking mm',tracking*1000,'fixed drift um',maximum_fixed_drift*1e6,flush=True)
            return not unexpected and not result['actual_configuration_failures']
        okay=True;zero=np.zeros(7);qstart=np.array(saved['start_q_rad']);qgoal=np.array(route['q_rad'][-1])
        if abs(opening-initial_gap)>1e-9:
            duration=max(2.,1.875*abs(opening-initial_gap)/.01);result['stages'].append(dict(stage='opening',duration_s=duration))
            for t in np.r_[np.arange(dt,duration,dt),duration]:
                g,gd=jaw_profile(t,duration,initial_gap,opening)
                if not advance(qstart,zero,g,gd,'opening',float(t)):okay=False;break
        if okay:
            result['stages'].append(dict(stage='reach',duration_s=profile['duration']))
            for t in np.r_[np.arange(dt,profile['duration'],dt),profile['duration']]:
                q,qd=profile['evaluate'](t)
                if not advance(q,qd,opening,0.,'reach',float(t)):okay=False;break
        if okay:
            for t in np.arange(dt,1.+dt/2,dt):
                if not advance(qgoal,zero,opening,0.,'settle',float(t)):okay=False;break
        # Freeze the approach verdict before any optional jaw closure.
        result['approach']=motion_metrics(trace,initial,last,model,route,dt,maximum_fixed_drift,result,okay)
        result['approach']['passed']=bool(result['approach']['passed'] and okay)
        if okay:
            native,tcp,R,actual,actual_gap=last
            result['planned_endpoint_grasp']=evaluate_grasp(saved,state=None,k=k,checker=checker,right=trace[-1]['actual_right_q_rad'],left=trace[-1]['actual_left_q_rad'],gap=.001,delta=delta)
            # As in the previous A replay, physical closure requires the
            # geometric grasp gate. The unchanged approach has already run.
            if not result['planned_endpoint_grasp']['passed']:
                result['closure_not_executed_reason']='Previous A geometric grasp gate failed at the measured endpoint'
                okay=False
        if okay:
            closure_checks=[]
            for g in np.linspace(actual_gap,closing,61):
                checker.park_left(trace[-1]['actual_left_q_rad'],g if side=='left' else trace[-1]['actual_left_gap_m'])
                closure_checks.append(checker.slacks(trace[-1]['actual_right_q_rad'],g if side=='right' else trace[-1]['actual_right_gap_m'],delta,allow_cloth_overlap=True,details=True))
            result['virtual_closure_checks']=closure_checks
            result['virtual_closure_passed']=all(c['passed'] for c in closure_checks)
            okay=result['virtual_closure_passed']
        if okay:
            duration=10.;result['stages'].append(dict(stage='closure',duration_s=duration))
            for t in np.r_[np.arange(dt,duration,dt),duration]:
                g,gd=jaw_profile(t,duration,opening,closing)
                if not advance(qgoal,zero,g,gd,'closure',float(t)):okay=False;break
        if okay:
            for t in np.arange(dt,1.+dt/2,dt):
                if not advance(qgoal,zero,closing,0.,'closed_settle',float(t)):okay=False;break
    native,tcp,R,actual,actual_gap=last
    result.update(motion_metrics(trace,initial,last,model,route,dt,maximum_fixed_drift,result,okay))
    result.update(executed=True,steps=len(trace),planned_reach_duration_s=profile['duration'],final_actual_gap_m=actual_gap,closing_gap_error_m=abs(actual_gap-closing),closure_completed=bool(okay and closure_gaps and abs(actual_gap-closing)<2e-4),maximum_tracking_error_m=maximum_tracking,maximum_fixed_orientation_drift_deg=fixed_angle,actual_distance_checks=all_checks)
    attr.request_particles();env._env.step(simulate=False);result['cloth_error_m']=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max())
    result['actual_closed_grasp']=evaluate_grasp(saved,state=None,k=k,checker=checker,right=trace[-1]['actual_right_q_rad'],left=trace[-1]['actual_left_q_rad'],gap=actual_gap,delta=delta)
    result['actual_closed_grasp']['physical_closure_executed']=bool(closure_gaps)
    result['approach']['cloth_error_m']=result['cloth_error_m']
    result['approach']['passed']=bool(result['approach']['passed'] and result['cloth_error_m']<5e-6)
    result['closure_dynamics_passed']=bool(result['passed'] and result['closure_completed'] and result['cloth_error_m']<5e-6)
    result['grasp_passed']=bool(result['approach']['passed'] and result['closure_dynamics_passed'] and result['actual_closed_grasp']['passed'])
    result['passed']=bool(result['approach']['passed'] and result['grasp_passed'])
    result['scope']='PhysX arm/jaw drives with frozen cloth; previous A geometric capture rule. No friction, retention or pull test.'
    write(out/'replay-result.json',result);print('COMMON REPLAY RESULT',side,saved['condition']['id'],'approach',result['approach']['passed'],'grasp',result['grasp_passed'],flush=True)
    return result
