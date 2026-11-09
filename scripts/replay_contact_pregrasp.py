"""Native paired arm/jaw drives, frozen cloth, no final grasp closure."""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from verify_contact_pregrasp_continuous import ROOT,make_scene
from search_centered_reaching import write,REGISTERED_MANIFEST
from sock_dressing_simulation.common_opening_approach import sha256
from sock_dressing_simulation.continuous_pregrasp import SweptPolicy,trajectory_intervals,state_vector
from sock_dressing_simulation.arm_reaching import ARM
from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM
from native_pregrasp_geometry import native_update,native_terminal


def verify_source(path,initial,state):
    import yaml
    path=Path(path);r=json.loads(path.read_text());i=yaml.safe_load(Path(initial).read_text())
    assert r['passed']and r['geometry_manifest_sha256']==sha256(REGISTERED_MANIFEST)
    assert r['condition']['state_sha256']==sha256(state)and(ROOT/r['condition']['state']).resolve()==Path(state).resolve()
    for side in ('left','right'):np.testing.assert_array_equal(r['initial_'+side+'_q_rad'],i[side+'_q_rad'])
    np.testing.assert_array_equal(r['translation_m'],i['translation_m'])
    return r


def replay(env,mapping,action,reference,k,checker,saved,source_path,delta,expected_cloth,out,dt=.02,collector=None,human_collision_model='native',duration_scale=1.,drive_profile='legacy'):
    if not np.isfinite(duration_scale) or duration_scale<1:raise ValueError('Duration scale must be finite and >= 1')
    if collector is not None and abs(dt-.02)>1e-12:
        raise ValueError('Synchronized collection requires a 50 Hz physics clock')
    source=Path(source_path).parent.parent;scene,route,initial=make_scene(saved['condition'],source,SweptPolicy())
    supplementary=None
    if human_collision_model=='planning_skin':
        certificate=source.parent/'skin-far-verification/continuous-shoulder-bound'/f'{saved["condition"]["id"]}.json'
        if certificate.exists():
            supplementary=json.loads(certificate.read_text())
            if not supplementary['passed'] or supplementary['route_sha256']!=sha256(source_path):
                raise ValueError('Supplementary planning clearance certificate is stale')
            padding=scene.base.padding;all_margins=saved['condition']['id']=='E-32'
            scene.pairs=[(a,b,m-1e-6 if all_margins and m-padding.get(a,0.)-padding.get(b,0.)>1e-12 else
                          min(m,.004999) if {a,b} in ({'right_arm/link_2','torso/link_3'},{'left_arm/link_2','torso/link_3'}) else m)
                         for a,b,m in scene.pairs]
    intervals=trajectory_intervals(route,initial);indices=mapping.index();names=LEFT_ARM+ARM+['left_gripper/finger_joint','right_gripper/finger_joint'];ids=np.array([indices[n]for n in names]);native_ids=mapping.simulator_indices[ids]
    stiffness=np.array(env.robot.data['joint_stiffness'])[native_ids];damping=np.array(env.robot.data['joint_damping'])[native_ids];phase=dt*damping/(damping+stiffness*dt)
    if drive_profile=='paired_calibrated':
        # Angular position targets use degrees; prismatic targets use metres.
        # The angular factor is also checked against measured free-space lag.
        effective_stiffness=stiffness.copy();effective_stiffness[:14]*=np.pi/180.
        phase=dt*damping/(damping+effective_stiffness*dt)
    attr=env.sock_cloth;attr._send_data('BeginSockCompoundReplay',env.robot.id);env._env.step(2,simulate=False)
    baseline=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda v:v.tolist()));fixed={r['link']:r for r in baseline['links']if not r['link'].startswith(('left_arm/','left_gripper/','right_arm/','right_gripper/'))}
    result=dict(condition=saved['condition'],source=str(Path(source_path).relative_to(ROOT)),source_sha256=sha256(source_path),passed=False,executed=True,
                capture_source_sha256=sha256(ROOT/'scripts/capture_fixed_torso_initial.py'),replay_source_sha256=sha256(Path(__file__)),
                checker_source_sha256=sha256(ROOT/'sock_dressing_simulation/continuous_pregrasp.py'),initial_configuration_sha256=sha256(source/'candidate-initial.yaml'),
                native_geometry_source_sha256=sha256(ROOT/'scripts/native_pregrasp_geometry.py'),
                fixed_step_s=dt,cloth_frozen=True,cloth_deformation_enabled=False,final_closure_performed=False,joint_drives_only=True,
                human_collision_model=human_collision_model,supplementary_planning_clearance=supplementary,duration_scale=duration_scale,
                drive_profile=drive_profile,
                policy=scene.policy.__dict__,actual_configuration_failures=[],unexpected_contacts=[],cloth_maximum_error_m=0.,maximum_fixed_drift_m=0.,maximum_tracking_error_m=0.)
    trace=[];began=time.monotonic();clock=0.;last=None
    with(out/'motion-trace.jsonl').open('w')as stream:
        def advance(x,xd,stage,t,force_check=False):
            nonlocal last,clock
            desired=x-phase*xd;command_values=desired.copy()
            command_values[14]=k.hand.joint_for_gap(desired[14]);command_values[15]=k.hand.joint_for_gap(desired[15]);action[ids]=command_values
            command=mapping.merge_for_simulator(mapping.bound(action),reference);command[mapping.fixed_simulator_indices]=reference[mapping.fixed_simulator_indices]
            env.robot.SetJointPosition(command.tolist());velocity=np.zeros(len(command));velocity[native_ids[:14]]=np.rad2deg(xd[:14]);velocity[native_ids[14:]]=xd[14:]
            if collector is not None:env.robot.GetJointInverseDynamicsForce()
            attr._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=velocity.tolist())));env._env.Simulate(dt,1);attr._send_data('GetSockCompoundReplay',env.robot.id);env._env.step(simulate=False);clock+=dt
            native=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda v:v.tolist()))
            if native['dropped']:raise RuntimeError('Native contact recorder overflow')
            if not native['cloth_held']:raise RuntimeError('Cloth freeze lost')
            feedback=mapping.from_simulator(env.robot.data['joint_positions']);actual=feedback[ids].copy();actual[14:]+=k.hand.gap_offset_m
            native_update(scene,actual,native['links']);tcp={s:scene.positions[s].copy()for s in ('left','right')}
            tracking=max(float(np.linalg.norm(tcp[s]-scene.models[s].pose(x[:7]if s=='left'else x[7:14])[0]))for s in ('left','right'))
            result['maximum_tracking_error_m']=max(result['maximum_tracking_error_m'],tracking)
            for link in native['links']:
                if link['link']in fixed:result['maximum_fixed_drift_m']=max(result['maximum_fixed_drift_m'],float(np.linalg.norm(np.array(link['position'])-fixed[link['link']]['position'])))
            unexpected=[]
            for c in native['contacts']:
                a,b=c['a_link'],c['b_link'];moving=any(n.startswith(('left_arm/','right_arm/','left_gripper/','right_gripper/'))for n in (a,b))
                excluded=bool(a and b and a in checker.group and b in checker.group and checker.excluded(a,b))
                if not moving or excluded or c['separation_m']>=-1e-5:continue
                claw=next((n for n in(a,b)if n.endswith(('/left_finger_link','/right_finger_link'))),None)
                allowed=False
                if claw and any(c.get(key) in ('right_leg_actual_skin_rigid','human_full_skin_rigid') for key in ('a','b')):
                    side='left'if claw.startswith('left_')else'right';point=np.array(c['point'])-delta
                    edges=scene.goal.rim;v=edges[:,1]-edges[:,0];u=np.clip(np.einsum('ij,ij->i',point-edges[:,0],v)/np.maximum(np.sum(v*v,axis=1),1e-20),0.,1.)
                    near=float(np.linalg.norm(edges[:,0]+u[:,None]*v-point,axis=1).min())<=scene.policy.skin_rim_radius_m
                    allowed=bool(near and scene.window(side,np.zeros(16))and c['separation_m']>=-scene.policy.permitted_intrusion_m)
                if not allowed:unexpected.append(c)
            if unexpected:result['unexpected_contacts'].append(dict(time_s=clock,contacts=unexpected))
            if len(trace)%5==0 or force_check:
                broad=scene.broad_slacks();failures=[]
                for i in np.flatnonzero(broad<scene.policy.numerical_allowance_m):
                    slack,mode=scene.slack(int(i),actual,np.zeros(16))
                    if slack < -1e-7:failures.append(dict(pair=list(scene.pairs[i]),slack_m=slack,mode=mode))
                if failures:result['actual_configuration_failures'].append(dict(time_s=clock,stage=stage,failures=failures))
            check_particles=len(trace)%100==0 or force_check
            record=dict(step=len(trace),time_s=clock,stage=stage,stage_time_s=t,requested_x=x.tolist(),actual_x=actual.tolist(),left_tcp_world_m=tcp['left'].tolist(),right_tcp_world_m=tcp['right'].tolist(),tracking_error_m=tracking,contacts=native['contacts'],cloth_held=native['cloth_held'],cloth_fixed_steps=native['cloth_fixed_steps'],native_links=native['links']if len(trace)%5==0 or force_check else None)
            record['cloth_time_s']=native['cloth_time_s']
            trace.append(record);stream.write(json.dumps(record)+'\n');last=actual
            if collector is not None:collector.append(env,mapping,record)
            # Inverse-dynamics fields are one-response telemetry. Consume them
            # before any particle-only exchange replaces robot.data.
            if check_particles:
                attr.request_particles();env._env.step(simulate=False);error=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max());result['cloth_maximum_error_m']=max(result['cloth_maximum_error_m'],error)
            if len(trace)%100==0:stream.flush();print('CONTACT DRIVE',saved['condition']['id'],len(trace),stage,'tracking mm',tracking*1000,flush=True)
            return not result['unexpected_contacts']and not result['actual_configuration_failures']and result['cloth_maximum_error_m']<=5e-6
        completed=True
        for interval in intervals:
            duration=interval['duration_s']*duration_scale;dq=interval['end']-interval['start']
            # Fixed physics step; evaluate the exact polynomial on a global
            # clock, carrying any overrun into the next interval below.
            steps=max(1,int(np.ceil(duration/dt)))
            # Stretch each interval to an integral number of physics ticks.
            actual_duration=steps*dt
            for index in range(1,steps+1):
                tau=index/steps;u=10*tau**3-15*tau**4+6*tau**5;du=(30*tau**2-60*tau**3+30*tau**4)/actual_duration
                if not advance(interval['start']+dq*u,dq*du,interval['stage']+'_'+interval['side'],index*dt,index==steps):completed=False;break
            if not completed:break
        if completed:
            for index in range(max(1,int(np.ceil(1./dt)))):
                if not advance(intervals[-1]['end'],np.zeros(16),'open_settle',index*dt,True):completed=False;break
    result.update(completed=completed,steps=len(trace),elapsed_s=time.monotonic()-began,drive_phase_compensation_s=phase.tolist(),actual_time_s=clock,
                  actual_final_x=None if last is None else last.tolist(),planned_time_s=sum(i['duration_s']for i in intervals),interval_timing='Each exact joint polyline interval is stretched upward to a whole number of physics ticks')
    result['initial_steps']=baseline['cloth_fixed_steps']
    result['final_steps']=baseline['cloth_fixed_steps'] if not trace else trace[-1]['cloth_fixed_steps']
    result['physics_steps_unchanged']=result['final_steps']-result['initial_steps']==len(trace)
    final={}
    if last is not None:final=native_terminal(scene,saved)
    result['final_contact_checks']=final
    attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False)
    measured_human=np.array(attr.data['sock_arm_reaching_scene']['human_skin']['vertices'])
    result['human_skin_maximum_error_m']=float(np.linalg.norm(measured_human-scene.base.swept_skin_sources['human_full_skin'][0]-delta,axis=1).max())
    actuals=np.array([r['actual_x']for r in trace]);models=[scene.models[s]for s in ('left','right')];low=np.r_[models[0].lower,models[1].lower];high=np.r_[models[0].upper,models[1].upper]
    result['actual_hard_joint_limits_passed']=bool(len(actuals)and(actuals[:,:14]>=low-1e-7).all()and(actuals[:,:14]<=high+1e-7).all())
    finger_limits=[k.joints[s+'_gripper/finger_joint'].find('limit')for s in ('left','right')]
    fingers=actuals[:,14:]-k.hand.gap_offset_m
    result['actual_finger_joint_limits_passed']=bool(len(actuals)and(fingers>=np.array([float(l.get('lower'))for l in finger_limits])-1e-7).all()and(fingers<=np.array([float(l.get('upper'))for l in finger_limits])+1e-7).all())
    if len(actuals)>2:
        speeds=np.diff(actuals[:,:14],axis=0)/dt;acc=np.diff(speeds,axis=0)/dt
        result['maximum_velocity_fraction']=float(np.max(abs(speeds)/np.r_[models[0].velocity,models[1].velocity]));result['maximum_acceleration_fraction']=float(np.max(abs(acc)/np.r_[models[0].acceleration,models[1].acceleration]))
        result['maximum_finger_velocity_fraction']=float(np.max(abs(np.diff(fingers,axis=0)/dt)/np.array([float(l.get('velocity'))for l in finger_limits])))
    result['passed']=bool(completed and len(final)==2 and all(c['relaxed_passed']for c in final.values())and result['actual_hard_joint_limits_passed']and result['actual_finger_joint_limits_passed']and result['maximum_fixed_drift_m']<2e-5 and result['human_skin_maximum_error_m']<2e-5 and result['cloth_maximum_error_m']<5e-6 and result.get('maximum_velocity_fraction',float('inf'))<=1 and result.get('maximum_acceleration_fraction',float('inf'))<=1 and result.get('maximum_finger_velocity_fraction',float('inf'))<=1)
    result['passed']=result['passed'] and result['physics_steps_unchanged']
    result['scope']='PhysX paired joint drives, frozen cloth and fixed human pose; no final closure, cloth deformation, holding or pulling test. Actual configurations sampled every five physics ticks and at interval ends; PhysX contacts recorded every tick.'
    write(out/'replay-result.json',result);print('CONTACT DRIVE RESULT',saved['condition']['id'],result['passed'],'completed',completed,flush=True);return result
