#!/usr/bin/env python3
"""Replay saved PCHIP routes with actual articulation drives and PhysX steps.

The reference cloth is held throughout reaching, matching the saved planning
problem. This is a reaching validation, not a claim of dynamic cloth grasping.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.joints import JointMap
from sock_dressing_simulation.arm_reaching import Kinematics,ROS_UNITY,HANDEDNESS
from search_compound_reaching import target_data
from search_translated_reaching import load
from apron_reaching_common import overlay_robot,AuditedScene
from validate_compound_trajectory import smooth_profile
from photograph_grasp_candidates import photograph


def route_inputs(path):
    saved=json.loads(path.read_text());old=json.loads(path.with_name(path.name.replace('trajectory','validation')).read_text());row=json.loads((ROOT/old['source']).read_text());candidate,state,_=target_data(row);attempt=row['attempts'][old['trial']]
    return saved,candidate,state,attempt


def native_tcp(row,k):
    link=next(p for p in row['links'] if p['link']=='right_gripper/gripper_base');R=Rotation.from_quat(link['rotation']).as_matrix();position=np.array(link['position'])+R@ROS_UNITY@k.hand.grasp_center_urdf
    return position,R@ROS_UNITY@HANDEDNESS


def replay(path,args):
    saved,candidate,state,attempt=route_inputs(path);number=saved['number'];out=args.output/f'pose-{number:02d}';out.mkdir(parents=True,exist_ok=True);delta=np.array(saved['translation_m']);expected_cloth=np.array(state['particles_world'])+delta
    config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number');settings['toe_contact']=None;settings['gravity_drape']=None
    config['rcareworld'].update(executable=args.player,graphics=True,port=args.port,log_level=0);config['assets']['output_dir']=str(out/'assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario);mapping=JointMap.from_config(config);pattern=next(p for p in read_patterns(Path(workbook)) if p.number==number);payload=json.loads((args.registration/'payload.json').read_text())
    source_scene=overlay_robot(load(number),json.loads((ROOT/'artifacts/apron-reaching/robot-scene.json').read_text()));source_scene['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',source_scene);checker=AuditedScene(k,source_scene,state['triangles'],args.registration/'manifest.json',ROOT)
    raw=np.array(attempt['trajectory']['start_route_q']+attempt['trajectory']['approach_q'][1:]);profile=smooth_profile(raw,k.velocity,k.acceleration);np.testing.assert_allclose(profile['q'],saved['q_rad'],rtol=0,atol=1e-12)
    result=dict(number=number,source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_cloth_held=True,dynamic_grasp_validated=False,executed=False,passed=False,stages=[],unexpected_contacts=[],actual_configuration_failures=[],images=[]);began=time.monotonic()
    with SockDressingEnv(config) as env:
        # The saved bulged cuff has a refined 1057-vertex topology. Load its
        # actual rest OBJ, rather than the default 801-particle sock.
        rest_obj=ROOT/f'artifacts/bulged-cuff/pose-{number:02d}/rest.obj'
        env.load(Path(prepared['runtime_urdf']),rest_obj,initial_joints=scenario.initial_joints);env.apply_scenario(scenario);attr=env.sock_cloth
        attr.set_pattern_held(False);attr.set_pattern_foot_pose(pattern.pitch_deg,settings['signs']['roll']*pattern.roll_deg);env._env.step(2,simulate=False);attr.request_preinsert_source();env._env.step(2,simulate=False)
        native_rest=np.array(attr.data['sock_preinsert_source']['rest_positions']);rest=np.array(state['source']['rest_positions']);distance,order=cKDTree(rest).query(native_rest)
        if len(native_rest)!=len(rest) or len(set(order.tolist()))!=len(rest) or distance.max()>5e-6:raise ValueError('Native refined cloth topology differs')
        expected_cloth=expected_cloth[order];result['rest_match_error_m']=float(distance.max());result['particle_count']=len(rest)
        attr._send_data('TranslateSockReplayScene',env.robot.id,*map(float,delta));env._env.step(2,simulate=False);attr.initialize_preinsert(expected_cloth,state['opening_normal']);env._env.step(2,simulate=False);attr.set_pattern_held(True);attr.request_particles();env._env.step(2,simulate=False)
        result['initial_cloth_error_m']=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max())
        attr._send_data('InstallSockRigidSkinContact',env.robot.id,0);env._env.step(2,simulate=False);attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False);native_scene=copy.deepcopy(attr.data['sock_arm_reaching_scene']);(out/'prepared-scene.json').write_text(json.dumps(native_scene))
        result['initial_human_error_m']=float(np.linalg.norm(np.array(native_scene['human_skin']['vertices'])-np.array(source_scene['human_skin']['vertices'])-delta,axis=1).max())
        # Compare every rigid obstacle corner/vertex, including the stationary
        # floor. Name and kind must agree before any motion is attempted.
        from sock_dressing_simulation.arm_reaching import shape_vertices
        native_obstacles={p['name']:p for p in native_scene['obstacles']};obstacle_errors=[]
        for p in source_scene['obstacles']:
            if p['name'] not in native_obstacles:raise ValueError('Missing obstacle '+p['name'])
            actual=shape_vertices(native_obstacles[p['name']]);expected=shape_vertices(p)+(0 if 'Ground' in p['name'] else delta);error=max(cKDTree(expected).query(actual)[0].max(),cKDTree(actual).query(expected)[0].max());obstacle_errors.append(dict(name=p['name'],error_m=float(error)))
        result['obstacle_errors']=obstacle_errors
        if max(result['initial_cloth_error_m'],result['initial_human_error_m'],max(p['error_m'] for p in obstacle_errors))>2e-5:
            (out/'result.json').write_text(json.dumps(result,indent=2));raise RuntimeError('Prepared scene differs from saved planning geometry')
        attr._send_data('InstallSockCompoundGeometry',env.robot.id,json.dumps(payload));env._env.step(3,simulate=False);assert attr.data['sock_compound_geometry']['part_count']==len(payload['parts'])
        attr._send_data('BeginSockCompoundReplay',env.robot.id);env._env.step(2,simulate=False);initial=copy.deepcopy(attr.data['sock_compound_replay']);result['initial_clock']=initial['cloth_time_s'];result['initial_steps']=initial['cloth_fixed_steps']
        (out/'initial-robot-data.json').write_text(json.dumps(env.robot.data,default=lambda value:value.tolist()))
        if args.prepare_only:
            result['prepared_only']=True;(out/'result.json').write_text(json.dumps(result,indent=2));return result
        attr._send_data('ConfigureSockReplayDrives',env.robot.id,args.hold_stiffness,args.hold_damping);env._env.step(simulate=False);result['drive_settings']=copy.deepcopy(attr.data['sock_replay_drive_settings'])
        attr._send_data('SetSockReplayFixedFloorPolicy',env.robot.id,args.ignore_fixed_chassis_floor);env._env.step(simulate=False);result['fixed_chassis_floor_policy']=copy.deepcopy(attr.data['sock_replay_fixed_floor_policy'])
        action=np.array(scenario.initial_joints);name_to_action=mapping.index();arm_indices=[name_to_action[n] for n in saved['joint_names']];finger_index=name_to_action['right_gripper/finger_joint'];initial_links={p['link']:p for p in initial['links']};maximum_tracking=0.;maximum_fixed_drift=0.;step=0
        fixed_simulator_reference=np.array(env.robot.data['joint_positions'],dtype=float).copy()
        native_arm_indices=mapping.simulator_indices[arm_indices];kp=np.array(env.robot.data['joint_stiffness'])[native_arm_indices];kd=np.array(env.robot.data['joint_damping'])[native_arm_indices];drive_phase=args.dt*kd/(kd+kp*args.dt);result['drive_phase_compensation_s']=drive_phase.tolist()
        trace=(out/'trace.jsonl').open('w')
        def advance(q,gap,stage,stage_time,qd=None,gap_rate=0.):
            nonlocal maximum_tracking,maximum_fixed_drift,step
            qd=np.zeros(7) if qd is None else np.asarray(qd);action[arm_indices]=np.asarray(q)-drive_phase*qd;action[finger_index]=k.hand.joint_for_gap(gap);command=mapping.merge_for_simulator(mapping.bound(action),fixed_simulator_reference);env.robot.SetJointPosition(command.tolist())
            velocity=np.zeros(len(command));velocity[mapping.simulator_indices[arm_indices]]=np.rad2deg(np.zeros(7) if qd is None else qd);velocity[mapping.simulator_indices[finger_index]]=gap_rate
            attr._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=velocity.tolist())));env._env.Simulate(args.dt,1);attr._send_data('GetSockCompoundReplay',env.robot.id);env._env.step(simulate=False);native=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda value:value.tolist()));feedback=env.robot_signals()['angle'];actual_q=feedback[arm_indices];position,rotation=native_tcp(native,k);desired,_=k.pose(q);tracking=float(np.linalg.norm(position-desired));maximum_tracking=max(maximum_tracking,tracking)
            drift,drift_link=max((np.linalg.norm(np.array(p['position'])-initial_links[p['link']]['position']),p['link']) for p in native['links'] if not p['link'].startswith(('right_arm/','right_gripper/')));maximum_fixed_drift=max(maximum_fixed_drift,float(drift));unexpected=[]
            for contact in native['contacts']:
                a,b=contact['a_link'],contact['b_link'];excluded=bool(a and b and a in checker.group and b in checker.group and checker.excluded(a,b))
                moving=a.startswith(('right_arm/','right_gripper/')) or b.startswith(('right_arm/','right_gripper/'))
                if moving and not excluded and contact['separation_m'] < -1e-5:unexpected.append(contact)
            if native['dropped']:raise RuntimeError('Contact log overflow')
            if unexpected:result['unexpected_contacts'].append(dict(step=step,stage=stage,contacts=unexpected))
            if step%10==0 or stage=='settle':
                actual_gap=float(feedback[finger_index]+k.hand.gap_offset_m);check=checker.slacks(actual_q,actual_gap,delta,details=True)
                if not check['passed']:result['actual_configuration_failures'].append(dict(step=step,stage=stage,check=check))
            record=dict(step=step,stage=stage,stage_time_s=stage_time,requested_q_rad=np.array(q).tolist(),requested_velocity_rad_s=(np.zeros(7) if qd is None else np.asarray(qd)).tolist(),actual_q_rad=actual_q.tolist(),joint_positions_native=env.robot.data['joint_positions'],joint_forces_native=env.robot.data['joint_force'],actual_gap_m=float(feedback[finger_index]+k.hand.gap_offset_m),tcp_world_m=position.tolist(),tracking_error_m=tracking,fixed_drift_m=float(drift),fixed_drift_link=drift_link,contacts=native['contacts'],cloth_time_s=native['cloth_time_s'],cloth_fixed_steps=native['cloth_fixed_steps']);trace.write(json.dumps(record,default=lambda value:value.tolist())+'\n');step+=1
            if step%100==0:trace.flush();print('replay',number,stage,step,'tracking mm',round(tracking*1000,4),'unexpected',len(result['unexpected_contacts']),flush=True)
            return native,position,rotation,actual_q
        for t in np.arange(args.dt,2+args.dt/2,args.dt):
            advance(k.start,saved['initial_gap_m']+(saved['opening_gap_m']-saved['initial_gap_m'])*min(1,t/2),'opening',float(t),gap_rate=(saved['opening_gap_m']-saved['initial_gap_m'])/2 if t<2 else 0.)
        result['stages'].append(dict(stage='opening',duration_s=2.))
        for t in np.r_[np.arange(args.dt,profile['duration'],args.dt),profile['duration']]:
            tau=t/profile['duration'];u=10*tau**3-15*tau**4+6*tau**5;qd=profile['curve'](u,1)*(30*tau**2-60*tau**3+30*tau**4)/profile['duration'];advance(profile['curve'](u),saved['opening_gap_m'],'reach',float(t),qd=qd)
        result['stages'].append(dict(stage='reach',saved_duration_s=profile['duration']))
        for t in np.arange(args.dt,args.settle+args.dt/2,args.dt):native,position,rotation,actual_q=advance(np.array(saved['q_rad'][-1]),saved['opening_gap_m'],'settle',float(t))
        trace.close();target=np.array(candidate['grasp_center_unity_world_m'])+delta;result['endpoint_position_error_m']=float(np.linalg.norm(position-target));result['endpoint_orientation_error_deg']=float(np.rad2deg(Rotation.from_matrix(rotation@np.array(candidate['rotation_matrix']).T).magnitude()));result['maximum_tracking_error_m']=maximum_tracking;result['maximum_fixed_drift_m']=maximum_fixed_drift
        (out/'final-robot-data.json').write_text(json.dumps(env.robot.data,default=lambda value:value.tolist()));(out/'final-native-frames.json').write_text(json.dumps(native))
        attr.request_particles();env._env.step(simulate=False);result['final_cloth_error_m']=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max());result['final_clock']=native['cloth_time_s'];result['final_steps']=native['cloth_fixed_steps'];result['steps']=step;result['executed']=True
        result['passed']=bool(result['endpoint_position_error_m']<=1e-4 and result['endpoint_orientation_error_deg']<=.5 and not result['unexpected_contacts'] and not result['actual_configuration_failures'] and maximum_fixed_drift<2e-5 and result['final_cloth_error_m']<5e-6 and result['final_steps']-result['initial_steps']==step)
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        for name,direction,distance in [('endpoint',np.array([-.6,.3,-1.]),.65),('overview',np.array([-.5,.2,-1.]),2.4)]:
            result['images'].append(photograph(env,out/f'{name}.png',target,direction,np.array([0.,1.,0.]),distance,45))
    result['elapsed_wall_s']=time.monotonic()-began;(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('REPLAY RESULT',number,result['passed'],'endpoint mm',result['endpoint_position_error_m']*1000,flush=True);return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',type=Path,default=ROOT/'artifacts/compound-reaching/registration/candidate');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/compound-reaching/registered-replay');ap.add_argument('--numbers',type=int,nargs='+',default=[10,38,50]);ap.add_argument('--port',type=int,default=16020);ap.add_argument('--dt',type=float,default=.02);ap.add_argument('--settle',type=float,default=2.);ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--player',default='scripts/run_registered_reaching_player.sh');ap.add_argument('--hold-stiffness',type=float,default=1e7);ap.add_argument('--hold-damping',type=float,default=9e4);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    args.output=args.output.resolve();args.registration=args.registration.resolve()
    args.ignore_fixed_chassis_floor=os.environ.get('SOCK_IGNORE_FIXED_CHASSIS_FLOOR','0')=='1'
    if not args.prepare_only:
        native_gate=json.loads((args.registration/'native-validation/summary.json').read_text());payload_hash=hashlib.sha256((args.registration/'payload.json').read_bytes()).hexdigest()
        if not native_gate['passed'] or native_gate['payload_sha256']!=payload_hash:raise ValueError('Native geometry gate is missing or belongs to another payload')
        audit=json.loads((args.registration/'source-change-audit.json').read_text())
        if not audit['passed'] or audit['payload_sha256']!=payload_hash:raise ValueError('Source shape audit is missing or stale')
        route_gate=json.loads((args.registration/'route-recheck/summary.json').read_text())
        manifest_hash=hashlib.sha256((args.registration/'manifest.json').read_bytes()).hexdigest()
        if not route_gate['offline_passed'] or any(p['geometry_manifest_sha256']!=manifest_hash for p in route_gate['routes']):raise ValueError('Saved route validation is missing or belongs to another geometry')
    results=[]
    for path in sorted((ROOT/'artifacts/apron-reaching/routes').glob('*-trajectory.json')):
        if json.loads(path.read_text())['number'] in args.numbers:
            result=replay(path,args);results.append(result)
            if not result['passed']:break
    summary=dict(passed=len(results)==len(args.numbers) and all(p['passed'] for p in results),requested_numbers=args.numbers,routes=results,reference_cloth_held=True,dynamic_grasp_validated=False);name='summary.json' if len(args.numbers)==3 else 'summary-'+'-'.join(map(str,args.numbers))+'.json';(args.output/name).write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
