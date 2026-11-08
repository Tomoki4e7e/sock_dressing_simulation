#!/usr/bin/env python3
"""Restore one selected initial configuration in Unity without replaying motion."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,REGISTERED_MANIFEST,URDF,state_for
from sock_dressing_simulation.reaching_dataset import verify_dataset_inputs
from search_fixed_torso_starts import context
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.pose_patterns import read_patterns
from sock_dressing_simulation.joints import JointMap
from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM
from sock_dressing_simulation.arm_reaching import ARM,shape_vertices
from photograph_grasp_candidates import photograph


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');ap.add_argument('--index',type=int,default=5);ap.add_argument('--port',type=int,default=16680);ap.add_argument('--trajectory',type=Path);ap.add_argument('--initial-configuration',type=Path);ap.add_argument('--replay',action='store_true')
    ap.add_argument('--common-route',type=Path,help='Replay one immutable common-opening route and evaluate the existing A grasp rule')
    ap.add_argument('--state',type=Path,help='Explicit frozen cloth state for an initial-configuration capture')
    ap.add_argument('--collect-episode',type=Path,help='New synchronized RGB/effort episode (controlled closure routes only)')
    ap.add_argument('--head-cameras', action='store_true', help='Use the two physical See3CAM frames for the learning streams')
    ap.add_argument('--lighting-profile', choices=['baseline','normals_only','head_fill','balanced','balanced_shadow_off','bright_fill'])
    ap.add_argument('--lighting-probe',action='store_true')
    ap.add_argument('--render-episode',type=Path)
    ap.add_argument('--render-output',type=Path)
    ap.add_argument('--render-frames',nargs='+',type=int)
    ap.add_argument('--policy-model-dir',type=Path)
    ap.add_argument('--policy-checkpoint',type=Path)
    ap.add_argument('--policy-episode',type=Path)
    ap.add_argument('--policy-steps',type=int,default=260)
    ap.add_argument('--policy-device',choices=['auto','cpu','cuda'],default='auto')
    # Parse the optional shape before loading the selected configuration.
    a=ap.parse_args();a.output=a.output.resolve()
    if a.policy_model_dir and (not a.policy_checkpoint or not a.policy_episode or not a.head_cameras or not a.lighting_profile or not a.trajectory or a.replay or a.render_episode):
        ap.error('Policy video requires checkpoint, episode, head cameras, lighting and trajectory, without replay/render-episode')
    if (a.lighting_profile or a.lighting_probe or a.render_episode) and not a.head_cameras:
        ap.error('Lighting and recorded-state rendering require --head-cameras')
    if a.render_episode and (not a.lighting_profile or a.replay or a.collect_episode):
        ap.error('--render-episode requires --lighting-profile and cannot accompany replay/collection')
    if (a.render_output or a.render_frames is not None) and not a.render_episode:
        ap.error('--render-output/--render-frames require --render-episode')
    if a.common_route and (not a.state or not a.initial_configuration or a.trajectory or a.replay):ap.error('--common-route requires --state and --initial-configuration only')
    if a.state and (a.trajectory or not a.initial_configuration):ap.error('--state requires --initial-configuration and cannot accompany --trajectory')
    if a.replay and not a.trajectory:ap.error('--replay requires --trajectory')
    if a.collect_episode and not a.replay:ap.error('--collect-episode requires --replay')
    if a.trajectory and a.initial_configuration:ap.error('Choose either --trajectory or --initial-configuration')
    if a.trajectory:
        saved=json.loads(a.trajectory.read_text());selected=dict(number=saved['number'],right_q_rad=saved['start_q_rad'],right_q_deg=np.rad2deg(saved['start_q_rad']).tolist(),left_q_rad=saved['left_q_rad'],left_q_deg=np.rad2deg(saved['left_q_rad']).tolist(),translation_m=saved['translation_m'],right_gap_m=saved['opening_gap_m'],left_gap_m=saved['left_gap_m'],trajectory=str(a.trajectory),left_foot_floor_allowance_m=saved.get('left_foot_floor_allowance_m',0.))
        verify_dataset_inputs(saved,ROOT)
        if saved.get('moving_side')=='left':
            selected.update(right_q_rad=saved['right_q_rad'],right_q_deg=np.rad2deg(saved['right_q_rad']).tolist(),left_q_rad=saved['start_q_rad'],left_q_deg=np.rad2deg(saved['start_q_rad']).tolist(),right_gap_m=saved['right_gap_m'],left_gap_m=saved.get('initial_opening_gap_m',saved['opening_gap_m']))
        elif 'initial_opening_gap_m' in saved:selected['right_gap_m']=saved['initial_opening_gap_m']
    elif a.initial_configuration:
        import yaml
        selected=yaml.safe_load(a.initial_configuration.read_text())
    else:selected=json.loads((a.output/'initial-configurations.json').read_text())['candidates'][a.index]
    if a.common_route:
        from replay_common_opening_motion import verify_source
        common=verify_source(a.common_route,a.initial_configuration,a.state)
        selected['number']=common['condition']['number']
    number=selected['number'];out=a.output/('native-route-initial' if a.trajectory or a.common_route else 'native-initial');out.mkdir(parents=True,exist_ok=True)
    # The dependency hardcodes writable lock files in ~/.rfuniverse. Keep its
    # configuration read-only and redirect only this process's lock files.
    lock_dir=out/'runtime-locks';lock_dir.mkdir(exist_ok=True)
    locker_path=ROOT/'.deps/RCareWorld/pyrcareworld/pyrcareworld/utils/locker.py'
    spec=importlib.util.spec_from_file_location('pyrcareworld.utils.locker',locker_path);locker=importlib.util.module_from_spec(spec);spec.loader.exec_module(locker)
    def local_lock_init(self,name):
        self.lck_path=str(lock_dir/(name+'.lck'));Path(self.lck_path).touch(exist_ok=True);self.fp=None
    locker.Locker.__init__=local_lock_init;sys.modules['pyrcareworld.utils.locker']=locker
    state=state_for(saved['geometry_record']) if a.trajectory else json.loads((a.state if a.state else ROOT/f'artifacts/bulged-cuff/pose-{number:02d}/state.json').read_text());scene_number=common['condition']['scene_number'] if a.common_route else state.get('scene_number',number);state['scene_number']=scene_number;delta=np.array(selected['translation_m']);k,checker,dl,du,contract,source_scene=context(number,state,REGISTERED_MANIFEST,left_foot_floor_allowance=selected.get('left_foot_floor_allowance_m',0.));checker.park_left(selected['left_q_rad'],selected['left_gap_m'])
    if any(abs(k.q0[n]-v)>1e-8 for n,v in selected.get('torso_joints_rad',{}).items()):raise ValueError('Initial configuration asks to change the fixed torso')
    offline=checker.slacks(np.array(selected['right_q_rad']),selected['right_gap_m'],delta,details=True)
    if not offline['passed']:raise ValueError('Initial configuration failed registered-geometry validation')
    config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number');settings['toe_contact']=None;settings['gravity_drape']=None
    config['rcareworld'].update(executable='scripts/run_registered_reaching_player.sh',graphics=True,port=a.port,log_level=0);config['assets']['output_dir']=str(out/'assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario);mapping=JointMap.from_config(config)
    rest_path=ROOT/f'artifacts/bulged-cuff/pose-{scene_number:02d}/rest.obj'
    if a.common_route:
        from prepare_bulged_cuffs import write_obj
        rest_path=out/'state-rest.obj';write_obj(rest_path,state['source']['rest_positions'],np.array(state['triangles']).reshape(-1,3),native_import=True)
    pattern=next(p for p in read_patterns(Path(workbook)) if p.number==number)
    result=dict(selected_index=a.index,initial_configuration=selected,offline_registered_check=offline,initial_only=True,motion_replayed=False,passed=False)
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),rest_path,initial_joints=scenario.initial_joints);env.apply_scenario(scenario);attr=env.sock_cloth
        attr.set_pattern_held(False);attr.set_pattern_foot_pose(pattern.pitch_deg,settings['signs']['roll']*pattern.roll_deg);env._env.step(2,simulate=False);attr.request_preinsert_source();env._env.step(2,simulate=False)
        native_rest=np.array(attr.data['sock_preinsert_source']['rest_positions']);rest=np.array(state['source']['rest_positions']);dist,order=cKDTree(rest).query(native_rest)
        if len(native_rest)!=len(rest) or len(set(order.tolist()))!=len(rest) or dist.max()>5e-6:raise ValueError('Native refined cloth topology differs')
        attr.initialize_preinsert(np.array(state['particles_world'])[order],state['opening_normal']);env._env.step(2,simulate=False);attr.set_pattern_held(True);env._env.step(2,simulate=False)
        attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False);before=copy.deepcopy(attr.data['sock_arm_reaching_scene']);native_reference=np.array(env.robot.data['joint_positions'],dtype=float)
        camera=env.recording_camera
        def snapshot_camera(name):
            if a.common_route:return {'image_capture_skipped':True}
            camera.GetRGB(1280,960,float(config['camera']['fov']));env._env.step(2,simulate=False)
            Image.fromarray(env._decode(camera.data['rgb'],'RGB')).save(out/name)
            return {key:copy.deepcopy(camera.data[key]) for key in ('position','rotation','quaternion','rotation_quaternion','local_position','local_rotation') if key in camera.data}
        result['camera_before']=snapshot_camera('fixed-camera-before.png')
        attr._send_data('TranslateSockReplayScene',env.robot.id,*map(float,delta));env._env.step(2,simulate=False)
        expected_cloth=(np.array(state['particles_world'])+delta)[order];attr.initialize_preinsert(expected_cloth,state['opening_normal']);env._env.step(2,simulate=False);attr.set_pattern_held(True);env._env.step(2,simulate=False)
        action=np.array(scenario.initial_joints);indices=mapping.index()
        for names,values in [(ARM,selected['right_q_rad']),(LEFT_ARM,selected['left_q_rad'])]:
            for n,v in zip(names,values):action[indices[n]]=v
        for side in ('left','right'):action[indices[side+'_gripper/finger_joint']]=k.hand.joint_for_gap(selected[side+'_gap_m'])
        command=mapping.merge_for_simulator(mapping.bound(action),native_reference)
        # Retain the actual baseline torso/head/base values bit-for-bit.
        command[mapping.fixed_simulator_indices]=native_reference[mapping.fixed_simulator_indices]
        env.robot.SetJointPositionDirectly(command.tolist());env._env.step(4,simulate=False)
        attr._send_data('InstallSockRigidSkinContact',env.robot.id,0);env._env.step(2,simulate=False)
        payload=json.loads((REGISTERED_MANIFEST.parent/'payload.json').read_text());attr._send_data('InstallSockCompoundGeometry',env.robot.id,json.dumps(payload));env._env.step(3,simulate=False)
        attr._send_data('ConfigureSockReplayDrives',env.robot.id,1e7,9e4);env._env.step(simulate=False)
        # Unity updates articulation world transforms on a physics step after
        # assigning reduced joint coordinates. This settles the initial state;
        # no reaching trajectory is replayed.
        env.robot.SetJointPosition(command.tolist());env._env.Simulate(.0001,1);env._env.step(simulate=False)
        for _ in range(50):
            env.robot.SetJointPosition(command.tolist());attr._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=np.zeros(len(command)).tolist())));env._env.Simulate(.02,1);env._env.step(simulate=False)
        result['initialization_physics_steps']=51;result['initialization_duration_s']=1.0001
        attr._send_data('GetSockArmReachingScene',env.robot.id);attr.request_particles();env._env.step(2,simulate=False);after=copy.deepcopy(attr.data['sock_arm_reaching_scene']);after['robot_data']=json.loads(json.dumps(env.robot.data,default=lambda v:v.tolist()))
        write(out/'native-scene.json',after);write(out/'before-scene.json',before)
        before_frames={r['name']:r for r in before['links']};fixed=[];moving=[]
        desired_frames=k.fk(np.array(selected['right_q_rad']),selected['right_gap_m'])
        feedback=mapping.from_simulator(env.robot.data['joint_positions']);actual_right=np.array([feedback[indices[n]] for n in ARM]);actual_left=np.array([feedback[indices[n]] for n in LEFT_ARM]);actual_right_gap=float(feedback[indices['right_gripper/finger_joint']]+k.hand.gap_offset_m);actual_left_gap=float(feedback[indices['left_gripper/finger_joint']]+k.hand.gap_offset_m)
        checker.park_left(actual_left,actual_left_gap);frames=k.fk(actual_right,actual_right_gap);actual_errors=[]
        for row in after['links']:
            name=row['name']
            if not name.startswith(('left_arm/','right_arm/','left_gripper/','right_gripper/')):
                old=before_frames[name];fixed.append(dict(link=name,position_error_m=float(np.linalg.norm(np.array(row['position'])-old['position'])),orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(Rotation.from_quat(row['quaternion']).as_matrix()@Rotation.from_quat(old['quaternion']).as_matrix().T).magnitude()))))
            if name in frames:
                expected=k.A@desired_frames[name][:3,3]+k.b;moving.append(dict(link=name,position_error_m=float(np.linalg.norm(np.array(row['position'])-expected))))
                actual_errors.append(float(np.linalg.norm(np.array(row['position'])-(k.A@frames[name][:3,3]+k.b))))
        result['fixed_frames']=fixed;result['maximum_fixed_frame_position_error_m']=max(r['position_error_m'] for r in fixed);result['maximum_fixed_frame_orientation_error_deg']=max(r['orientation_error_deg'] for r in fixed)
        result['maximum_planned_link_position_error_m']=max(r['position_error_m'] for r in moving)
        result['maximum_actual_fk_link_position_error_m']=max(actual_errors);result['actual_right_q_rad']=actual_right.tolist();result['actual_left_q_rad']=actual_left.tolist();result['actual_right_gap_m']=actual_right_gap;result['actual_left_gap_m']=actual_left_gap
        result['actual_initial_check']=checker.slacks(actual_right,actual_right_gap,delta,details=True)
        result['human_error_m']=float(np.linalg.norm(np.array(after['human_skin']['vertices'])-np.array(source_scene['human_skin']['vertices'])-delta,axis=1).max())
        result['cloth_error_m']=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max())
        result['camera_after']=snapshot_camera('fixed-camera-after.png');result['camera_transform_unchanged']=bool(result['camera_before'] and result['camera_before']==result['camera_after'])
        result['passed']=bool(result['maximum_fixed_frame_position_error_m']<2e-5 and result['maximum_fixed_frame_orientation_error_deg']<.002 and result['maximum_planned_link_position_error_m']<.002 and result['maximum_actual_fk_link_position_error_m']<3e-5 and result['actual_initial_check']['passed'] and max(result['human_error_m'],result['cloth_error_m'])<2e-5 and result['camera_transform_unchanged'])
        write(out/'result.json',result)
        if a.head_cameras:
            from sock_dressing_simulation.head_cameras import create_head_cameras, preview_head_cameras
            create_head_cameras(env)
            if a.lighting_probe:
                from sock_dressing_simulation.learning_lighting import probe_lighting
                result['lighting_probe']=probe_lighting(env,out/'lighting-probe',float(config['camera']['fov']))
            if a.lighting_profile:
                from sock_dressing_simulation.learning_lighting import apply_lighting
                result['learning_lighting']=apply_lighting(env,a.lighting_profile)
            result['head_cameras'] = preview_head_cameras(env, out, float(config['camera']['fov']))
            write(out/'result.json', result)
            if a.render_episode:
                if not result['passed'] or not a.lighting_profile:raise ValueError('Validated initial scene and a lighting profile required')
                from sock_dressing_simulation.lit_recording import render_recorded_episode
                result['lit_render']=render_recorded_episode(env,mapping,a.render_episode,a.render_output or (a.output/'rendered'),expected_cloth,a.render_frames)
            if a.policy_model_dir:
                if not result['passed']:raise ValueError('Validated initial scene required for policy inference')
                from sock_dressing_simulation.reaching_policy_video import run_policy_video
                result['policy_video']=run_policy_video(env,mapping,a.policy_episode.resolve(),a.policy_model_dir.resolve(),
                    a.policy_checkpoint.resolve(),a.output/'policy-video',a.policy_steps,a.policy_device)
        if a.common_route:
            if not result['passed']:raise ValueError('Native initial configuration failed; common route not replayed')
            from replay_common_opening_motion import replay
            result['motion']=replay(env,mapping,action,native_reference,k,checker,common,a.common_route.resolve(),delta,expected_cloth,out)
            result['initial_only']=False;result['motion_replayed']=True
            result['camera_validation_performed']=False
            write(out/'result.json',result)
            return
        if a.replay:
            if not result['passed']:raise ValueError('Native initial configuration failed; trajectory will not be replayed')
            if saved.get('close_after_reach'):
                from replay_controlled_photo_grasp import replay
            else:
                from replay_fixed_torso_motion import replay
            if a.collect_episode:
                if not saved.get('close_after_reach'):raise ValueError('Collection requires a controlled closure route')
                from sock_dressing_simulation.reaching_collection import ReachingCollector
                with ReachingCollector(a.collect_episode,mapping,saved,a.trajectory.resolve(),ROOT,config,head_cameras=a.head_cameras) as collector:
                    result['motion']=replay(env,mapping,action,native_reference,k,checker,saved,a.trajectory.resolve(),delta,expected_cloth,out,collector=collector)
                    collector.finish(result['motion'])
            else:
                result['motion']=replay(env,mapping,action,native_reference,k,checker,saved,a.trajectory.resolve(),delta,expected_cloth,out)
            result['initial_only']=False;result['motion_replayed']=True;result['camera_after_motion']=snapshot_camera('fixed-camera-endpoint.png')
            result['camera_transform_unchanged_during_motion']=result['camera_before']==result['camera_after_motion']
            write(out/'result.json',result)
        # Inspection view is taken only after the fixed-camera comparison.
        result['inspection_image']=photograph(env,out/'overview.png',np.array([-.144,.8,.35]),np.array([-.1,.2,-1.]),np.array([0.,1.,0.]),3.4,48)
        result['oblique_image']=photograph(env,out/'oblique.png',np.array([-.144,.8,.35]),np.array([-1.,.25,-.6]),np.array([0.,1.,0.]),3.1,48)
        if a.trajectory:
            target=np.array(saved['geometry_record']['candidate']['grasp_center_unity_world_m'])+delta
            result['close_image']=photograph(env,out/'endpoint-close.png',target,np.array([-.7,.3,-1.]),np.array([0.,1.,0.]),.65,45)
        if a.state:
            # Preserve the normal full-scene initial photograph above; use a
            # separate unlit cloth view so its tail and foot fit are visible.
            target=(np.array(state['source']['ankle'])+np.array(state['source']['toe']))*.5+delta+np.array([0.,-.015,.03])
            attr._send_data('SetSockInspectionMaterial',True,env.robot.id);env._env.step(simulate=False)
            result['sock_inspection_image']=photograph(env,out/'sock-inspection.png',target,np.array([-1.,.15,-.3]),np.array([0.,1.,0.]),.65,45)
            result['sock_inspection_front_image']=photograph(env,out/'sock-inspection-front.png',target,np.array([-.2,.1,1.]),np.array([0.,1.,0.]),.65,45)
            attr._send_data('SetSockInspectionMaterial',False,env.robot.id);env._env.step(simulate=False)
        result['inspection_camera_repositioned_after_comparison']=True
    write(out/'result.json',result);print('NATIVE INITIAL',result['passed'],'fixed drift',result['maximum_fixed_frame_position_error_m'],'planned link error',result['maximum_planned_link_position_error_m'],'camera unchanged',result['camera_transform_unchanged'],flush=True)


if __name__=='__main__':main()
