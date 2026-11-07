"""Re-render immutable measured teaching states with revised lighting."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .head_cameras import capture_head_cameras
from .reaching_collection import sha256


def render_recorded_episode(env,mapping,source,destination,expected_cloth,frames=None):
    source=Path(source).resolve();out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'metadata.json').exists():raise ValueError('Completed render must not be overwritten')
    original=json.loads((source/'metadata.json').read_text())
    if not original.get('raw_validation_passed') or not original.get('cloth_frozen'):raise ValueError('Validated frozen-cloth source required')
    angle=np.loadtxt(source/'angle.csv',delimiter=',',ndmin=2)
    frame_rows=list(csv.DictReader((source/'frames.csv').open()))
    source_poses=[json.loads(line) for line in (source/'camera_poses.jsonl').read_text().splitlines()]
    full=np.array(json.loads((source/'native-telemetry-sample.json').read_text())['joint_positions'])
    indices=list(range(len(angle))) if frames is None else list(frames)
    if not indices or len(set(indices))!=len(indices) or any(i<0 or i>=len(angle) for i in indices):
        raise ValueError('Render frames must be nonempty, unique source frame indices')
    for name in ('camera_right','camera_left','camera_overview','qa/visible_sock'):(out/name).mkdir(parents=True,exist_ok=True)
    report=[];pose_path=out/'camera_poses.jsonl'
    if pose_path.exists():
        for line in pose_path.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:break
            if len(report)>=len(indices) or row['frame']!=indices[len(report)]:break
            index=row['frame'];valid=True
            for name in ('camera_right','camera_left','camera_overview','qa/visible_sock'):
                try:
                    with Image.open(out/name/f'{index}.png') as image:image.verify()
                except (OSError,ValueError):valid=False;break
            if not valid:break
            report.append(row)
        pose_path.rename(out/f'camera_poses-interrupted-{time.time_ns()}.jsonl')
    # This is photometric re-rendering of already measured poses, not a new
    # dynamics rollout. Prevent contact-margin resolution from moving a saved
    # state during the articulation-to-render synchronization microstep.
    env.robot.EnabledAllCollider(False);env.robot.SetJointUseGravity(False);env._env.step(simulate=False)
    started=time.monotonic();angular=list(mapping.revolute_action_indices);linear=[i for i in range(18) if i not in angular]
    def clock():
        env.sock_cloth._send_data('GetSockCompoundReplay',env.robot.id);env._env.step(simulate=False)
        return int(env.sock_cloth.data['sock_compound_replay']['cloth_fixed_steps'])
    remaining=indices[len(report):]
    with pose_path.open('w') as poses:
        for row in report:poses.write(json.dumps(row)+'\n')
        for index in remaining:
            target=angle[index];command=mapping.merge_for_simulator(target,full)
            command[mapping.fixed_simulator_indices]=full[mapping.fixed_simulator_indices]
            env.robot.SetJointPositionDirectly(command.tolist());env.robot.SetJointPosition(command.tolist())
            env.robot.SetJointVelocity(np.zeros(len(command)).tolist())
            env.sock_cloth._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=np.zeros(len(command)).tolist())))
            # Unity propagates articulation coordinates to visual transforms on
            # a physics sync. These microsteps do not replace the source dynamics.
            env._env.Simulate(.000001,1);env._env.step(simulate=False)
            actual=mapping.from_simulator(env.robot.data['joint_positions']);error=abs(actual-target)
            if max(error[angular])>1e-6 or max(error[linear])>1e-7:raise RuntimeError(f'Render/source joint mismatch: {index}: {error.max()}')
            env.sock_cloth.request_particles();env._env.step(simulate=False)
            cloth_error=float(np.linalg.norm(np.array(env.sock_cloth.particles())-expected_cloth,axis=1).max())
            if cloth_error>2e-5:raise RuntimeError('Rendering changed frozen cloth geometry')
            before=clock();env.recording_camera.GetRGB(1280,960,original['camera_fov'])
            images,cameras=capture_head_cameras(env,original['camera_fov'])
            overview=env._decode(env.recording_camera.data['rgb'],'RGB')
            for side,rgb in images.items():Image.fromarray(rgb).save(out/f'camera_{side}/{index}.png')
            Image.fromarray(overview).save(out/f'camera_overview/{index}.png')
            pose_errors={}
            for side in ('right','left'):
                p=cameras['cameras'][side];old=source_poses[index]['cameras'][side]
                distance=float(np.linalg.norm(np.array(p['position'])-old['position']))
                rotation=float(np.rad2deg((Rotation.from_quat(p['quaternion'])*Rotation.from_quat(old['quaternion']).inv()).magnitude()))
                if distance>3e-5 or rotation>.003:raise RuntimeError(f'Head camera pose changed: {index}/{side}: {distance}/{rotation}')
                pose_errors[side]=dict(position_m=distance,rotation_deg=rotation)
            # Independent ID rendering for visibility QA, not training masks.
            from pyrcareworld.utils.rfuniverse_utility import EncodeIDAsColor
            env.sock_cloth._send_data('CaptureSockLearningVisibility',env.head_cameras['right'].id,1280,960,float(original['camera_fov']))
            env._env.step(simulate=False)
            ids=env._decode(env.sock_cloth.data['sock_learning_visibility'],'RGB')
            color=np.asarray(EncodeIDAsColor(env.sock_cloth.id)[:3],dtype='int16')
            truth=(abs(ids.astype('int16')-color).max(axis=2)<=1)
            if truth.sum()<100:raise RuntimeError('Independent visible-cloth reference is empty')
            if index==indices[0]:Image.fromarray(ids).save(out/'qa/visibility-ids-first.png')
            Image.fromarray(truth.astype('uint8')*255).save(out/f'qa/visible_sock/{index}.png')
            after=clock()
            if before!=after:raise RuntimeError('Camera capture advanced physics')
            row=dict(frame=index,step=int(frame_rows[index]['step']),source_time_s=float(frame_rows[index]['time_s']),
                rendering_contacts_disabled=True,
                render_sync_step=before,joint_error_rad=float(error[angular].max()),joint_error_m=float(error[linear].max()),
                cloth_error_m=cloth_error,source_camera_pose_errors=pose_errors,**cameras)
            poses.write(json.dumps(row)+'\n');poses.flush();report.append(row)
            if len(report)%25==0:
                (out/'render-progress.json').write_text(json.dumps(dict(processed=len(report),total=len(indices),elapsed_s=time.monotonic()-started))+'\n')
                print('LIT RENDER',source.name,len(report),'/',len(indices),flush=True)
    source_hashes={}
    if frames is None:
        for name in ('angle.csv','torque.csv','external_torque.csv','frames.csv','telemetry_50hz.jsonl','native-telemetry-sample.json'):
            shutil.copy2(source/name,out/name);source_hashes[name]=sha256(source/name)
        confirmation=out/'confirmation';confirmation.mkdir()
        (confirmation/'previous_head_data').symlink_to(source,target_is_directory=True)
        if (source/'confirmation/camera_overview_original').exists():
            (confirmation/'camera_overview_original').symlink_to(source/'confirmation/camera_overview_original',target_is_directory=True)
    metadata=dict(original)
    metadata.update(schema='shareset-head-lighting-rerender-v3',acquisition_mode='render_recorded_measured_joint_states',
        source_episode=str(source),source_metadata_sha256=sha256(source/'metadata.json'),source_signal_sha256=source_hashes,
        lighting_profile=env.learning_lighting['profile'],lighting=env.learning_lighting,render_sync_step_s=.000001,
        source_frame_count=original['frames'],frames=len(indices),sample_only=frames is not None,
        raw_validation_passed=frames is None,render_state_alignment_passed=True,
        effort_scope=original['effort_scope']+'; immutable source recording, not measured during visual re-render',
        physics_steps_unchanged=None,source_physics_steps_unchanged=original.get('physics_steps_unchanged'),
        rendering_is_dynamics_replay=False,rendering_contact_policy='Kinematic pose placement only; source dynamics, contacts and efforts are retained unchanged',
        perception_passed=False,reaching_learning_ready=False,render_completed=True,
        visibility_reference_scope='Rendering QA only, not training masks',
        maximum_source_joint_error_rad=max(r['joint_error_rad'] for r in report),
        maximum_source_joint_error_m=max(r['joint_error_m'] for r in report),
        maximum_cloth_error_m=max(r['cloth_error_m'] for r in report))
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'render-validation.json').write_text(json.dumps(dict(passed=True,source_signal_sha256=source_hashes,
        frames=len(indices),source_indices=indices,elapsed_s=time.monotonic()-started),indent=2)+'\n')
    print('LIT RENDER PASSED',source.name,len(indices),flush=True)
    return metadata
