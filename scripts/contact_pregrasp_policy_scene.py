#!/usr/bin/env python3
"""Reuse the validated contact-scene initializer, replacing route replay with policy control.

The contact route supplies initialization/provenance and geometric goal checks
only. No trajectory points or teacher commands enter the online policy loop.
"""
import argparse,copy,gzip,hashlib,json,os,select,subprocess,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.head_cameras import capture_head_cameras
from sock_dressing_simulation.pregrasp_gap_measurement import measure
from sock_dressing_simulation.contact_pregrasp import opening_edges
from sock_dressing_simulation.continuous_pregrasp import SweptPolicy
from verify_contact_pregrasp_continuous import make_scene
from native_pregrasp_geometry import native_update,native_terminal
from photograph_grasp_candidates import photograph


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


class Video:
    def __init__(self,path,size):
        self.path=path;self.frames=0
        self.p=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{size[0]}x{size[1]}','-r','5','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    def append(self,image):self.p.stdin.write(image.astype('uint8').tobytes());self.frames+=1
    def close(self):
        self.p.stdin.close()
        if self.p.wait()!=0:raise RuntimeError('Video encoding failed')


def pose_camera(camera,target,offset):
    forward=-offset/np.linalg.norm(offset);right=np.cross(np.array([0.,1.,0.]),forward);right/=np.linalg.norm(right);up=np.cross(forward,right)
    camera.SetPosition((target+offset).tolist());camera.SetRotationQuaternion(Rotation.from_matrix(np.column_stack([right,up,forward])).as_quat().tolist())


def execute(a,env,mapping,action,reference,k,checker,saved,source_path,delta,expected_cloth,out,**unused):
    dest=a.output/'policy';dest.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.episode/'metadata.json').read_text());state=json.loads((ROOT/meta['condition']['state']).read_text())
    steps=(meta['frames']+1)//2-1
    if a.steps is not None:steps=min(steps,a.steps)
    scene,_,_=make_scene(saved['condition'],Path(source_path).parent.parent,SweptPolicy())
    attr=env.sock_cloth;worker_log=(dest/'worker.log').open('w')
    worker=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'scripts/contact_pregrasp_policy_worker.py'),'--run',str(a.run),'--checkpoint',str(a.checkpoint),'--episode',str(a.episode),'--flow-ode-steps',str(a.flow_ode_steps),'--flow-execute-steps',str(a.flow_execute_steps),'--flow-seed',str(a.flow_seed)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=worker_log,text=True,bufsize=1)
    def response():
        if not select.select([worker.stdout],[],[],180)[0]:raise TimeoutError('GPU policy response timeout')
        line=worker.stdout.readline()
        if not line:raise RuntimeError('GPU worker exited; inspect worker.log')
        return json.loads(line)
    overview=Video(dest/'overview.mp4',(1280,960));head=Video(dest/'head-input.mp4',(640,480))
    trace=[];efforts=[];native=None;info={};failure=None;start=time.time();contacts=[];max_drift=0.;geometry={};dropped_contact_records=0
    fixed=np.array(env.robot.data['joint_positions'])[mapping.fixed_simulator_indices].copy()
    env.robot.SetJointUseGravity(False);attr._send_data('BeginSockCompoundReplay',env.robot.id);env._env.step(simulate=False)
    initial_physics_step=int(attr.data['sock_compound_replay']['cloth_fixed_steps'])
    target_center=np.array(state['particles_world']).mean(0)+delta
    pose_camera(env.recording_camera,target_center,np.array([-1.1,1.0,-1.1]));env._env.step(simulate=False)
    def capture_overview():
        env.recording_camera.GetRGB(1280,960,65.);env._env.step(simulate=False)
        return env._decode(env.recording_camera.data['rgb'],'RGB')
    def terminal_measurement():
        feedback=np.array(trace[-1]['feedback_angle']);idx=mapping.index();x=np.r_[feedback[[idx[f'{side}_arm/joint_{i}'] for side in ('left','right') for i in range(1,8)]],feedback[[idx['left_gripper/finger_joint'],idx['right_gripper/finger_joint']]]];x[-2:]+=k.hand.gap_offset_m
        native_update(scene,x,native['links']);contact_checks=native_terminal(scene,saved)
        attr.request_particles();env._env.step(simulate=False);particles=np.array(attr.particles());faces=np.array(state['triangles'],dtype=int).reshape(-1,3);material=np.array(state['source']['rest_positions']);depth=material[:,2]-material[:,2].min()
        # Native particle order is matched during initialization, while saved
        # topology uses source order. Undo that exact nearest-rest permutation.
        from scipy.spatial import cKDTree
        attr.request_preinsert_source();env._env.step(simulate=False);rest=np.array(attr.data['sock_preinsert_source']['rest_positions']);_,order=cKDTree(material).query(rest);assert len(set(order.tolist()))==len(order)
        source_particles=np.empty_like(particles);source_particles[order]=particles
        edges=opening_edges(faces,depth);geometry={}
        for side in ('left','right'):
            p=scene.positions[side];R=scene.rotations[side];fingers=[];world_fingers={}
            for suffix in ('left_finger_link','right_finger_link'):
                name=side+'_gripper/'+suffix;transform=scene.base.group_transforms[name];world=scene.base.group_vertices[name]@transform.getRotation().T+transform.getTranslation();fingers.append((world-p)@R);world_fingers[suffix]=world.tolist()
            result=measure((source_particles-p)@R,faces,depth,fingers,(source_particles[edges]-p)@R)
            result.update(contact_geometry_checks=contact_checks[side],tcp_world_m=p.tolist(),rotation=R.tolist(),actual_finger_joint_m=float(feedback[idx[side+'_gripper/finger_joint']]),measured_finger_world_m=world_fingers)
            geometry[side]=result
            for label,offset in [('gap-side',R[:,2]+R[:,0]*.15),('gap-opposite',-R[:,2]+R[:,0]*.15)]:photograph(env,dest/f'{side}-{label}.png',p,offset,R[:,1],.20,40)
        write(dest/'terminal-cloth-between-fingers.json',geometry)
        with gzip.open(dest/'terminal-geometry.json.gz','wt') as f:json.dump(dict(particles_world=source_particles.tolist(),faces=faces.tolist(),material_depth_m=depth.tolist(),native_links=native['links'],measured_angles=feedback.tolist()),f)
        return geometry
    try:
        info=response();assert info['ready'] and info['device']=='cuda' and all(v.startswith('cuda') for v in info['model_devices'].values())
        write(dest/'device.json',info)
        initial=capture_overview();overview.append(initial);Image.fromarray(initial).save(dest/'overview-initial.png')
        env.robot.GetJointInverseDynamicsForce();env._env.step(simulate=False)
        signals=env.robot_signals();assert signals['torque_available'];efforts.append(np.array(signals['torque']))
        for step in range(steps):
            q=np.array(signals['angle']);history=np.array(efforts[-29:]);window=min(len(history),29);window-=1-window%2
            filtered=savgol_filter(history[-window:],window,2,axis=0)[-1] if window>=3 else history[-1]
            views,poses=capture_head_cameras(env,92.,640,480);head.append(views['right']);rgb=dest/'current-rgb.png';Image.fromarray(views['right']).save(rgb)
            worker.stdin.write(json.dumps(dict(index=step,rgb=str(rgb),angle=q.tolist(),torque=filtered.tolist()))+'\n');worker.stdin.flush();prediction=response()
            proposed=np.array(prediction['action']);command=mapping.bound(proposed,q)
            full=np.array(env.robot.data['joint_positions']);target=mapping.merge_for_simulator(command,full);target[mapping.fixed_simulator_indices]=fixed
            for block in range(10//a.physics_batch):
                tick=(block+1)*a.physics_batch-1
                env.robot.SetJointPosition(target.tolist());attr._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=np.zeros(len(target)).tolist())))
                env._env.Simulate(.02,a.physics_batch)
                if tick in (4,9):env.robot.GetJointInverseDynamicsForce()
                attr._send_data('GetSockCompoundReplay',env.robot.id);env._env.step(simulate=False)
                native=json.loads(json.dumps(attr.data['sock_compound_replay'],default=lambda x:x.tolist()))
                if not native['cloth_held']:raise RuntimeError('Frozen cloth was released')
                # Overflow concerns only the optional contact-event recorder,
                # not the physics state, images or measured terminal cloth.
                dropped_contact_records+=int(native['dropped'])
                contacts.extend(dict(step=step,tick=tick,**c) for c in native['contacts'] if c['separation_m'] < -1e-5 and any(c.get(key,'').startswith(('left_arm/','right_arm/','left_gripper/','right_gripper/')) for key in ('a_link','b_link')))
                if tick in (4,9):
                    signals=env.robot_signals()
                    if not signals['torque_available']:raise RuntimeError('Measured drive effort unavailable')
                    efforts.append(np.array(signals['torque']))
            actual=np.array(signals['angle']);assert np.isfinite(actual).all()
            image=capture_overview();overview.append(image)
            if step in (steps//2,steps-1):Image.fromarray(image).save(dest/('overview-final.png' if step==steps-1 else 'overview-middle.png'))
            trace.append(dict(step=step,time_s=(step+1)*.2,observed_angle=q.tolist(),filtered_observed_torque=filtered.tolist(),prediction=prediction,executed_command=command.tolist(),feedback_angle=actual.tolist(),feedback_torque=signals['torque'].tolist(),maximum_command_clipping=float(np.max(abs(command-proposed)))))
            if step%25==0 or step==steps-1:
                attr.request_particles();env._env.step(simulate=False);drift=float(np.linalg.norm(attr.particles()-expected_cloth,axis=1).max());max_drift=max(max_drift,drift)
                if drift>2e-5:raise RuntimeError('Cloth moved outside frozen-scene tolerance')
                write(dest/'progress.json',dict(step=step+1,steps=steps,device='cuda',elapsed_s=time.time()-start));print('LIVE PREGRASP',meta['condition']['id'],step+1,'/',steps,flush=True)
        assert int(native['cloth_fixed_steps'])-initial_physics_step==steps*10, 'Physics step count mismatch'
        geometry=terminal_measurement()
    except BaseException as error:
        failure=repr(error)
        if trace and native is not None:
            try:geometry=terminal_measurement()
            except Exception as terminal_error:write(dest/'terminal-measurement-failure.json',dict(error=repr(terminal_error)))
        raise
    finally:
        if worker.poll() is None:
            try:worker.stdin.write('{"quit":true}\n');worker.stdin.flush();worker.stdin.close();worker.wait(timeout=15)
            except (BrokenPipeError,subprocess.TimeoutExpired):worker.terminate();worker.wait(timeout=15)
        worker_log.close();overview.close();head.close()
        with (dest/'motion-trace.jsonl').open('w') as f:
            for r in trace:f.write(json.dumps(r)+'\n')
        with gzip.open(dest/'native-penetrating-contacts.json.gz','wt') as f:json.dump(contacts,f)
        report=dict(completed=failure is None and len(trace)==steps,passed=bool(failure is None and len(geometry)==2 and all(g['contact_geometry_checks']['passed'] for g in geometry.values())),failure=failure,id=meta['condition']['id'],steps=len(trace),requested_steps=steps,simulated_duration_s=len(trace)*.2,overview_frames=overview.frames,head_frames=head.frames,overview_video=str(overview.path),head_video=str(head.path),device=info,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),live_observations=True,teacher_trajectory_used=False,command_blending=False,robot_joint_gravity=False,physics_step_s=.02,actual_physics_steps=None if native is None else int(native['cloth_fixed_steps'])-initial_physics_step,physics_batch_steps=a.physics_batch,contact_collection_interval_s=.02*a.physics_batch,torque_input='Causal endpoint Savitzky-Golay smoothing, up to 29 past measured samples at 10 Hz; warmup shorter odd windows; no synthesized gravity. Training used centered offline smoothing.',cloth_frozen=True,cloth_maximum_drift_m=max_drift,native_penetration_event_count=len(contacts),geometry=geometry,elapsed_s=time.time()-start,scope='Closed-loop GPU policy and native motor execution with frozen cloth/human; geometric jaw occupancy, not closure/retention validation')
        report['native_contact_records_dropped']=dropped_contact_records
        report['native_contact_recording_complete']=dropped_contact_records==0
        write(dest/'report.json',report)
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--episode',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--port',type=int,required=True);ap.add_argument('--steps',type=int);ap.add_argument('--physics-batch',type=int,choices=[1,5],default=1)
    ap.add_argument('--flow-ode-steps',type=int,default=8);ap.add_argument('--flow-execute-steps',type=int,choices=[1,2,4],default=1);ap.add_argument('--flow-seed',type=int,default=0);a=ap.parse_args()
    for key in ('run','checkpoint','episode','output'):setattr(a,key,getattr(a,key).resolve())
    meta=json.loads((a.episode/'metadata.json').read_text());source=ROOT/meta['trajectory'];initial=ROOT/meta['shared_initial_configuration'];state=ROOT/meta['condition']['state']
    import capture_fixed_torso_initial as capture
    import replay_contact_pregrasp as route_replay
    def policy(*args,**kwargs):return execute(a,*args,**kwargs)
    route_replay.replay=policy
    sys.argv=['capture_fixed_torso_initial.py','--initial-configuration',str(initial),'--state',str(state),'--contact-route',str(source),'--output',str(a.output),'--port',str(a.port),'--head-cameras','--lighting-profile','balanced','--image-size','640','480','--human-collision-model','planning_skin','--drive-profile','paired_calibrated']
    capture.main()
    path=a.output/'native-route-initial/result.json';result=json.loads(path.read_text());result.update(initialization_passed=result['passed'],passed=result['motion']['passed'],execution_completed=result['motion']['completed'],motion_replayed=False,policy_executed=True,teacher_trajectory_used=False);write(path,result)

if __name__=='__main__':main()
