#!/usr/bin/env python3
"""Verify calibrated gravity in Unity at the archived quasi-static poses."""
from __future__ import annotations
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='1'
import argparse,copy,importlib.util,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.joints import JointMap
from sock_dressing_simulation.gravity_reference import GravityTree,StaticGravityReference,ARM_COLUMNS,ARM_NAMES


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False,default=lambda value:value.tolist())+'\n')


def apply_gravity_reference(env,model_path):
    return StaticGravityReference(model_path).configure_unity(env)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/gravity-calibration-v1')
    ap.add_argument('--port',type=int,default=19380);a=ap.parse_args();out=a.output.resolve();native=out/'native';native.mkdir(exist_ok=True)
    model_path=out/'gravity-model.json';model=json.loads(model_path.read_text());report=json.loads((out/'report.json').read_text());manifest=json.loads((out/'reference-manifest.json').read_text())
    tree=GravityTree(model['calibrated_urdf']);mapping=None
    # Only per-process RPC locks move to this output; no hardware connection.
    lock_dir=native/'runtime-locks';lock_dir.mkdir(exist_ok=True)
    path=ROOT/'.deps/RCareWorld/pyrcareworld/pyrcareworld/utils/locker.py'
    spec=importlib.util.spec_from_file_location('pyrcareworld.utils.locker',path);locker=importlib.util.module_from_spec(spec);spec.loader.exec_module(locker)
    def local_lock(self,name):
        self.lck_path=str(lock_dir/(name+'.lck'));Path(self.lck_path).touch(exist_ok=True);self.fp=None
    locker.Locker.__init__=local_lock;sys.modules['pyrcareworld.utils.locker']=locker
    config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');config.pop('pose_pattern',None)
    config['rcareworld'].update(executable='scripts/run_registered_reaching_player.sh',graphics=True,port=a.port,log_level=0)
    config['assets']['output_dir']=str(native/'assets');config['scene']['robot_joint_gravity']=False
    scenario=scenario_from_config(config);prepared=_prepare(config,scenario);mapping=JointMap.from_config(config)
    targets=[]
    for row,segment in zip(manifest['rows'],report['segments']):
        q=np.loadtxt(Path(row['path'])/'angle.csv',delimiter=',')
        for index in range(segment['start'],segment['end_exclusive']):targets.append((row['id'],index,q[index]))
    result_rows=[]
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),Path(prepared['sock_obj']),initial_joints=scenario.initial_joints)
        env.sock_cloth.set_pattern_held(True);env.robot.EnabledAllCollider(False);env._env.step(simulate=False)
        full=np.array(env.robot.data['joint_positions'],dtype=float);n=len(full)
        # Configure the hold through the dedicated API below. The SDK's
        # SetJointDamping also changes dry friction and body drag.
        def set_pose(q):
            command=mapping.merge_for_simulator(q,full)
            env.robot.SetJointPositionDirectly(command.tolist());env.robot.SetJointPosition(command.tolist());env.robot.SetJointVelocity(np.zeros(n).tolist())
            env.sock_cloth._send_data('SynchronizeSockGravityReferenceMimics',env.robot.id)
            env.sock_cloth._send_data('SetSockReplayDriveVelocities',env.robot.id,json.dumps(dict(values=np.zeros(n).tolist())))
            env._env.Simulate(.000001,1);env.robot.GetJointInverseDynamicsForce();env._env.step(simulate=False)
            return mapping.from_simulator(env.robot.data['joint_positions'])
        # Read the existing import before the opt-in correction, at exactly
        # the same pose used afterwards. GetJointGravityForces itself returns
        # a model calculation even when useGravity is false.
        q=targets[0][2];actual=set_pose(q)
        before_force=np.array(env.robot.data['gravity_forces'])[mapping.simulator_indices][ARM_COLUMNS]
        env.sock_cloth._send_data('GetSockGravityReference',env.robot.id);env._env.step(simulate=False)
        before=copy.deepcopy(env.sock_cloth.data['sock_gravity_reference']);write(native/'inertials-before.json',before)
        applied=apply_gravity_reference(env,model_path);write(native/'inertials-applied.json',applied)
        for number,(episode,index,q) in enumerate(targets):
            bounded=q.copy()
            for column in ARM_COLUMNS:
                limits=tree.joints[mapping.names[column]].find('limit')
                bounded[column]=np.clip(bounded[column],float(limits.get('lower')),float(limits.get('upper')))
            clipped_error=float(abs(bounded[ARM_COLUMNS]-q[ARM_COLUMNS]).max())
            if clipped_error>5e-5:raise RuntimeError('Archived pose substantially exceeds the physical URDF bounds')
            actual=set_pose(bounded);joint_error=float(abs(actual[ARM_COLUMNS]-bounded[ARM_COLUMNS]).max())
            native_g=np.array(env.robot.data['gravity_forces'])[mapping.simulator_indices][ARM_COLUMNS]
            prediction=tree.predict(actual)[0];error=float(abs(native_g-prediction).max())
            if joint_error>2e-6:raise RuntimeError(f'Native archived pose differs: {joint_error}')
            if error>.005:raise RuntimeError(f'Native gravity and potential-gradient model differ: {episode}/{index}: {error}')
            result_rows.append(dict(episode=episode,frame=index,source_q_rad_or_m=q.tolist(),actual_q_rad_or_m=actual.tolist(),
                source_limit_overrun_rad=clipped_error,reference_q_rad_or_m=bounded.tolist(),
                maximum_arm_position_error_rad=joint_error,native_gravity_Nm=native_g.tolist(),analytic_gravity_Nm=prediction.tolist(),maximum_error_Nm=error))
            if number%25==0:print('NATIVE GRAVITY',number+1,'/',len(targets),'maximum error',error,flush=True)
        # Confirm gravity has an observable physical effect: let the servo
        # hold one posture without teleporting/resetting velocity each tick.
        q=targets[0][2];set_pose(q);hold=[]
        for k in range(150):
            env._env.Simulate(.01,1);env.robot.GetJointInverseDynamicsForce();env._env.step(simulate=False)
            if k>=100:
                positions=np.array(env.robot.data['joint_positions']);actual=mapping.from_simulator(positions)
                fixed=positions[mapping.fixed_simulator_indices].copy();fixed[list(mapping.fixed_revolute_indices)]=np.deg2rad(fixed[list(mapping.fixed_revolute_indices)])
                actual_fixed=dict(zip(mapping.fixed_names,fixed.tolist()));g=tree.predict(actual,fixed=actual_fixed)[0]
                drive=np.array(env.robot.data['drive_forces'])[mapping.simulator_indices][ARM_COLUMNS]
                hold.append(dict(actual_q=actual.tolist(),actual_fixed_joint_positions=actual_fixed,drive_effort_Nm=drive.tolist(),gravity_Nm=g.tolist()))
        hold_error=np.array([np.array(r['drive_effort_Nm'])-r['gravity_Nm'] for r in hold]);hold_rmse=float(np.sqrt(np.mean(hold_error**2)))
        write(native/'static-hold.json',dict(rows=hold,rmse_drive_vs_gravity_Nm=hold_rmse,scope='Stationary gravity hold only; no contact or motion-inertia validation'))
        if hold_rmse>.1:raise RuntimeError(f'Physical gravity hold mismatch: {hold_rmse}')
    result=dict(passed=True,poses=len(result_rows),joint_names=ARM_NAMES,
        maximum_native_vs_analytic_error_Nm=max(r['maximum_error_Nm'] for r in result_rows),
        before_import_rmse_Nm=float(np.sqrt(np.mean((before_force-result_rows[0]['analytic_gravity_Nm'])**2))),
        static_hold_drive_vs_gravity_rmse_Nm=hold_rmse,gravity_enabled=True,
        maximum_source_limit_overrun_rad=max(r['source_limit_overrun_rad'] for r in result_rows),
        collider_policy='Robot contacts disabled for this noncontact gravity isolation test',
        rows=result_rows,model_scope='Static gravity only; this test is not a dynamics replay')
    write(out/'native-validation.json',result);print('NATIVE GRAVITY VALIDATION PASSED',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
