#!/usr/bin/env python3
"""Capture omitted human collision shapes without changing collision policy."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,copy,importlib.util,json,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.common_opening_approach import inventory,sha256
from search_translated_reaching import load
from search_centered_reaching import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-placement-search/human-colliders');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 locker_path=ROOT/'.deps/RCareWorld/pyrcareworld/pyrcareworld/utils/locker.py';spec=importlib.util.spec_from_file_location('pyrcareworld.utils.locker',locker_path);locker=importlib.util.module_from_spec(spec);spec.loader.exec_module(locker)
 def local_lock(self,name):
  self.lck_path=str(out/(name+'.lck'));Path(self.lck_path).touch(exist_ok=True);self.fp=None
 locker.Locker.__init__=local_lock;sys.modules['pyrcareworld.utils.locker']=locker
 config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');config.pop('pose_pattern');config['rcareworld'].update(executable='scripts/run_registered_reaching_player.sh',graphics=True,port=24800,log_level=0);config['assets']['output_dir']=str(out/'assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
 poses={r['scene_number']:r for r in inventory(ROOT)};summary=json.loads((ROOT/'artifacts/bulged-cuff/summary.json').read_text());angles={r['number']:r for r in summary['poses']};manifest=dict(scenes={},player=str(Path(os.environ['SOCK_REGISTERED_PLAYER']).resolve()),script_sha256=sha256(Path(__file__)),exporter_sha256=sha256(ROOT/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockArmReaching.cs'))
 with SockDressingEnv(config) as env:
  env.load(Path(prepared['runtime_urdf']),ROOT/'artifacts/bulged-cuff/pose-10/rest.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario);attr=env.sock_cloth
  for n in sorted(poses):
   pose=angles[n];attr._send_data('RemoveSockRigidSkinContact');attr.set_pattern_held(False);attr.set_pattern_foot_pose(pose['pitch_deg'],pose['roll_deg']);env._env.step(2,simulate=False);attr._send_data('InstallSockRigidSkinContact',env.robot.id,0);env._env.step(2,simulate=False);attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False)
   scene=copy.deepcopy(attr.data['sock_arm_reaching_scene']);ref=load(n);error=float(np.linalg.norm(np.asarray(scene['human_skin']['vertices'])-ref['human_skin']['vertices'],axis=1).max())
   if error>5e-6:raise ValueError('Human shape mismatch '+str((n,error)))
   if 'human_colliders' not in scene:raise ValueError('Player does not export human colliders')
   data=dict(scene_number=n,human_skin_error_m=error,colliders=scene['human_colliders']);path=out/f'{n:02d}.json';write(path,data);manifest['scenes'][str(n)]=dict(path=str(path.relative_to(ROOT)),sha256=sha256(path));write(out/'manifest.json',manifest);print('HUMAN COLLIDERS',n,len(data['colliders']),error,flush=True)
if __name__=='__main__':main()
