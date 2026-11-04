#!/usr/bin/env python3
"""Photograph both arms in ShareSet's na posture and verify the native readback."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.arm_reaching import Kinematics,shape_vertices
from photograph_grasp_candidates import photograph

def main():
 out=ROOT/'artifacts/apron-start-pose';out.mkdir(exist_ok=True)
 config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number');settings['toe_contact']=None
 config['rcareworld'].update(executable='scripts/run_arm_reaching_player.sh',graphics=True,port=15980,log_level=0);config['assets']['output_dir']=str(out/'assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario);pattern=next(p for p in read_patterns(Path(workbook)) if p.number==3)
 with SockDressingEnv(config) as env:
  env.load(Path(prepared['runtime_urdf']),out/'assets/sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario);initialize_pattern(env,pattern,**settings);attr=env.sock_cloth
  attr._send_data('InstallSockRigidSkinContact',env.robot.id,0);env._env.step(2,simulate=False);attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False)
  scene=dict(attr.data['sock_arm_reaching_scene']);scene['robot_data']=json.loads(json.dumps(env.robot.data,default=lambda a:a.tolist()));(out/'native-scene.json').write_text(json.dumps(scene));k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',scene)
  arms={side:[float(np.rad2deg(k.q0[f'{side}_arm/joint_{i}'])) for i in range(1,8)] for side in ['left','right']};torso=[float(np.rad2deg(k.q0[f'torso/joint_{i}'])) for i in range(1,4)];expected=np.array([47,0,-2.7,131,1,1,20]);expected_torso=np.rad2deg(config['joints']['fixed']['positions']);passed=all(np.allclose(v,expected,atol=.002,rtol=0) for v in arms.values()) and np.allclose(torso,expected_torso,atol=.002,rtol=0)
  source=ROOT.parent/'ShareSet/airec_joy_controller/scripts/change_pose.py';result=dict(passed=passed,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_command='na',arm_joints_degrees=arms,torso_joints_degrees=torso,expected_arm_degrees=expected.tolist(),expected_torso_degrees=expected_torso.tolist(),config='config/arm_reaching_safe_start.yaml',images=[])
  if not passed:raise RuntimeError('Native arm or torso readback differs from requested posture: '+json.dumps(result))
  attr._send_data('SetSockInspectionMaterial',True,0);env._env.step(2,simulate=False)
  points=np.concatenate([shape_vertices(r) for r in scene['robot_colliders']]);lo=points.min(0);hi=points.max(0);target=(lo+hi)/2;target[1]=.85
  for name,direction,distance in [('front',[0,.08,-1],3.0),('oblique',[-.8,.18,-1],3.2)]:
   result['images'].append(photograph(env,out/f'{name}.png',target,np.array(direction),np.array([0.,1.,0.]),distance,44))
  # Additional inspection photos omit the human only while rendering, so
  # both robot arms remain visible. No physics step or robot joint command.
  env.human.SetActive(False);env._env.step(2,simulate=False)
  try:
   robot_target=target.copy();robot_target[2]=float(config['scene']['robot_position'][2])-.15
   for name,direction,distance in [('robot-front',[0,.10,-1],2.6),('robot-oblique',[-1.,.2,-.65],2.8)]:
    picture=photograph(env,out/f'{name}.png',robot_target,np.array(direction),np.array([0.,1.,0.]),distance,44);picture['human_hidden_for_photo_only']=True;result['images'].append(picture)
  finally:
   env.human.SetActive(True);env._env.step(2,simulate=False)
  (out/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
