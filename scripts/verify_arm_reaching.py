#!/usr/bin/env python3
"""Capture authoritative scenes and verify right-arm reaching without moving the base."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.joints import JointMap
from sock_dressing_simulation.grasp_geometry import RightHand
from photograph_grasp_candidates import photograph

CHOICES=[(3,8),(12,9)]

def run(number,index,port,snapshot_only,config_path='config/pose_patterns_toe_contact.yaml',output='artifacts/arm-reaching'):
    out=ROOT/output/f'pose-{number:02d}';out.mkdir(parents=True,exist_ok=True)
    row=json.loads((ROOT/f'artifacts/geometric-closure/pose-{number:02d}/report.json').read_text())['candidates'][index]
    prior=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text())
    saved=json.loads((ROOT/row['initial_state_source']).read_text())
    config=load_config(ROOT/config_path);settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number');settings['toe_contact']=None
    config['rcareworld'].update(executable='scripts/run_arm_reaching_player.sh',graphics=True,port=port,log_level=0)
    config['assets']['output_dir']=str(out/'assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
    pattern=next(p for p in read_patterns(Path(workbook)) if p.number==number)
    hand=RightHand(ROOT/prior['settings']['urdf'])
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),out/'assets/sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario)
        initialize_pattern(env,pattern,**settings);attr=env.sock_cloth
        attr.set_pattern_held(False);env._env.step(2,simulate=False)
        attr.initialize_preinsert(saved['initial_particles_world'],prior['samples'][row['candidate']['sample_index']]['normal']);env._env.step(2,simulate=False)
        attr.set_pattern_held(True);env._env.step(2,simulate=False)
        attr._send_data('InstallSockRigidSkinContact',env.robot.id,0);env._env.step(2,simulate=False)
        def snapshot():
            attr._send_data('GetSockArmReachingScene',env.robot.id);attr.request_particles();env._env.step(2,simulate=False)
            scene=copy.deepcopy(attr.data['sock_arm_reaching_scene'])
            scene['robot_data']=json.loads(json.dumps(env.robot.data,default=lambda a:a.tolist()));scene['particles_world']=attr.particles().tolist()
            return scene
        initial=snapshot();(out/'initial-scene.json').write_text(json.dumps(initial)+'\n')
        print('captured',number,'links',len(initial['links']),'robot shapes',len(initial['robot_colliders']),'obstacles',len(initial['obstacles']),flush=True)
        manifest=dict(number=number,candidate_index=index,candidate=row,urdf=prior['settings']['urdf'],config=config,
            right_arm_only=True,cloth_frozen=True,velocity_scale=.1,closing_speed_m_s=.005,
            position_tolerance_m=.0001,orientation_tolerance_deg=.5,arm_clearance_m=.005,hand_foot_clearance_m=.001,
            initial_joint_angles=initial['robot_data']['joint_positions'],
            requested_initial_opening_gap_m=row['opening_gap_m'],
            robot_commands_executed=False,physics_steps_after_initial_scene=0)
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        p=np.asarray(row['candidate']['grasp_center_unity_world_m'])
        # Keep the real robot visible; do not instantiate detached hand previews.
        attr._send_data('SetSockInspectionMaterial',True,0);env._env.step(2,simulate=False)
        photograph(env,out/'start-overview.png',p+np.array([0,.12,0]),np.array([-1.,.3,-.3]),np.array([0.,1.,0.]),1.3,52)
        links={r['name']:r for r in initial['links']}
        wrist=(np.array(links['right_arm/link_5']['position'])+np.array(links['right_gripper/gripper_base']['position']))*.5
        photograph(env,out/'wrist-close.png',wrist,np.array([-1.,.5,-.4]),np.array([0.,1.,0.]),.45,45)
        if not snapshot_only:
            from sock_dressing_simulation.arm_reaching import verify
            verify(env,manifest,initial,out,snapshot)

def main():
    p=argparse.ArgumentParser();p.add_argument('--number',type=int);p.add_argument('--snapshot-only',action='store_true');p.add_argument('--config',default='config/pose_patterns_toe_contact.yaml');p.add_argument('--output',default='artifacts/arm-reaching');a=p.parse_args()
    for n,i in CHOICES:
        if a.number is None or a.number==n:run(n,i,15900+n,a.snapshot_only,a.config,a.output)
if __name__=='__main__':main()
