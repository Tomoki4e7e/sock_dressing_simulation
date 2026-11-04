#!/usr/bin/env python3
"""Load the refined mesh as a real Obi actor and validate all frozen states."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from photograph_grasp_candidates import photograph

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--numbers',type=int,nargs='+');a=parser.parse_args()
    summary=json.loads((ROOT/'artifacts/bulged-cuff/summary.json').read_text())
    if not summary['passed']:raise RuntimeError('Do not install a shape that failed the geometry audit')
    rows=[r for r in summary['poses'] if a.numbers is None or r['number'] in a.numbers]
    output=ROOT/'artifacts/bulged-cuff/native';output.mkdir(exist_ok=True)
    config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');config.pop('pose_pattern');config['rcareworld'].update(executable='scripts/run_arm_reaching_player.sh',graphics=True,port=16200,log_level=0);config['assets']['output_dir']=str(output/'assets')
    scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
    first=ROOT/f"artifacts/bulged-cuff/pose-{rows[0]['number']:02d}";rest=np.asarray(json.loads((first/'state.json').read_text())['source']['rest_positions'])
    validation=[]
    previous=output/'validation.json'
    if previous.exists():validation=json.loads(previous.read_text()).get('poses',[])
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),first/'rest.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario);attr=env.sock_cloth
        for i,row in enumerate(rows):
            number=row['number'];folder=ROOT/f'artifacts/bulged-cuff/pose-{number:02d}';state=json.loads((folder/'state.json').read_text());target=np.asarray(state['particles_world'])
            attr._send_data('RemoveSockRigidSkinContact');attr.set_pattern_held(False);attr.set_pattern_foot_pose(row['pitch_deg'],row['roll_deg']);env._env.step(2,simulate=False)
            attr.request_preinsert_source();attr._send_data('GetSockPatternSkin');env._env.step(2,simulate=False)
            native_rest=np.asarray(attr.data['sock_preinsert_source']['rest_positions']);distance,order=cKDTree(rest).query(native_rest)
            (output/'native-rest-readback.json').write_text(json.dumps(native_rest.tolist())+'\n')
            if distance.max()>5e-6 or len(set(order.tolist()))!=len(rest) or len(order)!=len(rest):raise RuntimeError(f'Native rest topology differs: count {len(native_rest)}, expected {len(rest)}, error {distance.max()}')
            attr.initialize_preinsert(target[order],state['opening_normal']);env._env.step(2,simulate=False);attr.set_pattern_held(True);attr.request_particles();env._env.step(2,simulate=False)
            cloth_error=float(np.linalg.norm(attr.particles()-target[order],axis=1).max());native_faces=np.asarray(attr.data['particle_surface_triangles']).reshape(-1,3);mapped_faces=order[native_faces]
            same_faces={tuple(sorted(v)) for v in mapped_faces}=={tuple(sorted(v)) for v in np.asarray(state['triangles']).reshape(-1,3)}
            skin=attr.data['sock_pattern_skin'];skin_error=float(np.linalg.norm(np.asarray(skin['vertices'])-state['source']['skin_surface']['vertices'],axis=1).max())
            if cloth_error>5e-6 or skin_error>5e-6 or not same_faces:raise RuntimeError(f'Native shape mismatch: cloth {cloth_error}, skin {skin_error}, triangles {same_faces}')
            attr._send_data('InstallSockRigidSkinContact',env.robot.id,0);env._env.step(2,simulate=False);attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False)
            scene=copy.deepcopy(attr.data['sock_arm_reaching_scene']);scene['robot_data']=json.loads(json.dumps(env.robot.data,default=lambda v:v.tolist()));scene['particles_world']=target.tolist()
            if i==0:
                shared={key:scene[key] for key in ('links','robot_colliders','robot_data','externally_stepped')}
                shared_path=output/'shared-robot-scene.json'
                if not shared_path.exists():
                    temporary=shared_path.with_suffix('.tmp');temporary.write_text(json.dumps(shared)+'\n');temporary.replace(shared_path)
            for key in ('links','robot_colliders','robot_data','externally_stepped'):scene.pop(key)
            scene['shared_robot_scene']='artifacts/bulged-cuff/native/shared-robot-scene.json';scene['particle_order']='search mesh order; validated against mapped Native particles';(folder/'native-scene.json').write_text(json.dumps(scene)+'\n')
            # Hide the real robot for the cuff close-up, and use the existing
            # native preview shader for visible skin. The cloned hand stays
            # at the actual safe-start pose, away from the inspected cuff.
            links={r['name']:r for r in shared['links']}
            parts=[dict(link=name,position=dict(zip('xyz',links[name]['position'])),rotation=dict(zip('xyzw',links[name]['quaternion']))) for name in ['right_gripper/gripper_base','right_gripper/right_finger_base_link','right_gripper/right_finger_link','right_gripper/left_finger_base_link','right_gripper/left_finger_link']]
            attr._send_data('SetSockGraspPreview',json.dumps(dict(label='safe-start hand, not a grasp candidate',parts=parts)));attr._send_data('SetSockInspectionMaterial',True,env.robot.id);env._env.step(2,simulate=False)
            if number in (3,12) or i==0:
                profile=json.loads((folder/'profile.json').read_text());peak_id=int(np.flatnonzero(np.isclose(profile['depth_m'],.015,atol=1e-9))[0]);center=np.array(profile['reference_center_world'][peak_id])
                photograph(env,folder/'bulge-overview.png',center,np.array([-1.,.35,-.3]),np.array([0.,1.,0.]),.65,42)
                photograph(env,folder/'bulge-close.png',center,np.array([-1.,.15,-.2]),np.array([0.,1.,0.]),.22,42)
                before=np.array(json.loads((folder/'before-refined.json').read_text())['particles_world'])
                attr.set_pattern_held(False);env._env.step(2,simulate=False);attr.initialize_preinsert(before[order],state['opening_normal']);env._env.step(2,simulate=False);attr.set_pattern_held(True);env._env.step(2,simulate=False)
                photograph(env,folder/'before-close.png',center,np.array([-1.,.15,-.2]),np.array([0.,1.,0.]),.22,42)
                attr.set_pattern_held(False);env._env.step(2,simulate=False);attr.initialize_preinsert(target[order],state['opening_normal']);env._env.step(2,simulate=False);attr.set_pattern_held(True);attr.request_particles();env._env.step(2,simulate=False)
                if np.linalg.norm(attr.particles()-target[order],axis=1).max()>5e-6:raise RuntimeError('Comparison photography changed the final shape')
            report=dict(number=number,passed=True,particle_count=len(native_rest),rest_match_error_m=float(distance.max()),cloth_error_m=cloth_error,skin_error_m=skin_error,triangle_geometry_matches=same_faces,geometry_frozen=True,torso_ik_enabled=False)
            validation=[r for r in validation if r['number']!=number]+[report];(folder/'native-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
    (output/'validation.json').write_text(json.dumps(dict(passed=True,poses=validation),indent=2)+'\n')

if __name__=='__main__':main()
