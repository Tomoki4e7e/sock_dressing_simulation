#!/usr/bin/env python3
"""Compare zero gravity and normal gravity from the same held-A shape."""
from pathlib import Path
import argparse
import copy
import json
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.config import load_config,resolve_package_path
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.held_ankle_poses import audit_actual_skin


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/sock-pose-pattern/held-gravity/gravity-control')
    parser.add_argument('--port',type=int,default=15000)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=150)
    args=parser.parse_args()
    if args.steps<100:parser.error('At least 100 steps required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    config=load_config(ROOT/'config/pose_patterns.yaml')
    config['pose_pattern']['gravity_drape']['enabled']=False
    config['rcareworld'].update(port=args.port,graphics=not args.headless)
    config['assets']['output_dir']=str(out/'assets')
    scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario);images=[]
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),out/'assets/sock.obj',initial_joints=scenario.initial_joints)
        reference=env.apply_scenario(scenario)['pose_pattern'];attr=env.sock_cloth
        source=env._pose_pattern_source;initial_points=attr.particles().copy()
        def stats():
            attr._send_data('GetSockPatternGravityDrape');attr.request_particles();env._env.step(simulate=False)
            return copy.deepcopy(attr.data['sock_pattern_gravity_drape'])
        if not args.headless:
            attr._send_data('SetSockInspectionMaterial',True,env.robot.id);env._env.step(simulate=False)
            center=(initial_points.min(0)+initial_points.max(0))*.5+np.array([0,-.04,0])
            forward=np.array(reference['native_pose']['toe_direction']);forward[1]=0
            side=np.cross(forward,[0.,1.,0.]);side/=np.linalg.norm(side)
            position=center+side*.65;look=center-position
            env.recording_camera.SetTransform(position=position.tolist(),rotation=[float(-np.degrees(np.arctan2(look[1],np.linalg.norm(look[[0,2]])))),float(np.degrees(np.arctan2(look[0],look[2]))),0.])
            env._env.step(simulate=False)
        def capture(name):
            if args.headless:return
            attr._send_data('SetSockInspectionMaterial',False,env.robot.id);env._env.step(simulate=False)
            attr._send_data('SetSockInspectionMaterial',True,env.robot.id);env._env.step(simulate=False)
            env._capture_recording_camera(simulate=False)
            env._capture_recording_camera(simulate=False)
            image=Image.fromarray(env._capture_recording_camera(simulate=False)['rgb']);image.save(out/(name+'.png'))
            image.thumbnail((600,400));images.append((name,image.copy()))
        saved=resolve_package_path(config['pose_pattern']['held_reference'])/'fitted-prepared-state.json'
        attr._send_data('BeginSockPatternGravityDrape',saved.read_text(),float(config['pose_pattern']['gravity_drape'].get('pinned_length_m',.075)));env._env.step(simulate=False)
        attr._send_data('SetSockPatternDrapeGravity',False);env._env.step(simulate=False)
        start=stats();capture('initial')
        env.advance_physics(args.steps);zero=stats();capture('zero-gravity')
        attr._send_data('SetSockPatternDrapeGravity',True);env._env.step(simulate=False)
        for first in range(0,args.steps,50):env.advance_physics(min(50,args.steps-first))
        gravity=stats();capture('normal-gravity')
        body,surface=audit_actual_skin(attr.particles(),source['skin_surface'],attr.data['particle_edges'],attr.data['particle_surface_triangles'])
        zero_motion=float(np.linalg.norm(np.array(zero['tip_center_world'])-start['tip_center_world']))
        fall=float(zero['tip_center_world'][1]-gravity['tip_center_world'][1])
        checks=dict(zero_gravity_stationary=zero_motion<.001,normal_gravity_falls=fall>.02,
            actual_gravity=np.allclose(gravity['gravity_world'],[0,-9.81,0]),
            finite_mass_sections=gravity['dynamic_sections']>0 and gravity['free_mass_kg']>.005,
            fixed_region_preserved=gravity['maximum_fixed_error_m']<5e-6,
            actual_clock_advanced=gravity['physics_time_s']>zero['physics_time_s']>start['physics_time_s'],
            no_intersections=body['geometry_ok'] and surface['surface_ok'])
        result=dict(runtime_player=config['rcareworld']['executable'],passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
            initial=start,zero_gravity=zero,normal_gravity=gravity,zero_gravity_tip_motion_m=zero_motion,
            normal_gravity_tip_drop_m=fall,geometry_audit=body,surface_audit=surface)
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if images:
        page=Image.new('RGB',(600*len(images),430),'white');draw=ImageDraw.Draw(page)
        for i,(label,image) in enumerate(images):page.paste(image,(i*600,30));draw.text((i*600+10,8),label,fill='black')
        page.save(out/'gravity-comparison.jpg')
    if not result['passed']:raise SystemExit(2)

if __name__=='__main__':main()
