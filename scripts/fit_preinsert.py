#!/usr/bin/env python3
"""Shrink a sock's rest circumference and validate unpinned foot retention."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.circumference_fit import run_fit, restore_fit, hold_fit
from sock_dressing_simulation.preinsert import audit_preinsert, audit_preinsert_surface


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=list('ABCDE'),default='A')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/sock-pose-pattern/preinsert-fitted')
    p.add_argument('--restore',type=Path,help='Load a previously verified fit, then repeat its gravity hold')
    p.add_argument('--minimum-scale',type=float)
    p.add_argument('--scale-step',type=float)
    p.add_argument('--settle-steps',type=int)
    p.add_argument('--hold-steps',type=int)
    p.add_argument('--target-cuff-force',type=float)
    p.add_argument('--compliance',type=float)
    p.add_argument('--particle-radius',type=float)
    p.add_argument('--friction',type=float)
    p.add_argument('--port',type=int,default=8200)
    p.add_argument('--headless',action='store_true',help='Run physics and save states without camera images')
    p.add_argument('--player',type=str)
    p.add_argument('--friction-control-steps',type=int,default=0,
        help='After a valid hold, remove friction and record a comparison without changing the saved start state')
    args=p.parse_args()
    if args.restore:
        config=yaml.safe_load((args.restore/'runtime-config.yaml').read_text())
        args.stage=config['preinsert']['stage']
    else:
        config=load_config(ROOT/'config/preinsert_fitted.yaml')
    settings=config.pop('circumference_fit')
    for name,value in dict(minimum_scale=args.minimum_scale,scale_step=args.scale_step,
            settle_steps=args.settle_steps,hold_steps=args.hold_steps,target_cuff_force_n=args.target_cuff_force).items():
        if value is not None:settings[name]=value
    material=config['obi']['expected']
    for name,value in dict(stretch_compliance=args.compliance,friction=args.friction,particle_radius_m=args.particle_radius).items():
        if value is not None:material[name]=value
    out=args.output.resolve()/args.stage
    if args.restore and out==args.restore.resolve():
        p.error('--output must differ from the saved fit directory')
    out.mkdir(parents=True,exist_ok=True)
    (out/'telemetry.jsonl').write_text('')
    config['rcareworld'].update(graphics=not args.headless,port=args.port)
    if args.headless:config['rcareworld']['log_level']=0
    if args.player:config['rcareworld']['executable']=args.player
    config['preinsert']['stage']=args.stage
    config['assets']['output_dir']=str(out/'assets')
    config['camera']['fov']=50.
    saved_config=copy.deepcopy(config);saved_config['circumference_fit']=settings
    (out/'runtime-config.yaml').write_text(yaml.safe_dump(saved_config,sort_keys=False))
    scenario=scenario_from_config(config,seed=0)
    prepared=_prepare(config,scenario)
    images=[];started=time.monotonic()
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),out/'assets/sock.obj',initial_joints=scenario.initial_joints)
        application=env.apply_scenario(scenario)
        env._read_only_native_observation=True
        attr=env.sock_cloth
        source=copy.deepcopy(attr.data['sock_preinsert_source'])
        attr.request_prepared_state();env._env.step(simulate=False)
        (out/'initial-prepared-state.json').write_text(attr.data['prepared_sock_state']+'\n')
        toe,heel,calf,ankle=[np.asarray(source[k]) for k in ('toe','heel','calf','ankle')]
        axis=(toe-heel)/np.linalg.norm(toe-heel)
        side=np.cross(axis,calf-ankle);side/=np.linalg.norm(side)
        initial=attr.particles();center=(initial.min(axis=0)+initial.max(axis=0))*.5
        def capture(label,row):
            if args.headless:return
            attr._send_data('SetSockInspectionMaterial',True,env.robot.id)
            position=center+side*.65
            forward=center-position
            yaw=np.degrees(np.arctan2(forward[0],forward[2]))
            pitch=-np.degrees(np.arctan2(forward[1],np.linalg.norm(forward[[0,2]])))
            env.recording_camera.SetTransform(position=position.tolist(),rotation=[float(pitch),float(yaw),0.])
            env._env.step(simulate=False)
            camera=env._capture_recording_camera(simulate=False)
            image=Image.fromarray(camera['rgb']);image.save(out/(label+'.png'))
            image.thumbnail((640,480));images.append((label,image))
            attr._send_data('SetSockInspectionMaterial',False,env.robot.id);env._env.step(simulate=False)
        def sample(row):
            row['wall_elapsed_s']=time.monotonic()-started
            with (out/'telemetry.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            print(f"{args.stage} {row['phase']} scale={row['rest_circumference_scale']:.3f} contact={row['near_foot_particles']} cuff={row['cuff_force_median_n']:.4f} N motion={row['cuff_observed_speed_m_s']:.5f} m/s inside={row['inside_foot_particles']}",flush=True)
            if row['phase']=='contract' and round((1-row['rest_circumference_scale'])/.01)%10==0:
                capture(f"contract-{row['rest_circumference_scale']:.2f}",row)
        if args.restore:
            restored=restore_fit(env,args.restore)
            attr.request_circumference_fit();env._env.step(simulate=False)
            reference=attr.data['sock_circumference_fit']['cuff_center_world']
            result=hold_fit(env,steps=settings['hold_steps'],reference_center=reference,on_sample=sample)
            result.update(passed=result['assessment']['passed'],rest_circumference_scale=restored['rest_circumference_scale'],restoration=restored)
            result['foot_hull']=json.loads((args.restore/'foot-collision-hull.json').read_text())
            capture('held',result['telemetry'][-1])
        else:
            result=run_fit(env,**settings,on_sample=sample,on_phase=capture)
        attr.request_prepared_state();attr.request_particles();env._env.step(simulate=False)
        (out/'fitted-prepared-state.json').write_text(attr.data['prepared_sock_state']+'\n')
        geometry=dict(source=source,particles_world=attr.particles().tolist(),
            triangles=attr.data['particle_surface_triangles'],edges=attr.data['particle_edges'],
            rest_edge_lengths=attr.data['particle_rest_edge_lengths'])
        (out/'fitted-geometry.json').write_text(json.dumps(geometry,indent=2)+'\n')
        (out/'foot-collision-hull.json').write_text(json.dumps(result['foot_hull'],indent=2)+'\n')
        surface=dict(surfaces=[result['foot_hull']])
        if result['foot_hull'].get('calf_surface'):
            surface['surfaces'].append(result['foot_hull']['calf_surface'])
        result['geometry_audit']=audit_preinsert(geometry['particles_world'],surface,geometry['edges'],geometry['rest_edge_lengths'])
        result['surface_audit']=audit_preinsert_surface(geometry['particles_world'],surface,geometry['edges'],geometry['triangles'])
        result['passed']=result['passed'] and result['geometry_audit']['geometry_ok'] and result['surface_audit']['surface_ok']
        result.update(stage=args.stage,initial_application=application,material=material,
            material_rest_lengths_modified=True,wall_elapsed_s=time.monotonic()-started,
            physical_retention_validated=result['passed'],
            model='Native distance elasticity, finite-mass unilateral contact and Coulomb friction; no world anchors or gripper pins.')
        player=Path(config['rcareworld']['executable'])
        if not player.is_absolute():player=ROOT/player
        result['runtime']={name:hashlib.sha256((player.parent/'Player_Data/Managed'/name).read_bytes()).hexdigest()
                           for name in ('RCareWorld.dll','Obi.dll')}
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(passed=result['passed'],assessment=result['assessment'],surface=result['surface_audit'])),flush=True)
        if result['passed'] and args.friction_control_steps:
            reference=result['telemetry'][-1]['cuff_center_world']
            attr.set_fit_friction(0);env._env.step(simulate=False)
            control=hold_fit(env,steps=args.friction_control_steps,reference_center=reference)
            capture('friction-zero-control',control['telemetry'][-1])
            (out/'friction-zero-control.json').write_text(json.dumps(control,indent=2)+'\n')
    if images:
        columns=min(3,len(images));rows=math.ceil(len(images)/columns)
        montage=Image.new('RGB',(640*columns,510*rows),'white');draw=ImageDraw.Draw(montage)
        for i,(label,image) in enumerate(images):
            x=(i%columns)*640;y=(i//columns)*510
            montage.paste(image,(x,y+30));draw.text((x+10,y+8),label,fill='black')
        montage.save(out/'contraction-overview.jpg')
    if not result['passed']:sys.exit(2)

if __name__=='__main__':main()
