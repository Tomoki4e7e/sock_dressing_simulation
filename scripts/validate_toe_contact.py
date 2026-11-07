#!/usr/bin/env python3
"""Measure native cloth response using a clone of the actual finger mesh.

This diagnostic fixture does not move the robot or demonstrate a grasp.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.scenario import scenario_from_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--number',type=int,default=1)
    parser.add_argument('--config',type=Path,default=ROOT/'config/pose_patterns_toe_contact.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/toe-contact-probe')
    parser.add_argument('--port',type=int,default=14200)
    parser.add_argument('--settle-steps',type=int,default=300)
    parser.add_argument('--graphics',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    config=load_config(args.config)
    settings=config.pop('pose_pattern');settings['toe_contact']['settle_steps']=args.settle_steps
    workbook=Path(settings.pop('workbook'));settings.pop('number')
    pattern=next(p for p in read_patterns(workbook) if p.number==args.number)
    config['rcareworld'].update(graphics=args.graphics,port=args.port,log_level=0)
    config['assets']['output_dir']=str(args.output/'assets')
    scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
    result=dict(number=args.number,fixture='clone of actual Dry-AIREC finger collision mesh',
                robot_motion_or_grasp_validated=False)
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),args.output/'assets/sock.obj',initial_joints=scenario.initial_joints)
        env.apply_scenario(scenario)
        try:
            report=initialize_pattern(env,pattern,**settings)
        except RuntimeError as error:
            debug=getattr(env,'_toe_contact_debug',None)
            if debug is None:raise
            report=debug['report'];result['settling_error']=str(error)
        result['preparation']=report
        attr=env.sock_cloth
        attr.request_particles();env._env.step(simulate=False)
        before=attr.particles().copy()
        source=env._toe_contact_debug['source']
        if args.graphics:
            from PIL import Image
            attr._send_data('SetSockInspectionMaterial',True,env.robot.id);env._env.step(simulate=False)
            center=(before.min(0)+before.max(0))*.5
            forward=np.asarray(report['native_pose']['toe_direction']);horizontal=forward.copy();horizontal[1]=0
            side=np.cross(horizontal,[0.,1.,0.]);side/=np.linalg.norm(side)
            for name,direction in [('side',side),('dorsal',np.asarray(report['native_pose']['dorsal_direction']))]:
                position=center+direction*.60;look=center-position
                yaw=np.degrees(np.arctan2(look[0],look[2]));pitch=-np.degrees(np.arctan2(look[1],np.linalg.norm(look[[0,2]])))
                env.recording_camera.SetTransform(position=position.tolist(),rotation=[float(pitch),float(yaw),0.])
                env._env.step(simulate=False);env._capture_recording_camera(simulate=False)
                Image.fromarray(env._capture_recording_camera(simulate=False)['rgb']).save(args.output/(name+'.png'))
            attr._send_data('SetSockInspectionMaterial',False,env.robot.id);env._env.step(simulate=False)
        (args.output/'before.json').write_text(json.dumps(dict(particles_world=before.tolist(),source=source,
            triangles=attr.data['particle_surface_triangles'],edges=attr.data['particle_edges']),indent=2)+'\n')
        band=np.asarray(attr.data['sock_toe_contact']['band_particle_indices'],dtype=int)
        from scipy.spatial import ConvexHull
        planes=ConvexHull(np.asarray(source['surfaces'][0]['vertices'])).equations
        signed=(before@planes[:,:3].T+planes[:,3]).max(1)
        inside=int(band[np.argmax(signed[band])])
        rest=np.asarray(source['rest_positions']);material=rest[:,2]-rest[:,2].min()
        tail=np.flatnonzero(material>=material.max()-.025)
        separation=np.linalg.norm(before[tail,None]-before[band][None],axis=2).min(axis=1)
        outside=int(tail[np.argmax(separation)])
        trials=[]
        # First verify that an external probe cannot contact the excluded tail.
        for name,particle in [('outside_band',outside),('inside_band',inside)]:
            attr._send_data('GetSockToeContact');attr.request_particles();env._env.step(simulate=False)
            initial=copy.deepcopy(attr.data['sock_toe_contact']);start=attr.particles().copy()
            rows=[]
            for offset in np.linspace(.080,-.080,41):
                attr._send_data('SetSockToeFingerProbe',particle,float(offset));env._env.step(simulate=False)
                env.advance_physics(2)
                attr._send_data('GetSockToeContact');attr.request_particles();env._env.step(simulate=False)
                row=copy.deepcopy(attr.data['sock_toe_contact'])
                row['probe_offset_m']=float(offset)
                row['target_displacement_m']=float(np.linalg.norm(attr.particles()[particle]-start[particle]))
                rows.append(row)
            trial=dict(name=name,particle=particle,initial=initial,samples=rows,
                added_contacts=rows[-1]['native_right_finger_contacts']-initial['native_right_finger_contacts'],
                maximum_target_displacement_m=max(r['target_displacement_m'] for r in rows))
            trials.append(trial)
            attr._send_data('RemoveSockToeFingerProbe');env._env.step(simulate=False)
            print(json.dumps({k:v for k,v in trial.items() if k not in ('initial','samples')}),flush=True)
        result['trials']=trials
        result['checks']=dict(outside_band_rejected=trials[0]['added_contacts']==0,
            inside_band_native_contacts=trials[1]['added_contacts']>0,
            inside_band_impulse=max(r['maximum_normal_impulse'] for r in trials[1]['samples'])>1e-8,
            no_leaked_contacts=trials[1]['samples'][-1]['outside_band_contacts']==0,
            free_band=trials[1]['samples'][-1]['band_fixed_particles']==0,
            measurable_band_motion=trials[1]['maximum_target_displacement_m']>.001)
        result['contact_response_passed']=all(result['checks'].values())
        result['passed']=result['contact_response_passed'] and report['toe_contact']['passed']
        attr.request_particles();env._env.step(simulate=False)
        (args.output/'after.json').write_text(json.dumps(dict(particles_world=attr.particles().tolist(),source=source,
            triangles=attr.data['particle_surface_triangles'],edges=attr.data['particle_edges']),indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],contact_checks=result['checks'])),flush=True)
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
