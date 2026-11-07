#!/usr/bin/env python3
"""Replay saved residual rays and compare independent sphere overlap queries."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from audit_compound_rays import ray_interval
from diagnose_compound_geometry import vector,conditioned_part
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config


def residuals(row):
    points=np.array(row['vertices_world']);center=points.mean(0);planes=ConvexHull(points-center).equations
    result=[]
    for sample,ray in enumerate(row['rays']):
        if ray['kind']!='world_sphere':continue
        direction=np.array(ray['direction_world']);direction/=np.linalg.norm(direction)
        interval=ray_interval(planes,np.array(ray['origin_world'])-center,direction)
        if interval is not None and (not ray['world_hit'] or abs(ray['world_distance_m']-interval[0])>1e-4):
            result.append((sample,ray,interval))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'artifacts/compound-reaching/followup/consistency-probe.json');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/compound-reaching/followup/residual-contacts.json');ap.add_argument('--variants',nargs='+',default=['pca1']);mode=ap.add_mutually_exclusive_group();mode.add_argument('--all',action='store_true');mode.add_argument('--controls',action='store_true');ap.add_argument('--surface-overlaps',action='store_true');ap.add_argument('--scene-only',action='store_true');ap.add_argument('--simulate',action='store_true');ap.add_argument('--probe-radius',type=float,default=.0001);ap.add_argument('--approach',action='store_true');ap.add_argument('--indices',type=int,nargs='+');ap.add_argument('--identity-pose',action='store_true');args=ap.parse_args()
    if args.probe_radius<=0:ap.error('Probe radius must be positive')
    parts=[]
    originals=json.loads((ROOT/'artifacts/compound-reaching/native-compound-payload.json').read_text())['parts']
    if args.controls:
        for index,half in [(-1,[.005,.005,.005]),(-2,[.005,.005,.00005])]:
            original=dict(link='right_arm/link_5',vertices=[vector(np.array([x,y,z])*half) for x in [-1,1] for y in [-1,1] for z in [-1,1]])
            for variant in args.variants:
                parts.append(dict(shape=conditioned_part(original,index,variant),rays=[],points=[],sphereDirections=96,vertexDirections=True,interiorContactRadii=[5e-5,2e-4,1e-3]))
        rows=[]
    elif args.all or args.indices is not None:
        for index in args.indices if args.indices is not None else range(len(originals)):
            original=originals[index]
            for variant in args.variants:
                parts.append(dict(shape=conditioned_part(original,index,variant),rays=[],points=[],sphereDirections=96,vertexDirections=True,interiorContactRadii=[5e-5,2e-4,1e-3]))
        rows=[]
    else:
        raw=json.loads(args.source.read_text());payload=json.loads(args.source.with_suffix('.payload.json').read_text());rows=zip(raw['parts'],payload['parts'])
    for row,shape in rows:
        if row['variant']!='pca1':continue
        issues=residuals(row)
        if not issues:continue
        rays=[];points=[]
        for sample,ray,interval in issues:
            origin=np.array(ray['origin_world']);direction=np.array(ray['direction_world']);direction/=np.linalg.norm(direction)
            rays.append(dict(sample=sample,origin=vector(origin),direction=vector(direction),length=.5))
            # Include points on either side of the predicted entry and at the
            # middle of the ray interval. Different sphere radii check whether
            # the overlap result is sensitive to the probe's own size.
            distances=list(interval[0]+np.array([-.0002,-.0001,0.,.0001,.0002]))+[(interval[0]+interval[1])/2]
            for i,distance in enumerate(distances):
                for radius in [5e-5,2e-4,1e-3]:
                    points.append(dict(sample=sample,kind='entry_offset_'+str(i),center=vector(origin+direction*distance),radius=radius))
        for variant in args.variants:
            selected=shape if variant==shape['variant'] else conditioned_part(originals[row['index']],row['index'],variant)
            parts.append(dict(shape=selected,rays=rays,points=points))
    if not parts:raise ValueError('No parts in the supplied source')
    for part in parts:part.update(surfaceOverlapProbes=args.surface_overlaps,surfaceProbeRadius=args.probe_radius,approachContacts=args.approach)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.with_suffix('.payload.json').write_text(json.dumps(dict(parts=parts,sceneOnly=args.scene_only,identityPose=args.identity_pose)))
    config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');config.pop('pose_pattern',None);config['rcareworld'].update(executable='scripts/run_compound_reaching_player.sh',graphics=True,port=15970,log_level=0);config['assets']['output_dir']=str(args.output.parent/'contact-assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),Path(config['assets']['output_dir'])/'sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario)
        result=dict(parts=[],controls=[],physics_executed=args.simulate,isolated_scene_only=args.simulate,main_scene_physics_executed=False,robot_motion_executed=False,source=str(args.source))
        batches=args.output.with_suffix('.batches');batches.mkdir(exist_ok=True)
        for offset in range(0,len(parts),16):
            method,key=('SimulateSockCompoundContacts','sock_compound_simulated_contacts') if args.simulate else ('ProbeSockCompoundContacts','sock_compound_contacts')
            env.sock_cloth._send_data(method,env.robot.id,json.dumps(dict(parts=parts[offset:offset+16],sceneOnly=args.scene_only,identityPose=args.identity_pose)));env._env.step(3,simulate=False);data=env.sock_cloth.data[key];result['parts'].extend(data['parts']);result['controls'].extend(data.get('controls',[]));(batches/f'{offset:05d}.json').write_text(json.dumps(data));print('contact parts',len(result['parts']),len(parts),flush=True)
        args.output.write_text(json.dumps(result))


if __name__=='__main__':main()
