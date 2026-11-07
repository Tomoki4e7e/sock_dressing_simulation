#!/usr/bin/env python3
"""Compare cooking and transformed raycasts without installing replacements."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.arm_reaching import ROS_UNITY


def vector(value):
    return dict(zip('xyz', map(float, value)))


def conditioned_part(part, index, variant):
    base,_,modifier=variant.partition('_')
    if modifier:
        result=conditioned_part(part,index,base);result['variant']=variant
        if modifier in ('default','none'):result['cookingMode']=modifier
        elif modifier=='nobake':result['skipBake']=True
        elif modifier=='reverse':
            result['triangles']=np.array(result['triangles']).reshape(-1,3)[:,[0,2,1]].ravel().tolist()
        else:raise ValueError('Unknown cooking modifier: '+modifier)
        return result
    points = np.array([[v[a] for a in 'xyz'] for v in part['vertices']], dtype=np.float32)
    if variant in ('native','box'):
        center=points.astype(float).mean(0).astype(np.float32)
        axes=np.eye(3);scale=np.ones(3,dtype=np.float32)
    elif variant.startswith('uniform'):
        center = points[0].copy()
        half = np.zeros(3, dtype=np.float32)
        for point in points:
            low, high = np.minimum(center-half, point), np.maximum(center+half, point)
            half = (high-low)*np.float32(.5)
            center = low+half
        axes = np.eye(3)
        scale = np.full(3, np.max(half*2)/float(variant[7:]), dtype=np.float32)
    else:
        center = points.astype(float).mean(0)
        _, _, axes_t = np.linalg.svd(points-center, full_matrices=False)
        axes = axes_t.T
        if np.linalg.det(axes)<0:
            axes[:, -1] *= -1
        projected = (points-center)@axes
        midpoint = (projected.min(0)+projected.max(0))/2
        center += axes@midpoint
        scale = np.ptp(projected, axis=0)/float(variant[3:]) if variant.startswith('pca') and variant!='pcabaked' else None
        center = center.astype(np.float32)
    rotation = Rotation.from_matrix(axes).as_quat().astype(np.float32)
    axes = Rotation.from_quat(rotation).as_matrix()
    padding=None
    if variant.startswith('pad'):
        ratio=float('0.'+variant[3:]);projected=(points-center)@axes;width=np.ptp(projected,axis=0);padding=np.maximum(ratio*width.max()-width,0)/2
        offsets=np.array([[x,y,z] for x in [-padding[0],padding[0]] for y in [-padding[1],padding[1]] for z in [-padding[2],padding[2]]])
        projected=np.unique((projected[:,None,:]+offsets[None,:,:]).reshape(-1,3),axis=0)
        scale=np.full(3,np.ptp(projected,axis=0).max()/100,dtype=np.float32)
        normalized=(projected/scale).astype(np.float32)
        normalized=normalized[ConvexHull(normalized).vertices]
    elif variant=='pcabaked':
        scale=np.ones(3,dtype=np.float32)
        normalized=((points-center)@axes).astype(np.float32)
    else:
        scale = scale.astype(np.float32)
        normalized = (((points-center)@axes)/scale).astype(np.float32)
    hull = ConvexHull(normalized)
    faces = hull.simplices.copy()
    for face, plane in zip(faces, hull.equations):
        a, b, c = normalized[face].astype(float)
        if np.dot(np.cross(b-a,c-a),plane[:3])<0:
            face[1],face[2] = face[2],face[1]
    result=dict(index=index, link=part['link'], variant=variant,
                vertices=[vector(p) for p in normalized], triangles=faces.ravel().tolist(),
                position=vector(center), scale=vector(scale),
                rotation=dict(zip('xyzw', map(float, rotation))))
    if padding is not None:
        result['padding_upper_bound_m']=float(np.linalg.norm(padding))
    if variant=='box':result['primitiveBox']=True
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--payload-only', action='store_true', help='Write cook inputs without starting Unity')
    parser.add_argument('--variants', nargs='+', default=['uniform100', 'uniform1', 'pca1', 'pca100'])
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/compound-reaching/followup/conditioning-probe.json')
    args = parser.parse_args()
    source = ROOT/'artifacts/compound-reaching'
    payload = json.loads((source/'native-compound-payload.json').read_text())
    if args.manifest:
        manifest=json.loads(args.manifest.read_text())
        if not manifest['passed']:raise ValueError('Manifest has not passed decomposition checks')
        parts=[]
        for row in manifest['parts']:
            for part in json.loads((ROOT/row['parts_file']).read_text())['parts']:
                parts.append(dict(link=row['link'],vertices=[vector(v) for v in np.array(part['vertices'])@ROS_UNITY.T]))
        payload=dict(parts=parts)
    audit = json.loads((source/'followup/ray-audit.json').read_text())
    indices = args.indices
    if indices is None:
        indices = list(range(len(payload['parts']))) if args.all else sorted(set(
            [r['index'] for r in audit['parts'] if r['counts']['physx_misses']]
            +[r['index'] for r in audit['parts'] if r.get('hits_with_surface_distance_over_01mm')]
            +[0,4,8,13,201,420,742,849,888,2072]))
    parts = [conditioned_part(payload['parts'][i],i,variant) for i in indices for variant in args.variants]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.payload.json').write_text(json.dumps(dict(parts=parts)))
    if args.payload_only:
        return
    config = load_config(ROOT/'config/arm_reaching_safe_start.yaml')
    config.pop('pose_pattern', None)
    config['rcareworld'].update(executable='scripts/run_compound_reaching_player.sh',graphics=True,port=15970,log_level=0)
    config['assets']['output_dir'] = str(source/'followup/probe-assets')
    scenario = scenario_from_config(config,seed=0)
    prepared = _prepare(config,scenario)
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),Path(config['assets']['output_dir'])/'sock.obj',initial_joints=scenario.initial_joints)
        env.apply_scenario(scenario)
        attr = env.sock_cloth
        # Small batches bound the socket payload and preserve completed work.
        result = dict(parts=[],physics_executed=False,robot_motion_executed=False,
                      geometry_manifest=str(args.manifest),robot_data=json.loads(json.dumps(env.robot.data,default=lambda a:a.tolist())))
        for offset in range(0,len(parts),32):
            attr._send_data('ProbeSockCompoundGeometry',env.robot.id,json.dumps(dict(parts=parts[offset:offset+32])))
            env._env.step(3,simulate=False)
            result['parts'].extend(attr.data['sock_compound_probe']['parts'])
            args.output.write_text(json.dumps(result))
            print('probed',len(result['parts']),len(parts),flush=True)


if __name__=='__main__':
    main()
