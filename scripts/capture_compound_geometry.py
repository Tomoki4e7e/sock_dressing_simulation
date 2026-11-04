#!/usr/bin/env python3
"""Install experimental compound colliders and read back PhysX/Obi geometry."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,copy
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import cKDTree,ConvexHull,QhullError
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.arm_reaching import Kinematics,ROS_UNITY
OUT=ROOT/'artifacts/compound-reaching'

def main():
    manifest=json.loads((OUT/'decomposition.json').read_text());assert manifest['passed'];parts=[];canonical=[];skipped=[]
    for row in manifest['parts']:
        data=json.loads((ROOT/row['parts_file']).read_text())
        for part in data['parts']:
            vertices=np.array(part['vertices'])@ROS_UNITY.T
            # Triangulate in float coordinates used by Unity. Original double
            # facets can become non-convex numerically after casting thin cuts.
            vf=vertices.astype(np.float32);center=vf[0].copy();half=np.zeros(3,dtype=np.float32)
            for vertex in vf:
                minimum=np.minimum((center-half).astype(np.float32),vertex);maximum=np.maximum((center+half).astype(np.float32),vertex);half=((maximum-minimum)*np.float32(.5)).astype(np.float32);center=(minimum+half).astype(np.float32)
            extent=np.float32(np.max((half*np.float32(2)).astype(np.float32))/np.float32(100));normalized=((vf-center)/extent).astype(np.float32).astype(float)
            try:
                native_hull=ConvexHull(normalized);faces=native_hull.simplices.copy()
                for face,plane in zip(faces,native_hull.equations):
                    va,vb,vc=normalized[face]
                    if np.dot(np.cross(vb-va,vc-va),plane[:3])<0:face[1],face[2]=face[2],face[1]
            except QhullError:
                _,_,axes=np.linalg.svd(vertices-vertices.mean(0),full_matrices=False);thickness=float(np.ptp(vertices@axes[-1]))
                if thickness>1e-6:raise
                skipped.append(dict(link=row['link'],maximum_normal_thickness_m=thickness));continue
            for face in faces:
                va,vb,vc=normalized[face];normal=np.cross(vb-va,vc-va);length=np.linalg.norm(normal)
                if length and np.max((normalized-va)@(normal/length))>np.linalg.norm(np.ptp(normalized,axis=0))*1e-4:raise RuntimeError('Runtime float triangulation is not convex')
            parts.append(dict(link=row['link'],vertices=[dict(zip('xyz',v)) for v in vertices],triangles=faces.ravel().tolist()));canonical.append((row['link'],part))
    config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number');settings['toe_contact']=None
    config['rcareworld'].update(executable='scripts/run_compound_reaching_player.sh',graphics=True,port=15970,log_level=0);config['assets']['output_dir']=str(OUT/'native-assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
    pattern=next(p for p in read_patterns(Path(workbook)) if p.number==3)
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),OUT/'native-assets/sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario);initialize_pattern(env,pattern,**settings);attr=env.sock_cloth
        attr._send_data('InstallSockRigidSkinContact',env.robot.id,0);env._env.step(2,simulate=False)
        attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False);before=copy.deepcopy(attr.data['sock_arm_reaching_scene']);before['robot_data']=json.loads(json.dumps(env.robot.data,default=lambda a:a.tolist()))
        k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',before);frames=k.fk(k.start)
        (OUT/'native-compound-payload.json').write_text(json.dumps(dict(parts=parts,cookUnitScale=100)))
        attr._send_data('InstallSockCompoundGeometry',env.robot.id,json.dumps(dict(parts=parts,cookUnitScale=100)));env._env.step(3,simulate=False)
        native=copy.deepcopy(attr.data['sock_compound_geometry']);(OUT/'native-compound.json').write_text(json.dumps(native));checks=[]
        for (name,part),row in zip(canonical,native['parts']):
            T=frames[name];local=np.array(part['vertices']);expected=(local@T[:3,:3].T+T[:3,3])@k.A.T+k.b;actual=np.array(row['vertices_world']);obi=np.array(row['obi_vertices_world'])
            vertex_error=max(cKDTree(expected).query(actual)[0].max(),cKDTree(actual).query(expected)[0].max());obi_error=max(cKDTree(expected).query(obi)[0].max(),cKDTree(obi).query(expected)[0].max())
            world=trimesh.Trimesh(expected,part['faces'],process=True);queries=np.array([r['query_world'] for r in row['physx_closest_probes']]);observed=np.array([r['closest_world'] for r in row['physx_closest_probes']]);distance=np.linalg.norm(queries-observed,axis=1);expected_dist=trimesh.proximity.closest_point(world,queries)[1];cook_error=float(abs(distance-expected_dist).max());explicit=np.array([r['explicit_closest_world'] for r in row['physx_closest_probes']]);static=np.array([r['static_closest_world'] for r in row['physx_closest_probes']]);explicit_error=float(abs(np.linalg.norm(queries-explicit,axis=1)-expected_dist).max());static_error=float(abs(np.linalg.norm(queries-static,axis=1)-expected_dist).max())
            ray_probes=row['physx_face_ray_probes'];ray_error=max(abs(v['distance_m']-.01) if v['hit'] else 1. for v in ray_probes)
            planes=ConvexHull(expected).equations;radial_errors=[]
            for probe in row['physx_radial_probes']:
                o=np.array(probe['origin_world']);direction=np.array(probe['direction_world']);denom=planes[:,:3]@direction;offset=planes[:,:3]@o+planes[:,3];enter=max(-offset[denom< -1e-12]/denom[denom< -1e-12]);radial_errors.append(abs(probe['distance_m']-enter) if probe['hit'] else 1.)
            radial_error=float(max(radial_errors))
            checks.append(dict(physx_radial_ray_error_m=radial_error,physx_radial_ray_probes=len(radial_errors),physx_vertex_containment_error_m=row['physx_vertex_containment_error_m'],physx_explicit_closest_error_m=explicit_error,physx_static_closest_error_m=static_error,physx_ray_error_m=float(ray_error),physx_ray_probes=len(ray_probes),link=name,vertex_error_m=float(vertex_error),obi_vertex_error_m=float(obi_error),physx_probe_distance_error_m=cook_error,physx_probes=len(queries),convex=row['convex'],obi_enabled=row['obi_enabled']))
        attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False);(OUT/'native-scene.json').write_text(json.dumps(attr.data['sock_arm_reaching_scene']))
        assert len(checks)==len(canonical)
        result=dict(passed=all(r['vertex_error_m']<2e-5 and r['obi_vertex_error_m']<2e-5 and r['physx_radial_ray_error_m']<1e-4 and r['convex'] and r['obi_enabled'] for r in checks),parts=checks,physics_executed=False,robot_motion_executed=False,skipped_float_degenerate_parts=skipped,physx_gate='Source/Obi vertices and external rays: 96 sphere directions plus every input-vertex direction; ClosestPoint and tiny face-centroid probes retained as diagnostics')
        (OUT/'native-validation.json').write_text(json.dumps(result,indent=2));print('parts',len(checks),'passed',result['passed'],flush=True)
if __name__=='__main__':
    try:main()
    except Exception as error:
        (OUT/'native-validation.json').write_text(json.dumps(dict(passed=False,error=str(error),physics_executed=False),indent=2))
        raise
