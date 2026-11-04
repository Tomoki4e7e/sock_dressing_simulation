#!/usr/bin/env python3
"""Validate every installed input/Obi shape and PhysX ray in the real pose."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audit_compound_rays import ray_interval
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.arm_reaching import Kinematics,ROS_UNITY
from sock_dressing_simulation.convex_distance import point_convex_distance


def check_part(row,shape,cooked,k):
    v=np.array([[v[a] for a in 'xyz'] for v in shape['vertices']]);scale=np.array([shape['scale'][a] for a in 'xyz']);position=np.array([shape['position'][a] for a in 'xyz']);R=Rotation.from_quat([shape['rotation'][a] for a in 'xyzw']).as_matrix();local=((v*scale)@R.T+position)@ROS_UNITY
    T=k.fk(k.start,only=[shape['link']])[shape['link']];expected=(local@T[:3,:3].T+T[:3,3])@k.A.T+k.b;actual=np.array(row['vertices_world']);obi=np.array(row['obi_vertices_world']);vertex_error=float(np.linalg.norm(expected-actual,axis=1).max());obi_error=float(np.linalg.norm(expected-obi,axis=1).max())
    matrix=np.array(row['matrix']);world=v@matrix[:3,:3].T+matrix[:3,3];hull=ConvexHull(world);rays=row['rays'];points=np.array([r['point'] for r in rays if r['hit']]);distances=point_convex_distance(points,world,surface=True) if len(points) else np.array([np.inf]);expected_misses=0;misses=0
    for ray in rays:
        interval=ray_interval(hull.equations,np.array(ray['origin']),np.array(ray['direction']))
        expected_misses+=interval is None;misses+=not ray['hit']
    boundary=float(distances.max());plane_error=cooked['serialized_plane_hull_hausdorff_m'];cook=max(cooked['cooked_vertex_hull_hausdorff_m'],plane_error if plane_error is not None else np.inf);geometry_bound=cook+max(vertex_error,obi_error)
    passed=bool(row['convex'] and row['obi_enabled'] and cooked['serialized_plane_hull_closed'] and geometry_bound<1e-4 and vertex_error<2e-5 and obi_error<2e-5 and misses==0 and expected_misses==0 and boundary+vertex_error<1e-4)
    return dict(index=row['index'],parent_index=shape['parent_index'],link=shape['link'],passed=passed,planner_input_vertex_error_m=vertex_error,planner_obi_vertex_error_m=obi_error,cook_boundary_error_m=cook,combined_geometry_bound_m=geometry_bound,rays=len(rays),misses=misses,input_ray_misses=expected_misses,maximum_hit_to_input_surface_m=boundary,contact_offset_m=row['contact_offset_m'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--registration',type=Path,default=ROOT/'artifacts/compound-reaching/registration/candidate');ap.add_argument('--port',type=int,default=16010);ap.add_argument('--player',default='scripts/run_registered_reaching_player.sh');args=ap.parse_args();folder=args.registration;payload=json.loads((folder/'payload.json').read_text());comparison=json.loads((folder/'comparison.json').read_text());cooked={p['index']:p for p in comparison['parts']}
    if any(not p['serialized_plane_hull_closed'] or max(p['cooked_vertex_hull_hausdorff_m'],p['serialized_plane_hull_hausdorff_m'])>=1e-4 for p in comparison['parts']):raise ValueError('Serialized cook gate has not passed')
    out=folder/'native-validation';out.mkdir(exist_ok=True);config=load_config(ROOT/'config/arm_reaching_safe_start.yaml');config.pop('pose_pattern',None);config['rcareworld'].update(executable=args.player,graphics=True,port=args.port,log_level=0);config['assets']['output_dir']=str(out/'assets');scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario);reports=[]
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']),out/'assets/sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario);attr=env.sock_cloth;attr._send_data('GetSockArmReachingScene',env.robot.id);env._env.step(2,simulate=False);scene=copy.deepcopy(attr.data['sock_arm_reaching_scene']);scene['robot_data']=json.loads(json.dumps(env.robot.data,default=lambda a:a.tolist()));k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',scene)
        attr._send_data('InstallSockCompoundGeometry',env.robot.id,json.dumps(payload));env._env.step(3,simulate=False);install=attr.data['sock_compound_geometry'];assert install['part_count']==len(payload['parts'])
        for start in range(0,len(payload['parts']),32):
            count=min(32,len(payload['parts'])-start);attr._send_data('ReadSockRegisteredCompound',start,count,True);env._env.step(2,simulate=False);raw=copy.deepcopy(attr.data['sock_registered_compound']);(out/f'batch-{start:05d}.json').write_text(json.dumps(raw))
            assert [p['index'] for p in raw['parts']]==list(range(start,start+count))
            reports.extend(check_part(p,payload['parts'][p['index']],cooked[p['index']],k) for p in raw['parts']);print('native validated',len(reports),'/',len(payload['parts']),'failures',sum(not p['passed'] for p in reports),flush=True)
    result=dict(passed=all(p['passed'] for p in reports) and len(reports)==len(payload['parts']),parts=reports,part_count=len(reports),ray_count=sum(p['rays'] for p in reports),misses=sum(p['misses'] for p in reports),input_ray_misses=sum(p['input_ray_misses'] for p in reports),maximum_combined_geometry_bound_m=max(p['combined_geometry_bound_m'] for p in reports),maximum_hit_to_input_surface_m=max(p['maximum_hit_to_input_surface_m'] for p in reports),payload_sha256=hashlib.sha256((folder/'payload.json').read_bytes()).hexdigest(),physics_executed=False,robot_motion_executed=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('NATIVE GATE',result['passed'],'rays',result['ray_count'],'misses',result['misses'],flush=True)


if __name__=='__main__':main()
