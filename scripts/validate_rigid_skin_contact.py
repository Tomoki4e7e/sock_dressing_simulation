#!/usr/bin/env python3
"""Validate baked rigid skin and native hand queries on prepared ankle poses."""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.grasp_geometry import RightHand,HANDEDNESS
ROS_UNITY=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])
def summarize():
 directory=ROOT/'artifacts/rigid-skin-contact'
 trials=json.loads((ROOT/'artifacts/toe-contact-final/summary.json').read_text())['trials'];rows=[]
 for t in trials:
  n=t['number'];path=directory/f'pose-{n:02d}'/'report.json';r=json.loads(path.read_text())
  prior=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{n:02d}/report.json').read_text());state=json.loads((ROOT/prior['input_state']).read_text())
  skin=r['preparation']['rigid_skin_contact'];final=r['preparation']['toe_contact']['final']
  skin_error=float(np.linalg.norm(np.asarray(skin['vertices_world'])-np.asarray(state['source']['skin_surface']['vertices']),axis=1).max())
  band_error=float(np.linalg.norm(np.asarray(final['band_positions_world'])-np.asarray(state['particles_world'])[final['band_particle_indices']],axis=1).max())
  same=skin_error<1e-6 and band_error<1e-6 and np.array_equal(skin['triangles'],state['source']['skin_surface']['triangles'])
  rows.append(dict(number=n,pitch_deg=t['pitch_deg'],roll_deg=t['roll_deg'],passed=r['passed'],
   report=str(path.relative_to(ROOT)),skin_error_vs_search_m=skin_error,band_error_vs_search_m=band_error,
   prior_search_geometry_unchanged=bool(same),reusable_skin_clear_candidates=r['skin_clear_candidates'] if same else None,
   native_hand_geometry_error_m=max(p['maximum_geometry_error_m'] for q in r['queries'] for p in q['parts']),
   physical_probe_contacts=r['probe']['contacts'],physical_probe_maximum_impulse=r['probe']['maximum_impulse'],
   contact_offset_m=skin['contact_offset_m'],right_gripper_pairs_enabled=skin['right_gripper_pairs_enabled']))
 result=dict(player='Build/SockDressingPlayer/RigidSkinV1/Player.x86_64',passed=len(rows)==32 and all(r['passed'] for r in rows),unique_angle_count=len(rows),
  maximum_skin_readback_error_m=max(json.loads((ROOT/r['report']).read_text())['preparation']['rigid_skin_contact']['maximum_skin_vertex_error_m'] for r in rows),
  maximum_skin_error_vs_search_m=max(r['skin_error_vs_search_m'] for r in rows),maximum_band_error_vs_search_m=max(r['band_error_vs_search_m'] for r in rows),
  prior_search_geometry_unchanged=all(r['prior_search_geometry_unchanged'] for r in rows),
  reusable_skin_clear_candidates=sum(r['reusable_skin_clear_candidates'] or 0 for r in rows),
  native_candidate_queries=sum(len(json.loads((ROOT/r['report']).read_text())['queries']) for r in rows),
  cloth_contact_solver='unchanged convex support and friction',robot_ik_path_or_grasp_success_validated=False,trials=rows)
 contact=directory/'cloth-contact-response/report.json'
 if contact.exists():
  contact_report=json.loads(contact.read_text());result['cloth_contact_response_passed']=contact_report['passed'];result['cloth_contact_response_report']=str(contact.relative_to(ROOT))
 (directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--summarize',action='store_true');p.add_argument('--all',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--number',type=int,default=1);p.add_argument('--port',type=int,default=14900);a=p.parse_args()
 if a.summarize:
  summarize();return
 if a.all:
  import concurrent.futures,subprocess,shutil
  directory=ROOT/'artifacts/rigid-skin-contact';directory.mkdir(parents=True,exist_ok=True)
  trials=json.loads((ROOT/'artifacts/toe-contact-final/summary.json').read_text())['trials']
  def run(pair):
   i,t=pair;n=t['number']
   with (directory/f'pose-{n:02d}.log').open('w') as log:
    status=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--number',str(n),'--port',str(a.port+3*i)],stdout=log,stderr=log)
   shutil.rmtree(directory/f'pose-{n:02d}'/'assets',ignore_errors=True)
   print(f"pose {n}: exit {status.returncode}",flush=True)
   return dict(number=n,exit_code=status.returncode)
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(run,enumerate(trials)))
  (directory/'run-summary.json').write_text(json.dumps(results,indent=2)+'\n')
  if any(r['exit_code'] for r in results):raise SystemExit(2)
  summarize();return
 out=ROOT/'artifacts/rigid-skin-contact'/f'pose-{a.number:02d}';out.mkdir(parents=True,exist_ok=True)
 config=load_config(ROOT/'config/pose_patterns_toe_contact.yaml');settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number')
 pattern=next(p for p in read_patterns(Path(workbook)) if p.number==a.number)
 config['rcareworld'].update(graphics=False,port=a.port,log_level=0);config['assets']['output_dir']=str(out/'assets')
 scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
 previous=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{a.number:02d}/report.json').read_text())
 hand=RightHand(ROOT/previous['settings']['urdf'])
 with SockDressingEnv(config) as env:
  env.load(Path(prepared['runtime_urdf']),out/'assets/sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario)
  report=initialize_pattern(env,pattern,**settings);attr=env.sock_cloth
  candidates=sorted([c for c in previous['candidates'] if c['native_contact_face']],key=lambda c:c['endpoint_clearance_m'],reverse=True)
  chosen=[candidates[0],min(candidates,key=lambda c:c['endpoint_clearance_m'])]
  queries=[]
  for i,c in enumerate(chosen):
   L=np.asarray(c['rotation_matrix'])@HANDEDNESS;base=np.asarray(c['gripper_base_position_unity_world_m']);parts=[]
   for link,_ in hand.meshes.values():
    T=hand.link_transform(link,c['pinch_joint_m']);pos=base+L@T[:3,3];q=Rotation.from_matrix(L@T[:3,:3]@ROS_UNITY.T).as_quat()
    parts.append(dict(link=link,position=dict(zip('xyz',pos)),rotation=dict(zip('xyzw',q))))
   attr._send_data('QuerySockRigidSkinHand',json.dumps(dict(label=str(i),parts=parts)),True);env._env.step(simulate=False)
   query=copy.deepcopy(attr.data['sock_rigid_skin_hand_query'])
   for row in query['parts']:
    link=row['link'];raw=hand.meshes[link][1];T=hand.link_transform(link,c['pinch_joint_m'])
    expected=(raw@T[:3,:3].T+T[:3,3])@L.T+base
    row['maximum_geometry_error_m']=float(cKDTree(expected).query(np.asarray(row.pop('vertices_world')))[0].max());row.pop('triangles');row.pop('old_hull_intersection');row.pop('old_hull_penetration_m')
   query.pop('old_hull_intersections');queries.append(query)
  skin=report['rigid_skin_contact'];source=env._toe_contact_debug['source'];v=np.asarray(source['skin_surface']['vertices']);tri=np.asarray(source['skin_surface']['triangles']).reshape(-1,3)
  face=v[tri];centers=face.mean(1);idx=np.argmin(np.linalg.norm(centers-np.asarray(source['toe']),axis=1));f=face[idx];normal=np.cross(f[1]-f[0],f[2]-f[0]);normal/=np.linalg.norm(normal)
  position=centers[idx]+normal*.004
  attr._send_data('StartSockRigidSkinProbe',position.tolist(),.005);env._env.step(simulate=False);env.advance_physics(10)
  attr._send_data('GetSockRigidSkinProbe');env._env.step(simulate=False);probe=copy.deepcopy(attr.data['sock_rigid_skin_probe'])
  attr._send_data('RemoveSockRigidSkinProbe');env._env.step(simulate=False)
  result=dict(number=a.number,selected_candidates=[{k:c[k] for k in ('sample_index','yaw_deg','tilt_deg','closing_sign','endpoint_clearance_m')} for c in chosen],preparation=report,queries=queries,probe=probe,skin_clear_candidates=len(candidates),passed=report['toe_contact']['passed'] and all(q['skin_intersections']==0 and all(p['maximum_geometry_error_m']<1e-5 for p in q['parts']) for q in queries) and probe['contacts']>0)
  (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(number=a.number,passed=result['passed'],queries=queries,probe=probe)),flush=True)
 if not result['passed']:raise SystemExit(2)
if __name__=='__main__':main()
