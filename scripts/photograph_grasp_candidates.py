#!/usr/bin/env python3
"""Photograph static native hand meshes against the settled sock in Unity."""
import argparse,copy,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.grasp_geometry import RightHand,HANDEDNESS,SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
import yaml
ROS_UNITY=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])
def unit(v):return v/np.linalg.norm(v)
def photograph(env,path,target,direction,up,distance,fov):
 camera=env.recording_camera;pos=target+unit(direction)*distance;forward=unit(target-pos);right=unit(np.cross(up,forward));vertical=np.cross(forward,right)
 camera.SetPosition(pos.tolist());camera.SetRotationQuaternion(Rotation.from_matrix(np.column_stack([right,vertical,forward])).as_quat().tolist())
 env._env.step(simulate=False)
 for _ in range(2):camera.GetRGB(1400,1100,fov);env._env.step(simulate=False)
 picture=env._decode(camera.data['rgb'],'RGB');Image.fromarray(picture).save(path)
 return dict(path=str(path.relative_to(ROOT)),position=pos.tolist(),target=target.tolist(),fov_deg=fov)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--refiltered',action='store_true');parser.add_argument('--hold-seconds',type=float,default=0);parser.add_argument('--port',type=int,default=15700);args=parser.parse_args()
 out=ROOT/('artifacts/grasp-refiltered-photos' if args.refiltered else 'artifacts/grasp-candidate-photos');out.mkdir(parents=True,exist_ok=True)
 config=load_config(ROOT/'config/pose_patterns_toe_contact.yaml');settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number')
 config['rcareworld'].update(executable='Build/SockDressingPlayer/GraspPreviewV1/Player.x86_64',graphics=True,port=args.port,log_level=0);config['assets']['output_dir']=str(out/'assets')
 pattern=next(p for p in read_patterns(Path(workbook)) if p.number==1);scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
 previous=json.loads((ROOT/'artifacts/right-grasp-geometry/pose-01/report.json').read_text());hand=RightHand(ROOT/previous['settings']['urdf'])
 choices=[('A-cuff-10mm',9,-30,300,.010),('B-depth5-5mm',41,-30,300,.005),('C-cuff-steep-10mm',9,-60,300,.010),('D-cuff-clear-50mm',8,-60,60,.050)]
 selected={}
 if args.refiltered:
  screened=json.loads((ROOT/'artifacts/grasp-cloth-refilter/pose-01/report.json').read_text())
  choices=[]
  for label,sample,yaw,gap in [('E-clear-30mm',9,330,.030),('F-clear-40mm',10,300,.040),('G-clear-50mm',8,60,.050)]:
   row=next(c for c in screened['accepted'] if c['sample_index']==sample and c['yaw_deg']==yaw and c['gap_m']==gap and c['closing_sign']==1 and c['straight_approach_clear'])
   selected[label]=row;choices.append((label,sample,row['tilt_deg'],yaw,gap))
 result=dict(number=1,pitch_deg=previous['pitch_deg'],roll_deg=previous['roll_deg'],static_collision_mesh_preview=True,robot_ik_or_grasp_executed=False,candidates=[])
 with SockDressingEnv(config) as env:
  env.load(Path(prepared['runtime_urdf']),out/'assets/sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario)
  preparation=initialize_pattern(env,pattern,**settings);attr=env.sock_cloth;source=env._toe_contact_debug['source'];skin=SkinBody(source['skin_surface'])
  attr._send_data('SetSockInspectionMaterial',True,env.robot.id);env._env.step(simulate=False)
  for name,sample_id,tilt,yaw,gap in choices:
   directory=out/name;directory.mkdir(exist_ok=True)
   c=next(c for c in previous['candidates'] if c['sample_index']==sample_id and c['tilt_deg']==tilt and c['yaw_deg']==yaw and c['closing_sign']==1)
   R=np.asarray(c['rotation_matrix']);L=R@HANDEDNESS;p=np.asarray(c['grasp_center_unity_world_m']);base=p-L@hand.grasp_center_urdf;q=hand.joint_for_gap(gap);parts=[];expected={}
   for link,raw in hand.meshes.values():
    T=hand.link_transform(link,q);pos=base+L@T[:3,3];quat=Rotation.from_matrix(L@T[:3,:3]@ROS_UNITY.T).as_quat()
    parts.append(dict(link=link,position=dict(zip('xyz',pos)),rotation=dict(zip('xyzw',quat))))
    expected[link]=(raw@T[:3,:3].T+T[:3,3])@L.T+base
   payload=json.dumps(dict(label=name,parts=parts));attr._send_data('SetSockGraspPreview',payload);env._env.step(simulate=False)
   readback=copy.deepcopy(attr.data['sock_grasp_preview']);error=max(float(cKDTree(expected[r['link']]).query(np.asarray(r['vertices_world']))[0].max()) for r in readback['parts'])
   if error>1e-5:raise RuntimeError(f'Preview mesh mismatch {error}')
   clear,distance,limiting,reason=skin.inspect(hand.bank(gap),R,p,margin=.001)
   live_screen=None
   if args.refiltered:
    attr.request_particles();env._env.step(simulate=False)
    live_screen=ClothScreen(attr.particles(),attr.data['particle_surface_triangles']).inspect(hand,gap,R,p,**yaml.safe_load((ROOT/'config/grasp_cloth_screen.yaml').read_text()))
    if not clear or not live_screen['passed']:raise RuntimeError(f'Live candidate rejected: {name}: {live_screen}')
   images=[]
   views=[('overview',R[:,2]+R[:,0]*.25,p-R[:,0]*.08,.48,48),('gap-side',R[:,2]+R[:,0]*.10,p,.16,35),('gap-opposite',-R[:,2]+R[:,0]*.10,p,.16,35),('tip',R[:,0]+R[:,2]*.25,p,.18,35),('oblique',R[:,2]+R[:,0]*.65+R[:,1]*.25,p,.20,35)]
   for view,direction,target,d,fov in views:images.append(photograph(env,directory/(view+'.png'),target,direction,R[:,1],d,fov))
   row=dict(name=name,refiltered_candidate=selected.get(name),live_cloth_screen=live_screen,candidate=c,gap_m=gap,sample=previous['samples'][sample_id],native_preview_geometry_error_m=error,skin_clear=clear,skin_clearance_m=distance,limiting_part=limiting,images=images,placements=parts)
   (directory/'report.json').write_text(json.dumps(row,indent=2)+'\n');result['candidates'].append(row);print(name,'skin clear',clear,'error',error,flush=True)
  result['preparation']=preparation
  attr.request_particles();env._env.step(simulate=False)
  (out/'state.json').write_text(json.dumps(dict(particles_world=attr.particles().tolist(),source=source,triangles=attr.data['particle_surface_triangles']),indent=2)+'\n')
  if args.hold_seconds>0:
   print(f'Keeping final candidate visible for {args.hold_seconds} seconds',flush=True);time.sleep(args.hold_seconds)
  attr._send_data('RemoveSockGraspPreview');attr._send_data('SetSockInspectionMaterial',False,env.robot.id);env._env.step(simulate=False)
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
