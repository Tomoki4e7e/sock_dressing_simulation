#!/usr/bin/env python3
"""Measure every screened candidate using a kinematic native hand and free Obi band."""
import argparse,copy,json,sys,time,os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
 os.environ[key]=os.environ.get("SOCK_CLOSURE_BLAS_THREADS","1")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.pose_patterns import initialize_pattern,read_patterns
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.grasp_geometry import RightHand,HANDEDNESS
MEASURE_ROOT=ROOT/os.environ.get('SOCK_CLOSURE_OUTPUT','artifacts/grasp-closure-measurements')
ROS_UNITY=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])

def analyze(data,rotation=None):
 rotation=np.eye(3) if rotation is None else rotation
 rows=[s for s in data['samples'] if 'pieces' in s];jaw=lambda s,name:next(p for p in s['pieces'] if p['link'].endswith('/'+name+'_finger_link'))
 bilateral=[s for s in rows if s.get('bilateral_physics_steps',int(all(jaw(s,k)['active_events']>0 for k in ('left','right'))))>0]
 holds=[s for s in rows if s['phase'] in ('hold','pull','post_hold')]
 supported=[s for s in holds if all(jaw(s,k)['active_events']>0 for k in ('left','right'))]
 hold_both_fraction=sum(s.get('bilateral_physics_steps',int(s in supported)) for s in holds)/sum(s.get('observation_physics_steps',1) for s in holds) if holds else None
 slip=None
 if supported:
  start=supported[0];baseline=np.array(start['target_relative_tcp_world']);slip=max(float(np.linalg.norm(np.array(s['target_relative_tcp_world'])-baseline)) for s in holds if s['time_s']>=start['time_s'])
 tangential={};ids=data['band_particle_indices']
 for name in ('left','right'):
  baseline={};shifts={};observations={}
  for sample in holds:
   positions=dict(zip(ids,np.asarray(sample['band_positions_world'])))
   for particle in jaw(sample,name)['active_particle_ids']:
    if particle not in positions:continue
    relative=(positions[particle]-np.asarray(sample['tcp_world']))@rotation
    if particle not in baseline:baseline[particle]=relative
    shifts[particle]=max(shifts.get(particle,0.),float(np.linalg.norm((relative-baseline[particle])[[0,2]])))
    observations[particle]=observations.get(particle,0)+1
  values=[shifts[p] for p in shifts if observations[p]>=3]
  tangential[name]=dict(tracked_material_particles=len(values),maximum_tangential_drift_m=max(values) if values else None,
    median_tangential_drift_m=float(np.median(values)) if values else None)
 return dict(closing_stop_reason=data.get('closing_stop_reason'),closing_stop_gap_m=data.get('closing_stop_gap_m'),tangential_contact_material_drift=tangential,abort_reason=data['abort_reason'],sample_count=len(rows),bilateral_contact_frames=len(bilateral),
  first_bilateral_gap_m=data.get('first_bilateral_gap_m',bilateral[0]['gap_m'] if bilateral else None),
  hold_bilateral_fraction=hold_both_fraction,target_material_relative_drift_m=slip,
  maximum_band_speed_m_s=max((s['maximum_band_speed_m_s'] for s in rows),default=None),
  provisional_contact_persistence_pass=data['abort_reason'] is None and hold_both_fraction is not None and hold_both_fraction>=.8,
  capture_or_retention_success_validated=False,
  slip_is_material_target_drift_not_proven_interfacial_slip=True)

def trial(args):
 n,limit,begin=args;out=MEASURE_ROOT/f'pose-{n:02d}';out.mkdir(parents=True,exist_ok=True)
 screen=json.loads((ROOT/f'artifacts/grasp-cloth-refilter/pose-{n:02d}/report.json').read_text());candidates=screen['accepted'][begin:begin+limit] if limit else screen['accepted'][begin:]
 if not candidates:return dict(number=n,candidate_count=0,status='no_screened_candidates',trials=[])
 config=load_config(ROOT/'config/pose_patterns_toe_contact.yaml');settings=config.pop('pose_pattern');workbook=settings.pop('workbook');settings.pop('number')
 config['rcareworld'].update(executable='scripts/run_closure_player.sh',graphics=False,port=16000+n*500+begin*2+int(os.environ.get('SOCK_CLOSURE_PORT_OFFSET','0')),log_level=0);config['assets']['output_dir']=str(out/f'assets-{begin:03d}')
 pattern=next(p for p in read_patterns(Path(workbook)) if p.number==n);scenario=scenario_from_config(config,seed=0);prepared=_prepare(config,scenario)
 prior=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{n:02d}/report.json').read_text());hand=RightHand(ROOT/prior['settings']['urdf']);rows=[]
 with SockDressingEnv(config) as env:
  env.load(Path(prepared['runtime_urdf']),out/f'assets-{begin:03d}/sock.obj',initial_joints=scenario.initial_joints);env.apply_scenario(scenario)
  # Exact same explicit physics loop in the Player, batched to avoid one network roundtrip per step.
  def advance(count):
   if count:env._env.Simulate(.02,int(count));env._env.step(simulate=False)
  env.advance_physics=advance
  for index,c in enumerate(candidates,start=begin):
   path=out/f'candidate-{index:03d}.json'
   if path.exists():
    previous=json.loads(path.read_text());rows.append(previous['analysis']);continue
   preparation=initialize_pattern(env,pattern,**settings);attr=env.sock_cloth
   R=np.asarray(c['rotation_matrix']);L=R@HANDEDNESS;p=np.asarray(c['grasp_center_unity_world_m']);negative_shift=float(os.environ.get('SOCK_CLOSURE_NEGATIVE_SHIFT_M','0'));p=p-negative_shift*np.asarray(prior['samples'][c['sample_index']]['normal']);base=p-L@hand.grasp_center_urdf
   pieces=[]
   for link,_ in hand.meshes.values():
    a=hand.link_transform(link,hand.joint_for_gap(c['gap_m']));b=hand.link_transform(link,hand.joint_for_gap(0));quat=Rotation.from_matrix(L@a[:3,:3]@ROS_UNITY.T).as_quat()
    pieces.append(dict(link=link,start=dict(zip('xyz',base+L@a[:3,3])),end=dict(zip('xyz',base+L@b[:3,3])),rotation=dict(zip('xyzw',quat))))
   sample=prior['samples'][c['sample_index']];face=np.asarray(attr.data['particle_surface_triangles']).reshape(-1,3)[sample['face']].tolist()
   payload=dict(progressPath=str(out/f'native-progress-{begin:03d}.csv'),label=f'{n}-{index}',parts=pieces,openingGap=c['gap_m'],bilateralStopSteps=5,closingSpeed=.005,holdSeconds=2.,pullSeconds=1.,postHoldSeconds=1.,pullDistance=.005,
       tcp=dict(zip('xyz',p)),closingAxis=dict(zip('xyz',R[:,1])),pullAxis=dict(zip('xyz',-R[:,0])),face=face,weights=dict(zip('xyz',sample['weights'])))
   attr._send_data('RunSockClosureMeasurement',json.dumps(payload));env._env.step(simulate=False)
   data=copy.deepcopy(attr.data['sock_closure_measurement']);analysis=dict(number=n,candidate_index=index,**analyze(data,R))
   attr.request_particles();env._env.step(simulate=False)
   final=attr.particles().copy()
   edges=np.asarray(attr.data['particle_edges']).reshape(-1,2);rest=np.asarray(attr.data['particle_rest_edge_lengths']);good=rest>1e-9
   strain=np.linalg.norm(final[edges[:,1]]-final[edges[:,0]],axis=1)[good]/rest[good]
   analysis['final_edge_stretch_ratio_max']=float(strain.max());analysis['final_edge_stretch_ratio_p95']=float(np.quantile(strain,.95))
   result=dict(negative_control_shift_m=negative_shift,job_worker_count=int(os.environ.get('SOCK_CLOSURE_WORKERS','2')),player='ClosureMeasureV5',protocol_version=5,final_particles_world=final.tolist(),initial_particles_world=env._toe_contact_debug['points'],triangles=attr.data['particle_surface_triangles'],candidate=c,protocol=payload,initial_toe_checks=preparation['toe_contact']['checks'],measurement=data,analysis=analysis)
   path.write_text(json.dumps(result,indent=2)+'\n');rows.append(analysis);print(json.dumps(analysis),flush=True)
 result=dict(number=n,candidate_count=len(candidates),trials=rows);(out/f'job-{begin:03d}-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 import shutil
 shutil.rmtree(out/f'assets-{begin:03d}',ignore_errors=True)
 return result

def safe_trial(arguments):
 try:return trial(arguments)
 except Exception as error:
  import traceback
  n,limit,begin=arguments;out=MEASURE_ROOT/f'pose-{n:02d}';out.mkdir(parents=True,exist_ok=True)
  (out/f'error-{begin:03d}.txt').write_text(traceback.format_exc())
  print(f'FAILED pose {n} candidate {begin}: {error}',flush=True)
  return dict(number=n,candidate_count=0,trials=[],error=str(error),candidate_index=begin)

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=0);p.add_argument('--number',type=int);p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if a.number is not None:trial((a.number,a.limit,a.start));return
 source=json.loads((ROOT/'artifacts/grasp-cloth-refilter/summary.json').read_text())
 jobs=[(r['number'],1,i) for r in source['trials'] for i in range(min(a.limit,r['counts'].get('static_accepted',0)) if a.limit else r['counts'].get('static_accepted',0))]
 cached=[];pending=[]
 for job in jobs:
  n,_,i=job;path=MEASURE_ROOT/f'pose-{n:02d}/candidate-{i:03d}.json'
  if path.exists():
   saved=json.loads(path.read_text())
   if saved.get('protocol_version')!=5:raise RuntimeError(f'Wrong measurement version: {path}')
   cached.append(dict(number=n,candidate_count=1,trials=[saved['analysis']]))
  else:pending.append(job)
 with ProcessPoolExecutor(max_workers=a.workers) as pool:completed=cached+list(pool.map(safe_trial,pending))
 rows=[]
 for source_row in source['trials']:
  n=source_row['number'];results=[r for job in completed if job['number']==n for r in job['trials']]
  rows.append(dict(number=n,candidate_count=len(results),status='completed' if results else 'no_screened_candidates',trials=results))
 out=MEASURE_ROOT;out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(dict(poses=rows,completed_candidates=sum(r['candidate_count'] for r in rows),errors=[r for r in completed if 'error' in r]),indent=2)+'\n')
if __name__=='__main__':main()
