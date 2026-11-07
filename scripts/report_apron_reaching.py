#!/usr/bin/env python3
"""Summarize fresh-apron reruns and export only offline-passing joint profiles."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-apron-matplotlib')
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/apron-reaching'

def main():
 phases={}
 for name in ['search','connected/search']:
  rows=[json.loads(p.read_text()) for p in (OUT/name).glob('*.json')];a=[v for r in rows for v in r['attempts']];phases[name]=dict(targets=len(rows),trials=len(a),endpoint_and_start_passes=sum(v['endpoint_and_start_passed'] for v in a),sampled_route_passes=sum(v.get('trajectory',{}).get('passed',False) for v in a))
 accepted=[];rejected=[]
 for path in sorted((OUT/'routes').glob('*-validation.json')):
  r=json.loads(path.read_text());plan_path=path.with_name(path.name.replace('-validation.json','-trajectory.json'));plan=json.loads(plan_path.read_text());record=dict(number=r['number'],validation=str(path.relative_to(ROOT)),trajectory=str(plan_path.relative_to(ROOT)),translation_m=plan['translation_m'],duration_s=plan['duration_s'],sample_count=r['sample_count'],minimum_joint_margin_deg=r['minimum_joint_margin_deg'],joint_margin_policy=r['joint_margin_policy'],opening_gap_m=plan['opening_gap_m'])
  if not r['offline_passed']:rejected.append(record);continue
  accepted.append(record);table=plan_path.with_suffix('.csv')
  with table.open('w',newline='') as f:
   writer=csv.writer(f);writer.writerow(['time_s']+[f'{name}_deg' for name in plan['joint_names']]);writer.writerows([[t]+list(np.rad2deg(q)) for t,q in zip(plan['times_s'],plan['q_rad'])])
  record['csv']=str(table.relative_to(ROOT))
 if accepted:
  preferred=min(accepted,key=lambda a:a['number']);plan=json.loads((ROOT/preferred['trajectory']).read_text());t=np.array(plan['times_s']);fig,axes=plt.subplots(1,2,figsize=(11,4))
  axes[0].plot(t,np.rad2deg(plan['q_rad']));axes[0].legend([f'J{i}' for i in range(1,8)],ncol=2,fontsize=8);axes[0].set_ylabel('Joint angle [degrees]');axes[1].plot(t,np.array(plan['tcp_world_m'])*1000);axes[1].legend(['X','Y','Z']);axes[1].set_ylabel('TCP world position [mm]')
  for ax in axes:ax.set_xlabel('Time [s]');ax.grid(alpha=.2)
  fig.suptitle(f"No.{preferred['number']}: apron start, offline checked; not executed");fig.tight_layout();fig.savefig(OUT/'trajectory-profile.png',dpi=170)
 barrier=json.loads((OUT/'shoulder-barrier.json').read_text());summary=dict(phases=phases,offline_passing_routes=accepted,rejected_smoothed_routes=rejected,distinct_passing_pose_numbers=sorted(set(r['number'] for r in accepted)),barrier=barrier,native_geometry_passed=False,executed=False,continuous_collision_certified=False,initial_joint4_margin_deg=4,approach_and_goal_margin_deg=5)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('offline passing',summary['distinct_passing_pose_numbers'],'phase counts',phases)
if __name__=='__main__':main()
