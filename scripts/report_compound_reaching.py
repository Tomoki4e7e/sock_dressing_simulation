#!/usr/bin/env python3
"""Final experiment summary and plots, retaining failed native gates explicitly."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-compound-matplotlib')
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/compound-reaching'

def main():
 geometry=json.loads((OUT/'decomposition.json').read_text());native=json.loads((OUT/'native-validation.json').read_text());validation=json.loads((OUT/'validation.json').read_text());phases={}
 for folder in ['search','expanded/search','floor-aware/search']:
  rows=[json.loads(p.read_text()) for p in (OUT/folder).glob('target-*.json')];trials=[a for r in rows for a in r['attempts']];phases[folder]=dict(targets=len(rows),trials=len(trials),reported_endpoint_passes=sum(a['endpoint_and_start_passed'] for a in trials),floor_constraints_in_optimizer=folder=='floor-aware/search')
 coupled=[json.loads(p.read_text()) for p in (OUT/'coupled-approach').glob('*.json')];native_parts=native.get('parts',[]);raw=json.loads((OUT/'native-compound.json').read_text())
 native_summary=dict(passed=native['passed'],parts=len(native_parts),maximum_input_vertex_error_m=max((r['vertex_error_m'] for r in native_parts),default=None),maximum_obi_vertex_error_m=max((r['obi_vertex_error_m'] for r in native_parts),default=None),parts_failing_physx_100um_or_missed_ray=sum(r['physx_radial_ray_error_m']>1e-4 for r in native_parts),ray_probes=sum(r['physx_radial_ray_probes'] for r in native_parts),missed_ray_probes=sum(not p['hit'] for r in raw['parts'] for p in r.get('physx_radial_probes',[])),note='The 1.0 value in per-part radial error means at least one ray missed; it is not a measured one-metre geometric error.')
 summary=dict(phases=phases,source_geometry=geometry,native=native_summary,coupled_solutions=[dict(number=r['number'],source=r['source'],translation_m=r['translation_m'],gap_m=r['gap_m'],endpoint_and_preapproach_passed=r['endpoint_and_preapproach_passed'],trajectory_stage=r.get('trajectory',{}).get('stage')) for r in coupled],distinct_grasp_poses=len(set(r['source'] for r in coupled)),rrt_runs=validation['rrt_runs'],rrt_iterations=validation['rrt_iterations'],rrt_successes=validation['rrt_successes'],optimized_route_successes=validation['optimized_route_successes'],final_accepted=0,robot_motion_executed=False)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 fig,ax=plt.subplots(1,2,figsize=(10,4));x=np.arange(2);fractions=[r['volume_fraction_of_single_hull'] for r in geometry['parts']];ax[0].bar(x-.15,[1,1],width=.3,label='Single hull',color='lightgray');ax[0].bar(x+.15,fractions,width=.3,label='Compound union',color='tab:blue');ax[0].set_xticks(x);ax[0].set_xticklabels(['Palm','Forearm']);ax[0].set_ylabel('Volume / single-hull volume');ax[0].legend();ax[0].set_title('Concavity retained; subdivision preserves union')
 for r in coupled:
  d=np.array(r['translation_m'])*1000;ax[1].scatter(d[0],d[1],label=f"No.{r['number']}");ax[1].annotate(f"No.{r['number']}",(d[0],d[1]),xytext=(4,4),textcoords='offset points')
 ax[1].scatter([-273.33],[-300],marker='x',color='red',label='Rejected: human below floor');ax[1].axhline(-23.398,linestyle='--',color='gray',label='Whole-human floor bound');ax[1].set_xlabel('World X translation [mm]');ax[1].set_ylabel('World Y translation [mm]');ax[1].set_title('Endpoint + preapproach solutions; full routes failed');ax[1].legend(fontsize=7);ax[1].grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'diagnostics.png',dpi=160)
 print('final accepted',summary['final_accepted'],'native',native_summary,'coupled solutions',len(coupled))
if __name__=='__main__':main()
