#!/usr/bin/env python3
"""Summarize reverse-routing results and the independent shoulder-space audit."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-reverse-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/reverse-reaching'
barrier=json.loads((OUT/'shoulder-barrier.json').read_text());route=json.loads((OUT/'target-013-trial-1/result.json').read_text());waypoints=json.loads((OUT/'target-013-trial-1/waypoints.json').read_text());iks=[json.loads(p.read_text()) for p in sorted(OUT.glob('target-*-start-side-ik.json'))];endpoints=json.loads((OUT/'endpoints-triangle-validation.json').read_text())
summary=dict(found=False,executed=False,reverse_routing=route,wrist_preparation_passes=sum(r.get('passed',False) for r in waypoints if r['kind']=='wrist_preparation'),alternative_ik_family_states=[r['states'] for r in waypoints if r['kind']=='alternative_preapproach_ik'],reverse_escape_states={r['mode']:r['states'] for r in waypoints if r['kind']=='reverse_escape'},shoulder_barrier=barrier,endpoint_shoulder_pair_rechecks_passed=all(r['passed'] for r in endpoints),start_side_ik_trials=sum(len(stage['attempts']) for r in iks for stage in r['results'].values()),start_side_ik_solutions=sum(len(stage['solutions']) for r in iks for stage in r['results'].values()),native_geometry_gate_still_unresolved=True,scope='Fixed offline robot geometry, torso and 5mm clearance; not a physical-robot impossibility claim')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));data=np.load(OUT/'shoulder-slice.npz');fig,ax=plt.subplots(figsize=(8,4));ax.plot(np.rad2deg(data['joint2']),data['distance']*1000,label='Triangle-surface distance');ax.axhline(5,color='red',linestyle='--',label='Required 5 mm');ax.axhline(barrier['conservative_distance_upper_bound_m']*1000,color='orange',linestyle=':',label='Sample + motion bound + numerical allowance');ax.set_xlabel('Joint 2 [degrees]');ax.set_ylabel('Shoulder / torso distance [mm]');ax.set_title('Necessary crossing: joint 1 = 110 degrees');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'shoulder-barrier.png',dpi=170);print('barrier',barrier['barrier_confirmed'],'IK trials',summary['start_side_ik_trials'],'routes',summary['found'])
