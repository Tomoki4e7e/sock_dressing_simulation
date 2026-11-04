#!/usr/bin/env python3
"""Summarize the saved translation experiment without changing the simulator."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'artifacts/translated-reaching'
rows=[json.loads(p.read_text()) for p in sorted(out.glob('target-*.json'))];good=lambda a:a['position_error_m']<=.0001 and a['orientation_error_deg']<=.5
fig,ax=plt.subplots(1,2,figsize=(10,4));groups={}
for label,source,color in [('Original cuff','geometric-closure','tab:blue'),('Bulged cuff','bulged-cuff','tab:orange')]:
 group=[r for r in rows if source in r['source']];valid=[a for r in group for a in r['attempts'] if good(a)];groups[label]=dict(targets=len(group),trials=sum(len(r['attempts']) for r in group),targets_with_ik=sum(any(good(a) for a in r['attempts']) for r in group),ik_trials=len(valid));d=np.array([a['translation_m'] for a in valid])*1000
 ax[0].scatter(d[:,0],d[:,2],label=label,s=12,alpha=.5,color=color)
 ax[1].scatter(d[:,0],d[:,1],label=label,s=12,alpha=.5,color=color)
for a in ax:a.set_xlabel('World X translation [mm]');a.grid(alpha=.2);a.legend()
ax[0].set_ylabel('World Z translation [mm]');ax[1].set_ylabel('World Y translation [mm]');fig.suptitle('Endpoint IK solutions with >=5 deg joint margin; all fail collision gate');fig.tight_layout();fig.savefig(out/'translation-ik-map.png',dpi=160);(out/'per-source-summary.json').write_text(json.dumps(groups,indent=2))
