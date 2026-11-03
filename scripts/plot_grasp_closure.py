#!/usr/bin/env python3
"""Plot the complete closure dataset; never plot partial data as final."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'artifacts/grasp-closure-measurements';summary=json.loads((out/'measurement-report.json').read_text())
 if not summary['complete']:raise RuntimeError('Complete all measurements before publishing final plots')
 rows=list(csv.DictReader((out/'measurements.csv').open()));valid=[r for r in rows if r.get('stop_gap_m')]
 fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
 ax=axes[0,0];ax.scatter([float(r['opening_gap_m'])*1000 for r in valid],[float(r['stop_gap_m'])*1000 for r in valid],s=18,alpha=.5)
 ax.plot([0,50],[0,50],color='.6',ls='--');ax.set(xlabel='Initial opening (mm)',ylabel='Opening at closure stop (mm)',title='Closure stop at bilateral contact or safety limit',xlim=(0,52),ylim=(0,52))
 held=[r for r in rows if r.get('hold_bilateral_fraction') and r.get('target_relative_drift_m')]
 ax=axes[0,1];artist=ax.scatter([float(r['hold_bilateral_fraction']) for r in held],[float(r['target_relative_drift_m'])*1000 for r in held],c=[float(r['target_in_pad_fraction']) for r in held],vmin=0,vmax=1,cmap='viridis',s=18,alpha=.7)
 ax.axvline(.8,color='.5',ls='--');ax.axhline(3,color='.5',ls='--');ax.set(xlabel='Fraction of hold physics steps with bilateral contact',ylabel='Target material drift relative to hand (mm)',title='Contact alone versus material retention',xlim=(-.03,1.03));fig.colorbar(artist,ax=ax,label='Fraction of samples with target inside finite pad window')
 ax=axes[1,0]
 for label,sample,yaw,gap in [('E / 30 mm',9,330,.03),('F / 40 mm',10,300,.04),('G / 50 mm',8,60,.05)]:
  match=next((r for r in rows if int(r['number'])==1 and r.get('sample_index') and int(r['sample_index'])==sample and int(r['yaw_deg'])==yaw and int(r['closing_sign'])==1 and abs(float(r['opening_gap_m'])-gap)<1e-8),None)
  if match is None:continue
  data=json.loads((ROOT/match['raw_report']).read_text());samples=[s for s in data['measurement']['samples'] if s['phase'] in ('hold','pull','post_hold') and 'pieces' in s]
  if not samples:continue
  axis=np.array([data['protocol']['pullAxis'][k] for k in 'xyz']);axis/=np.linalg.norm(axis);origin=np.array(samples[0]['target_material_world']);times=[s['time_s']-samples[0]['time_s'] for s in samples]
  movement=[float((np.array(s['target_material_world'])-origin)@axis)*1000 for s in samples]
  ax.plot(times,movement,label=label)
 ax.plot([0,2,3,4],[0,0,5,5],color='black',ls='--',label='Hand command');ax.axvspan(2,3,color='.8',alpha=.4);ax.set(xlabel='Time since first hold sample (s)',ylabel='Target material movement along pull (mm)',title='The three photographed candidates (same ankle state)');ax.legend(fontsize=8)
 ax=axes[1,1];poses=summary['poses'];x=np.arange(len(poses));ax.bar(x,[p['completed_candidates'] for p in poses],color='#4477aa',label='Measured configurations')
 ax.bar(x,[p['status_counts'].get('provisional_retention_proxy',0) for p in poses],color='#228833',label='Provisional retention proxy')
 ax.set_xticks(x,[f"{p['number']}\n{p['pitch_deg']:g}/{p['roll_deg']:g}" for p in poses],rotation=90,fontsize=7);ax.set(xlabel='Workbook row / ankle pitch and roll (deg)',ylabel='Configurations',title='Coverage of all 32 ankle states');ax.legend(fontsize=8)
 for ax in axes.ravel():ax.grid(alpha=.2)
 fig.suptitle(f"Closure measurement: {summary['completed_candidates']} independent configurations\nKinematic hand; toe support retained; material drift is not calibrated interfacial slip",fontsize=13)
 fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
