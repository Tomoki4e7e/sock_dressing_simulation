#!/usr/bin/env python3
"""Check closure data provenance, material-point identity and time accounting."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def inspect(path):
 r=json.loads(path.read_text());p=r['protocol'];m=r['measurement'];initial=np.asarray(r['initial_particles_world']);final=np.asarray(r['final_particles_world']);band=m['band_particle_indices'];rows=[s for s in m['samples'] if 'pieces' in s]
 failures=[]
 def check(value,name):
  if not value:failures.append(name)
 check(r.get('protocol_version')==5,'V5 provenance');check(all(r['initial_toe_checks'].values()),'initial QA')
 check(m['toe_support_retained'] and not m['cloth_grasp_constraints_added'],'no artificial capture constraint')
 check(len(band)==96 and np.all(np.asarray(m['band_inverse_masses'])>0),'free native band')
 check(initial.shape==final.shape and initial.shape[1]==3,'particle shape')
 weights=np.array([p['weights'][k] for k in 'xyz']);tcp=np.array([p['tcp'][k] for k in 'xyz']);target=weights@initial[p['face']]
 initial_error=float(np.linalg.norm(target-tcp));check(initial_error<5e-6,'target material point matches initial TCP')
 check(abs(m['dt_s']-.02)<1e-7 and abs(p['closingSpeed']-.005)<1e-9,'fixed protocol')
 check(len(p['parts'])==5 and p['bilateralStopSteps']==5,'hand and stop rule')
 for s in rows:
  count=s['observation_physics_steps'];both=s['bilateral_physics_steps'];check(0<=both<=count<=5,'contact step counters')
  jaws=[v for v in s['pieces'] if v['link'].endswith('/left_finger_link') or v['link'].endswith('/right_finger_link')]
  check(len(jaws)==2 and all(v['active_physics_steps']>=both for v in jaws),'simultaneous contact counters')
  check(all(0<=v['active_physics_steps']<=count and v['active_events']>=v['active_physics_steps'] and v['proximity_events']>=v['active_events'] for v in s['pieces']),'piece contact counters')
  relative=np.array(s['target_material_world'])-np.array(s['tcp_world']);check(np.linalg.norm(relative-s['target_relative_tcp_world'])<3e-6,'relative position identity')
  pull=np.array([p['pullAxis'][k] for k in 'xyz']);pull/=np.linalg.norm(pull)
  check(np.linalg.norm(np.array(s['tcp_world'])-tcp-pull*s['pull_m'])<3e-6,'hand pull identity')
 if rows and m['abort_reason'] is None:
  check(np.max(np.linalg.norm(final[band]-np.array(rows[-1]['band_positions_world']),axis=1))<5e-6,'final band equals last observation')
  check(abs(rows[-1]['time_s']/m['dt_s']-sum(s['observation_physics_steps'] for s in rows))<.01,'time accounting')
 check(not r['analysis']['capture_or_retention_success_validated'],'no unsupported success assertion')
 return dict(path=str(path.relative_to(ROOT)),passed=not failures,failures=sorted(set(failures)),initial_target_error_m=initial_error)
def main():
 p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');a=p.parse_args();out=ROOT/'artifacts/grasp-closure-measurements';rows=[inspect(path) for path in sorted(out.glob('pose-*/candidate-*.json'))]
 result=dict(passed=all(r['passed'] for r in rows) and (a.allow_partial or len(rows)==436),checked_candidates=len(rows),complete=len(rows)==436,
  maximum_initial_target_error_m=max((r['initial_target_error_m'] for r in rows),default=None),failed=[r for r in rows if not r['passed']])
 (out/'data-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 if not result['passed']:raise SystemExit(2)
if __name__=='__main__':main()
