#!/usr/bin/env python3
"""Rescreen all 32 saved skin-clear orientation candidates at every opening."""
import argparse,json,sys,csv
import yaml
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,HANDEDNESS
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen

def trial(arguments):
    n,patch_settings=arguments
    source=ROOT/f'artifacts/right-grasp-geometry/pose-{n:02d}/report.json';prior=json.loads(source.read_text());state=json.loads((ROOT/prior['input_state']).read_text());settings=prior['settings']
    hand=RightHand(ROOT/settings['urdf']);skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);counts=Counter();accepted=[];rejected=[]
    for index,c in enumerate(prior['candidates']):
        if not c['native_contact_face']:continue
        R=np.asarray(c['rotation_matrix']);p=np.asarray(prior['samples'][c['sample_index']]['position'])
        for gap in sorted(settings['opening_gaps_m']):
            counts['tested_openings']+=1
            clear,distance,_,_=skin.inspect(hand.bank(gap),R,p,settings['body_clearance_m'])
            if not clear:counts['foot_rejected']+=1;continue
            result=cloth.inspect(hand,gap,R,p,**patch_settings)
            if not result['passed']:
                counts[result['reason']]+=1;rejected.append(dict(source_candidate_index=index,gap_m=gap,**result));continue
            # Require a cloth-free straight reach, but do not require cloth-free closure:
            # closure must be checked later with deformable cloth dynamics.
            reach_skin,reach_distance,_,_=skin.inspect(hand.bank(gap,'approach',approach=settings['approach_distance_m']),R,p,settings['body_clearance_m'])
            reach_hits=cloth.intersections(hand.bank(gap,'approach',approach=settings['approach_distance_m']),R,p)
            counts['static_accepted']+=1
            if reach_skin and not reach_hits:counts['approach_accepted']+=1
            accepted.append(dict(source_candidate_index=index,sample_index=c['sample_index'],gap_m=gap,
                yaw_deg=c['yaw_deg'],tilt_deg=c['tilt_deg'],closing_sign=c['closing_sign'],
                rotation_matrix=R.tolist(),grasp_center_unity_world_m=p.tolist(),
                gripper_base_position_unity_world_m=(p-R@HANDEDNESS@hand.grasp_center_urdf).tolist(),
                finger_joint_m=hand.joint_for_gap(gap),mimic_joint_m=hand.joint_for_gap(gap)*.5,
                skin_clearance_m=distance,cloth_screen=result,
                straight_approach_clear=bool(reach_skin and not reach_hits),approach_skin_clearance_m=reach_distance,
                approach_cloth_intersecting_parts=reach_hits))
    out=ROOT/f'artifacts/grasp-cloth-refilter/pose-{n:02d}';out.mkdir(parents=True,exist_ok=True)
    report=dict(number=n,pitch_deg=prior['pitch_deg'],roll_deg=prior['roll_deg'],source_report=str(source.relative_to(ROOT)),
        source_state=prior['input_state'],counts=dict(counts),accepted=accepted,rejected=rejected,
        finite_patch_is_sampled=True,closure_or_capture_validated=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    row={k:report[k] for k in ('number','pitch_deg','roll_deg','counts')};print(json.dumps(row),flush=True);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'config/grasp_cloth_screen.yaml');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    patch_settings=yaml.safe_load(a.config.read_text())
    if set(patch_settings)!={'half_extent_m','minimum_area_m2','edge_clearance_m'} or any(not np.isfinite(v) or v<=0 for v in patch_settings.values()):raise ValueError('Invalid pad screening settings')
    if patch_settings['minimum_area_m2']>4*patch_settings['half_extent_m']**2:raise ValueError('Required area exceeds pad window')
    trials=json.loads((ROOT/'artifacts/toe-contact-final/summary.json').read_text())['trials']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(trial,[(t['number'],patch_settings) for t in trials]))
    total=Counter()
    for r in rows:total.update(r['counts'])
    result=dict(unique_angle_count=len(rows),counts=dict(total),poses_with_static_candidates=sum(r['counts'].get('static_accepted',0)>0 for r in rows),
        poses_with_approach_candidates=sum(r['counts'].get('approach_accepted',0)>0 for r in rows),trials=rows,
        scope='Previously skin-clear orientations only; every configured opening rechecked against current skin and full cloth. Not a new full orientation grid.',
        pad_patch=dict(patch_settings,grid='5x5',area_rule='sum of grid cells whose four corners are supported'),
        closure_or_grasp_success_validated=False)
    (ROOT/'artifacts/grasp-cloth-refilter/summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (ROOT/'artifacts/grasp-cloth-refilter/per-pose-summary.csv').open('w',newline='') as stream:
        fields=['number','pitch_deg','roll_deg','tested_openings','foot_rejected','hand_cloth_intersection','insufficient_pad_patch','static_accepted','approach_accepted']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow(dict(number=row['number'],pitch_deg=row['pitch_deg'],roll_deg=row['roll_deg'],**{k:row['counts'].get(k,0) for k in fields[3:]}))
if __name__=='__main__':main()
