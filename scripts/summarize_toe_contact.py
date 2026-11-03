#!/usr/bin/env python3
"""Summarize accepted unique ankle poses and their 50-row workbook mapping."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.pose_patterns import read_patterns


def tail_bend(path,minimum_z):
    state=json.loads(path.read_text());points=np.asarray(state['particles_world'])
    z=np.asarray(state['source']['rest_positions'])[:,2]
    levels=np.unique(np.round(z,6))
    centers=np.array([points[np.isclose(z,value,atol=1e-6)].mean(0) for value in levels])
    vectors=np.diff(centers,axis=0);vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    angles=np.degrees(np.arccos(np.clip((vectors[1:]*vectors[:-1]).sum(1),-1,1)))
    return float(angles[levels[1:-1]>=minimum_z-1e-6].max())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/toe-contact-final/summary.json')
    args=parser.parse_args()
    roots=[ROOT/'artifacts/toe-contact-boundary-check',ROOT/'artifacts/toe-contact-final']
    accepted={}
    for root in roots:
        for path in sorted(root.rglob('report.json')):
            report=json.loads(path.read_text())
            if not report.get('toe_contact',{}).get('passed'):continue
            if not report.get('geometry_audit',{}).get('geometry_ok') or not report.get('surface_audit',{}).get('surface_ok'):continue
            if not path.with_name('visual-state.json').exists():continue
            pattern=report['pattern'];angle=(pattern['pitch_deg'],pattern['roll_deg'])
            accepted[angle]=(path,report)
    patterns=read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
    mapping=[];missing=[]
    for pattern in patterns:
        angle=(pattern.pitch_deg,pattern.roll_deg)
        if angle not in accepted:
            missing.append(pattern.number);continue
        path,report=accepted[angle]
        mapping.append(dict(number=pattern.number,pitch_deg=pattern.pitch_deg,roll_deg=pattern.roll_deg,
            physical_trial_number=report['pattern']['number'],report=str(path.relative_to(ROOT)),
            reuse_reason='Only ankle pitch/roll are applied; workbook sock IDs and other conditions are ignored.'))
    results=[]
    for angle,(path,report) in sorted(accepted.items()):
        toe=report['toe_contact'];final=toe['final'];number=report['pattern']['number']
        previous=next((ROOT/'artifacts/sock-pose-pattern/held-gravity').glob(f'{number:02d}-*/visual-state.json'))
        support=toe['support']
        results.append(dict(number=number,pitch_deg=angle[0],roll_deg=angle[1],report=str(path.relative_to(ROOT)),
            fixed_particles=final['fixed_particles'],band_particles=len(final['band_particle_indices']),
            band_fixed_particles=final['band_fixed_particles'],native_finite_mass_particles=final['native_mesh_finite_mass_particles'],
            dynamic_tail_follower_particles=final['dynamic_tail_follower_particles'],
            fixed_error_m=final['maximum_fixed_error_m'],free_speed_m_s=final['maximum_free_speed_m_s'],
            band_speed_m_s=final['maximum_band_speed_m_s'],whole_cloth_recent_excursion_m=final.get('whole_cloth_recent_excursion_m'),
            support_material_range_m=[support['material_min_m'],support['material_max_m']],
            previous_maximum_tail_centerline_bend_deg=tail_bend(previous,support['material_max_m']),
            maximum_tail_centerline_bend_deg=tail_bend(path.with_name('visual-state.json'),support['material_max_m'])))
    contact_path=ROOT/'artifacts/toe-contact-final/contact-response/report.json'
    contact=json.loads(contact_path.read_text()) if contact_path.exists() else {}
    summary=dict(passed=not missing and len(mapping)==50 and contact.get('passed',False),
        workbook_rows=len(patterns),unique_angle_count=len({(p.pitch_deg,p.roll_deg) for p in patterns}),
        accepted_physical_angle_trials=len(accepted),mapped_rows=len(mapping),missing_rows=missing,
        model='native_Obi_contact_band_with_gravity_section_tail_and_local_toe_support',
        maximum_fixed_error_m=max((r['fixed_error_m'] for r in results),default=None),
        maximum_free_speed_m_s=max((r['free_speed_m_s'] for r in results),default=None),
        maximum_band_speed_m_s=max((r['band_speed_m_s'] for r in results),default=None),
        contact_response_report=str(contact_path.relative_to(ROOT)),
        contact_response_passed=contact.get('passed',False),contact_response_pose=contact.get('number'),
        contact_checks=contact.get('checks',{}),
        trials=results,workbook_mapping=mapping,
        limitations=['Finger fixture contact is tested in one representative pose; arm reaching and grasp success are not tested.',
            'The hanging tail is a reduced section model, not an independent native cloth surface.',
            'Toe support remains active; whole-sock pulling needs release and bidirectional coupling.',
            'The 30 mm band uses material coordinates and complete mesh faces; boundary faces are conservatively excluded.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('trials','workbook_mapping','limitations')},indent=2))
    if not summary['passed']:raise SystemExit(2)


if __name__=='__main__':main()
