#!/usr/bin/env python3
"""Cross-check every skin-clear candidate against prepared foot contact hulls."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import RightHand,ConvexFootProxy,orientation


def full_grid_trial(path):
    import fcl
    path=Path(path);r=json.loads(path.read_text());settings=r['settings']
    state=json.loads((ROOT/r['input_state']).read_text())
    hand=RightHand(ROOT/settings['urdf'],settings['tip_inset_m'],settings['maximum_gap_m'])
    parts=hand.bank(settings['nominal_pinch_gap_m']);proxy=ConvexFootProxy(state['source']['surfaces'])
    counts=Counter();raw_candidates=[];request=fcl.DistanceRequest()
    for sample in r['samples']:
        for yaw in settings['yaw_degrees']:
            for tilt in settings['tilt_degrees']:
                for sign in settings['closing_signs']:
                    R=orientation(sample,yaw,tilt,sign);p=np.asarray(sample['position']);minimum=np.inf
                    for part in parts:
                        part.obj.setTransform(fcl.Transform(R,p))
                        d=float(fcl.distance(part.obj,proxy.objects[0],request,fcl.DistanceResult()))
                        if not np.isfinite(d):raise ValueError('Nonfinite proxy distance')
                        minimum=min(minimum,d)
                        if minimum<=0:break
                    counts['tested']+=1
                    if not sample['native_contact_face']:continue
                    counts['native_tested']+=1
                    if minimum>0:
                        counts['native_foot_proxy_nonintersecting']+=1
                        counts['native_foot_proxy_1mm_clear']+=minimum>=settings['body_clearance_m']
                        raw_candidates.append(dict(sample_index=sample['index'],yaw_deg=yaw,tilt_deg=tilt,closing_sign=sign,clearance_m=minimum))
    result=dict(number=r['number'],counts=dict(counts),raw_candidates=raw_candidates,
        scope='Every original sampled orientation, independently of the actual-skin distance filter; foot convex solid only.')
    r['full_grid_foot_proxy_check']=result;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    return result


def check_outputs(output):
    output=Path(output);summary=json.loads((output/'summary.json').read_text());settings=summary['settings']
    hand=RightHand(ROOT/settings['urdf'],settings['tip_inset_m'],settings['maximum_gap_m'])
    parts=hand.bank(settings['nominal_pinch_gap_m']);margin=settings['body_clearance_m'];rows=[]
    for item in summary['trials']:
        path=output/f'pose-{item["number"]:02d}'/'report.json';report=json.loads(path.read_text())
        state_path=ROOT/report['input_state']
        if hashlib.sha256(state_path.read_bytes()).hexdigest()!=report['input_sha256']:raise ValueError('Prepared state changed')
        state=json.loads(state_path.read_text());proxy=ConvexFootProxy(state['source']['surfaces'])
        counts=Counter();rejected=Counter()
        for c in report['candidates']:
            d,limiting=proxy.distances(parts,np.asarray(c['rotation_matrix']),np.asarray(c['grasp_center_unity_world_m']))
            clear=bool(np.all(d>=margin));nonintersection=bool(np.all(d>0))
            c['convex_foot_proxy_intersection']=bool(d[0]<=0)
            c['convex_calf_proxy_intersection']=bool(d[1]<=0)
            c['convex_foot_calf_proxy_clear_endpoint']=clear
            c['convex_foot_calf_proxy_no_intersection']=nonintersection
            c['convex_proxy_endpoint_clearance_m']=float(max(0,d.min()))
            c['convex_proxy_limiting_part']=limiting[int(d.argmin())]
            if c['native_contact_face']:
                counts['skin_clear_candidates']+=1
                counts['proxy_no_intersection']+=nonintersection
                counts['proxy_clear_candidates']+=clear
                counts['foot_only_no_intersection']+=bool(d[0]>0)
                counts['foot_only_clear_candidates']+=bool(d[0]>=margin)
                if not clear:rejected[c['convex_proxy_limiting_part']]+=1
        check=dict(representation='Prepared foot/calf convex contact hulls, checked separately from actual skin.',
            scope='Every actual-skin-clear candidate; all jointly clear candidates must belong to this set.',
            endpoint_gap_m=settings['nominal_pinch_gap_m'],body_clearance_m=margin,
            counts=dict(counts),first_limiting_parts=dict(rejected))
        report['simulator_proxy_check']=check
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        csv_path=path.with_name('candidates.csv')
        with csv_path.open(newline='') as stream:
            reader=csv.DictReader(stream);csv_rows=list(reader);columns=reader.fieldnames
        if len(csv_rows)!=len(report['candidates']):raise ValueError('Candidate CSV and JSON disagree')
        additions=['convex_foot_proxy_intersection','convex_calf_proxy_intersection','simulator_proxy_clear','convex_proxy_clearance_mm']
        columns=columns+[x for x in additions if x not in columns]
        for row,c in zip(csv_rows,report['candidates']):
            if int(row['sample_index'])!=c['sample_index']:raise ValueError('Candidate order changed')
            row.update(convex_foot_proxy_intersection=c['convex_foot_proxy_intersection'],
                convex_calf_proxy_intersection=c['convex_calf_proxy_intersection'],simulator_proxy_clear=c['convex_foot_calf_proxy_clear_endpoint'],
                convex_proxy_clearance_mm=c['convex_proxy_endpoint_clearance_m']*1000)
        with csv_path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(csv_rows)
        rows.append(dict(number=report['number'],pitch_deg=report['pitch_deg'],roll_deg=report['roll_deg'],**check))
    result=dict(pose_count=len(rows),trials=rows,
        poses_with_native_clear_endpoint=sum(r['counts'].get('proxy_clear_candidates',0)>0 for r in rows),
        poses_with_native_nonintersection=sum(r['counts'].get('proxy_no_intersection',0)>0 for r in rows),
        poses_with_foot_only_nonintersection=sum(r['counts'].get('foot_only_no_intersection',0)>0 for r in rows))
    (output/'simulator-proxy-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    summary['current_convex_proxy_poses_with_clear_endpoint']=result['poses_with_native_clear_endpoint']
    summary['current_convex_proxy_poses_with_no_intersection']=result['poses_with_native_nonintersection']
    summary['current_foot_only_proxy_poses_with_no_intersection']=result['poses_with_foot_only_nonintersection']
    summary['current_proxy_candidate_execution_validated']=False
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/right-grasp-geometry')
    parser.add_argument('--all-grid',action='store_true',help='Also test every sampled pose against the foot convex solid, without the skin-clear filter')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();r=check_outputs(args.output)
    print(json.dumps({k:v for k,v in r.items() if k!='trials'},indent=2))
    if args.all_grid:
        paths=[str(args.output/f'pose-{t["number"]:02d}'/'report.json') for t in r['trials']]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            trials=[]
            for f in as_completed([pool.submit(full_grid_trial,path) for path in paths]):
                trial=f.result();trials.append(trial);print(json.dumps(dict(number=trial['number'],counts=trial['counts'])),flush=True)
        result=dict(pose_count=len(trials),trials=trials,
            tested=sum(t['counts']['tested'] for t in trials),
            poses_with_native_nonintersection=sum(t['counts'].get('native_foot_proxy_nonintersecting',0)>0 for t in trials),
            poses_with_native_1mm_clearance=sum(t['counts'].get('native_foot_proxy_1mm_clear',0)>0 for t in trials))
        (args.output/'full-grid-foot-proxy-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='trials'},indent=2))


if __name__=='__main__':main()
