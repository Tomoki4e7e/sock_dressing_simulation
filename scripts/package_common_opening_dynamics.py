#!/usr/bin/env python3
"""Bind all 140 measured replay verdicts to their immutable planned sources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from common_opening_replay_support import verify_source
from search_centered_reaching import write
from validate_common_opening_dynamics import report


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/'artifacts/common-opening-approach/negative-wrist-start');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/common-opening-dynamics');a=ap.parse_args();source=a.source.resolve();out=a.output.resolve();rows=[];members=[];criteria=digest(ROOT/'config/geometric_closure.yaml')
    paths=sorted(source.glob('routes/*/*.json'))
    if len(paths)!=140:raise ValueError('Expected 140 immutable source routes')
    for path in paths:
        record=json.loads(path.read_text());verify_source(path,source/'initial.yaml',ROOT/record['condition']['state']);destination=out/'replays'/record['side']/record['condition']['id'];status=json.loads((destination/'batch-status.json').read_text())
        if not status['finished'] or status['source_sha256']!=digest(path):raise ValueError('Missing or stale trial: '+str(path))
        if status['condition']!=record['condition'] or status['side']!=record['side']:raise ValueError('Condition mismatch')
        motion_path=destination/'native-route-initial/replay-result.json'
        if not motion_path.exists():raise ValueError('A runtime error remains unresolved: '+str(path))
        motion=json.loads(motion_path.read_text());native=json.loads((destination/'native-route-initial/result.json').read_text())
        if motion['source_sha256']!=digest(path) or motion['actual_closed_grasp']['criteria_sha256']!=criteria or not native['passed'] or not native['motion_replayed']:raise ValueError('Stale criteria or failed initialization: '+str(path))
        if motion['approach']['passed']!=status['approach_passed'] or motion['grasp_passed']!=status['grasp_passed'] or status['returncode']!=0:raise ValueError('Inconsistent or interrupted replay')
        planned=json.loads((destination/'planned-goal-grasp.json').read_text())
        if planned['source_sha256']!=digest(path) or planned['criteria_sha256']!=criteria:raise ValueError('Stale planned goal grasp check')
        trace=destination/'native-route-initial/motion-trace.jsonl'
        with trace.open() as f:
            count=sum(1 for _ in f)
        if count!=motion['steps']:raise ValueError('Incomplete measured trace')
        rows.append(status)
        for relative in ['planned-goal-grasp.json','batch-status.json','measured.csv','replay.log','native-route-initial/replay-result.json','native-route-initial/result.json','native-route-initial/motion-trace.jsonl']:
            member=destination/relative
            if not member.is_file():raise ValueError('Missing record: '+str(member))
            members.append(member)
    report(out,rows)
    protocol_path=out/'protocol.json';protocol=json.loads(protocol_path.read_text())
    for entry in protocol['files']:
        if entry['path'].startswith(('scripts/','sock_dressing_simulation/')):entry['sha256']=digest(ROOT/entry['path'])
        elif entry['sha256']!=digest(ROOT/entry['path']):raise ValueError('A fixed model/configuration input changed: '+entry['path'])
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    for name in ['summary.json','summary.csv','cases.csv','report.html','protocol.json','a-criteria-equivalence.json','threshold-gate-equivalence.json','tests.txt','planned-grasp-summary.json']:
        members.append(out/name)
    for name in ['a-criteria-equivalence.json','threshold-gate-equivalence.json']:
        if not json.loads((out/name).read_text())['passed']:raise ValueError('A rule equivalence control failed')
    manifest=dict(planned_archive=dict(path=str((source/'ik-trajectories-70-states.zip').relative_to(ROOT)),sha256=digest(source/'ik-trajectories-70-states.zip')),complete=True,trajectories=140,measured_steps=sum(json.loads((out/'replays'/r['side']/r['condition']['id']/'native-route-initial/replay-result.json').read_text())['steps'] for r in rows),approach_passed=sum(r['approach_passed'] for r in rows),grasp_passed=sum(r['grasp_passed'] for r in rows),criteria_sha256=criteria,files=[dict(path=str(p.relative_to(out)),sha256=digest(p),bytes=p.stat().st_size) for p in members],scope='Frozen-cloth PhysX motion and previous A geometric capture rule; no friction, retention, gravity load or effort-limit validation')
    write(out/'validation-manifest.json',manifest);members.append(out/'validation-manifest.json')
    archive=out/'validation-results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in members:z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failure')
    write(out/'archive.json',dict(path=archive.name,sha256=digest(archive),bytes=archive.stat().st_size,members=len(members)))
    print('PACKAGED',len(rows),'trials',len(members),'members',archive.stat().st_size,'bytes',flush=True)

if __name__=='__main__':main()
