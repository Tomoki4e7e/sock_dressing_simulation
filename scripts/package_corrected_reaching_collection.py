#!/usr/bin/env python3
"""Combine verified A and corrected B demonstrations under one shared start."""
import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.reaching_dataset import verify_dataset_inputs,file_sha256
from sock_dressing_simulation.held_stage_shapes import current_shape_qualified


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--collections',type=Path,nargs='+',default=[ROOT/'artifacts/diverse-left-reaching/final',ROOT/'artifacts/diverse-left-reaching/b-natural-toe'])
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/corrected-collection')
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    common='config/diverse_left_reaching_shared.yaml';rows=[];seen=set();files=[]
    for collection in a.collections:
        collection=collection.resolve();summary=json.loads((collection/'summary.json').read_text())
        for row in summary['rows']:
            if not row['actual_closed_capture_passed'] or not row.get('current_shape_qualified',True):continue
            path=ROOT/row['trajectory'];plan=json.loads(path.read_text());verify_dataset_inputs(plan,ROOT)
            if plan['shared_initial_configuration']!=common:raise ValueError('Collections use different initial configurations')
            state=json.loads((ROOT/plan['geometry_record']['state_file']).read_text())
            if not current_shape_qualified(row['stage'],state):continue
            dest=collection/'replays'/row['id'];native=json.loads((dest/'native-route-initial/replay-result.json').read_text());closed=json.loads((dest/'actual-closed-grasp-recheck.json').read_text())
            if native['source_sha256']!=file_sha256(path) or not native['passed'] or not closed['passed']:raise ValueError('Replay gate is stale')
            if row['id'] in seen:raise ValueError('Duplicate demonstration ID')
            seen.add(row['id']);rows.append(dict(id=row['id'],number=row['number'],pitch_deg=row['pitch_deg'],roll_deg=row['roll_deg'],stage=row['stage'],source_collection=str(collection.relative_to(ROOT)),trajectory=row['trajectory'],angle_file=str((dest/'angle.csv').relative_to(ROOT)),angle_sha256=file_sha256(dest/'angle.csv'),shape_profile=state.get('shape_profile','saved_held_A_cuff_bulge')))
            for name in ('angle.csv','frames.csv','demonstration.json','actual-closed-grasp-recheck.json'):files.append((dest/name,row['id']+'/'+name))
    manifest=dict(shared_initial_configuration=common,shared_initial_sha256=file_sha256(ROOT/common),count=len(rows),
                  stage_counts={stage:sum(r['stage']==stage for r in rows) for stage in 'ABCDE'},
                  conditions=len({(r['pitch_deg'],r['roll_deg'],r['stage']) for r in rows}),rows=rows,learning_ready=False,
                  scope='Measured joint demonstrations and geometric closure with frozen cloth. Legacy opening-stage shapes excluded; deformation, friction retention and pulling are not validated.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with (out/'manifest.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['id']);writer.writeheader();writer.writerows(rows)
    with zipfile.ZipFile(out/'demonstrations.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT/common,'shared_initial_configuration.yaml');archive.write(out/'manifest.json','manifest.json');archive.write(out/'manifest.csv','manifest.csv')
        for path,name in files:archive.write(path,name)
    print(json.dumps({k:manifest[k] for k in ('count','stage_counts','conditions')}),flush=True)


if __name__=='__main__':main()
