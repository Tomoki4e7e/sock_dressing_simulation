#!/usr/bin/env python3
"""Export a separate static-gravity reference for the adopted twelve routes."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.gravity_reference import StaticGravityReference,ARM_NAMES,JOINT_NAMES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,default=ROOT/'artifacts/gravity-calibration-v1/gravity-model.json')
    ap.add_argument('--source',type=Path,default=ROOT/'artifacts/reaching-shareset-lit-v3')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/gravity-calibration-v1/simulation-routes')
    a=ap.parse_args();model=StaticGravityReference(a.model);collection=json.loads((a.source/'collection.json').read_text());rows=[]
    for item in collection['rows']:
        source=Path(item['path']);out=a.output/item['split']/item['id'];out.mkdir(parents=True,exist_ok=True)
        before={name:sha(source/name) for name in ('angle.csv','torque.csv','external_torque.csv','frames.csv')}
        q=np.loadtxt(source/'angle.csv',delimiter=',',ndmin=2);gravity=model.predict(q)
        if gravity.shape!=(len(q),14) or not np.isfinite(gravity).all():raise ValueError('Invalid gravity output')
        np.savetxt(out/'gravity_arm.csv',gravity,delimiter=',',fmt='%.12g')
        metadata=dict(id=item['id'],split=item['split'],frames=len(q),input_joint_names=JOINT_NAMES,
            output_joint_names=ARM_NAMES,units='N*m',model=str(a.model.resolve()),model_sha256=sha(a.model),
            source=str(source),source_sha256=before,output_sha256=sha(out/'gravity_arm.csv'),
            component='gravity_only',gripper_force_columns_included=False,original_torque_overwritten=False,
            scope='Static physical gravity at saved poses; no motion, friction, contact, or fitted observation offset')
        for name,digest in before.items():
            if sha(source/name)!=digest:raise RuntimeError('Original reaching data changed')
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');rows.append(metadata)
    result=dict(completed=True,episodes=len(rows),frames=sum(r['frames'] for r in rows),rows=rows,
                output_columns=14,original_dataset_unchanged=True)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('GRAVITY ROUTE EXPORT PASSED',result['episodes'],result['frames'],flush=True)


if __name__=='__main__':main()
