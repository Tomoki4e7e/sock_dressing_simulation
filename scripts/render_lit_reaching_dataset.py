#!/usr/bin/env python3
"""Re-render the adopted twelve measured episodes under corrected lighting."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json,os
from pathlib import Path
import subprocess,sys
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.reaching_collection import sha256
from sock_dressing_simulation.learning_lighting import PROFILES


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=ROOT/'artifacts/reaching-shareset-head-v2')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/reaching-shareset-lit-v3')
    ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();source=a.source.resolve();out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
    original=json.loads((source/'collection.json').read_text());items=original['rows']
    if len(items)!=12:raise ValueError('Exactly the adopted twelve source episodes are required')
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',LP_NUM_THREADS='2',
        MPLCONFIGDIR='/tmp/sock-collection-mpl',PYTHONUNBUFFERED='1',__GLX_VENDOR_LIBRARY_NAME='mesa',LIBGL_ALWAYS_SOFTWARE='1',
        SOCK_REGISTERED_BATCHMODE='1',SOCK_REGISTERED_PLAYER=str(ROOT/'Build/SockDressingPlayer/RegisteredLightingV1/Player.x86_64'),
        PYTHONPATH=str(ROOT/'.deps/RCareWorld/pyrcareworld')+':/opt/ros/noetic/lib/python3/dist-packages')
    def render(item):
        index,row=item;src=Path(row['path']);meta=json.loads((src/'metadata.json').read_text())
        ep=out/'data/data_sock_reaching_v1'/row['split']/row['id'];native=out/'native'/row['id'];native.mkdir(parents=True,exist_ok=True)
        if not (ep/'metadata.json').exists():
            settings=dict(env,SOCK_REGISTERED_PLAYER_LOG=str(native/'player.log'))
            print('RENDER',row['id'],flush=True)
            with (native/'render.log').open('w') as stream:
                subprocess.run(['/tmp/sock-diversity-venv/bin/python',str(ROOT/'scripts/capture_fixed_torso_initial.py'),
                    '--trajectory',str(ROOT/meta['trajectory']),'--output',str(native),'--head-cameras','--lighting-profile','balanced',
                    '--render-episode',str(src),'--render-output',str(ep),'--port',str(18080+index*10)],cwd=ROOT,env=settings,stdout=stream,stderr=subprocess.STDOUT,check=True)
        current=json.loads((ep/'metadata.json').read_text());validation=json.loads((ep/'render-validation.json').read_text())
        if not current.get('raw_validation_passed') or current.get('lighting_profile')!='balanced' or not validation['passed']:raise ValueError('Lighting render gate failed')
        if any(current['lighting'].get(key)!=value for key,value in PROFILES['balanced'].items()):
            raise ValueError('Saved lighting settings differ; render into a fresh output directory')
        for name,digest in current['source_signal_sha256'].items():
            if sha256(src/name)!=digest or sha256(ep/name)!=digest:raise ValueError('Source teaching signals changed')
        expected={str(i) for i in range(row['frames'])}
        for folder in ('camera_right','camera_left','camera_overview','qa/visible_sock'):
            if {p.stem for p in (ep/folder).glob('*.png')}!=expected:raise ValueError('Rendered image IDs differ from source')
        print('LIT RAW PASSED',row['id'],flush=True)
        return dict(row,path=str(ep),source_path=str(src),source_metadata_sha256=sha256(src/'metadata.json'))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(render,enumerate(items)))
    payload=dict(original,rows=rows,source_collection=str(source/'collection.json'),source_collection_sha256=sha256(source/'collection.json'),
        lighting_profile='balanced',acquisition_mode='re-render recorded measured states; signals remain byte-identical')
    (out/'collection.json').write_text(json.dumps(payload,indent=2)+'\n')
    manifest={'data_sock_reaching_v1':{split:{r['id']:dict(start=0,end=r['frames']) for r in rows if r['split']==split} for split in ('train','validation','test')}}
    (out/'data/dataset_reaching.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
    print('ALL LIT RENDERS PASSED',len(rows),sum(r['frames'] for r in rows),flush=True)


if __name__=='__main__':main()
