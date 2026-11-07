#!/usr/bin/env python3
"""Process lit recorded-state renders, freeze train-only depth, and smoke-train ShareSet."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json,os
from pathlib import Path
import subprocess,sys,time
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.reaching_collection import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/'artifacts/reaching-shareset-head-v2')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/reaching-shareset-lit-v3');ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args();out=a.output.resolve();source=a.source.resolve();out.mkdir(exist_ok=True,parents=True)
    original=json.loads((source/'collection.json').read_text());rows=original['rows']
    episodes={r['id']:out/'data/data_sock_reaching_v1'/r['split']/r['id'] for r in rows}
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR='/tmp/sock-collection-mpl',PYTHONUNBUFFERED='1')
    def run(cmd,log):
        with log.open('w') as stream:subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def ready(ep):
        p=ep/'metadata.json'
        if not p.exists():return False
        try:m=json.loads(p.read_text())
        except json.JSONDecodeError:return False
        return m.get('raw_validation_passed') and m.get('lighting_profile')=='balanced'
    def process(ep,phase):
        path=ep/'perception/validation.json'
        if path.exists():
            d=json.loads(path.read_text())
            if d.get('passed') and d.get('processing_source_sha256')==sha256(ROOT/'scripts/process_lit_reaching.py'):
                if phase=='full' and not d.get('depth_pending'):return
                if phase=='train-cache' and (out/'depth-raw-cache'/ep.name/'stats.json').exists():return
        print('LIT PROCESS',ep.name,phase,flush=True)
        run(['/usr/bin/python3',str(ROOT/'scripts/process_lit_reaching.py'),'--episode',str(ep),'--output',str(out),'--phase',phase],out/(ep.name+'-lit-perception.log'))
        print('LIT MASKS QUALIFIED',ep.name,phase,flush=True)
    train=[r for r in rows if r['split']=='train']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={r['id'] for r in train};futures=[]
        while pending:
            for key in sorted(pending):
                if ready(episodes[key]):futures.append(pool.submit(process,episodes[key],'train-cache'));pending.remove(key)
            for f in futures:
                if f.done():f.result()
            if pending:time.sleep(10)
        for f in futures:f.result()
    samples=[json.loads((out/'depth-raw-cache'/r['id']/'stats.json').read_text()) for r in train]
    checkpoint=Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth')
    maximum=max(r['maximum'] for r in samples)
    calibration=dict(camera_profile='head-stereo-zero-offset-v2',lighting_profile='balanced',depth_min=0.,depth_max=maximum*256/255,
        training_maximum=maximum,training_frames=sum(r['count'] for r in samples),samples=samples,
        checkpoint_sha256=sha256(checkpoint),calibration_version='full-training-frames-v3',
        source='Every lit training frame only; native ReLU floor 0, one encoded-bin upper headroom')
    (out/'data/data_sock_reaching_v1/train/depth.json').write_text(json.dumps(calibration,indent=2)+'\n')
    print('LIT DEPTH FROZEN',calibration['training_frames'],calibration['depth_max'],flush=True)
    def encode(ep):
        run(['/usr/bin/python3',str(ROOT/'scripts/recalibrate_head_depth.py'),'--output',str(out),'--episode',str(ep),'--mode','apply'],out/(ep.name+'-depth-encode.log'))
        p=ep/'perception/validation.json';d=json.loads(p.read_text());d['depth_pending']=False;p.write_text(json.dumps(d,indent=2)+'\n')
        print('LIT DEPTH QUALIFIED',ep.name,flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(encode,episodes[r['id']]) for r in train]
        pending={r['id'] for r in rows if r['split']!='train'}
        while pending:
            for key in sorted(pending):
                if ready(episodes[key]):futures.append(pool.submit(process,episodes[key],'full'));pending.remove(key)
            for f in futures:
                if f.done():f.result()
            if pending:time.sleep(10)
        for f in futures:f.result()
    while not (out/'collection.json').exists():time.sleep(5)
    share=ROOT.parent/'ShareSet';config=yaml.safe_load((share/'share_folder/configs/reaching_head_simulation.yaml').read_text())
    config['dataset'].update(root=str(out/'data'),manifest=str(out/'data/dataset_reaching.yaml'),expected_lighting_profile='balanced')
    cfg=out/'training-lighting-config.yaml';cfg.write_text(yaml.safe_dump(config,sort_keys=False))
    for representative in (True,False):
        name='model-representative-smoke' if representative else 'model-all12-smoke'
        cmd=['/usr/bin/python3',str(share/'share_folder/src/main/SAMDAMSARNN_learning_reaching.py'),'--config',str(cfg),'--output',str(out/name),'--epochs','1']
        if representative:cmd.append('--representative-only')
        print('LIT TRAIN START',name,flush=True);run(cmd,out/(name+'.log'));print('LIT TRAIN PASSED',name,flush=True)
    print('ALL LIT PERCEPTION AND TRAINING PASSED',flush=True)


if __name__=='__main__':main()
