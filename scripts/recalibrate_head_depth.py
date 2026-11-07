#!/usr/bin/env python3
"""Fit depth encoding on every training image and regenerate depth without rerunning SAM."""
from __future__ import annotations
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from process_reaching_shareset import load_depth,digest,write_json
PROFILE='head-stereo-zero-offset-v2'
VERSION='full-training-frames-v3'
CHECKPOINT=Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth')


def worker(out,ep,mode):
    cache=out/'depth-raw-cache'/ep.name;cache.mkdir(parents=True,exist_ok=True)
    with (cache/'.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        worker_unlocked(out,ep,mode)


def worker_unlocked(out,ep,mode):
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    meta=json.loads((ep/'metadata.json').read_text());count=meta['frames']
    cache=out/'depth-raw-cache'/ep.name;cache.mkdir(parents=True,exist_ok=True)
    stats_path=cache/'stats.json'
    if mode=='cache' and stats_path.exists():return
    model=None
    if mode=='cache' or not (cache/'stats.json').exists():model=load_depth(CHECKPOINT)
    if mode=='apply':
        cal=json.loads((out/'data/data_sock_reaching_v1/train/depth.json').read_text())
        path=ep/'perception/validation.json';report=json.loads(path.read_text())
        backup=ep/'perception/validation-before-full-depth-range.json'
        if not backup.exists():shutil.copy2(path,backup)
        original=ep/'confirmation/depth_before_full_training_range';original.mkdir(parents=True,exist_ok=True)
        for name in ('camera_depth','depth_mask'):
            if not (original/name).exists():shutil.move(str(ep/name),str(original/name))
        (ep/'camera_depth').mkdir(exist_ok=True)
        for kind in ('sock','foot'):(ep/f'depth_mask/{kind}_depth').mkdir(parents=True,exist_ok=True)
    rows=[];started=time.monotonic()
    with torch.inference_mode():
        for index in range(count):
            source=ep/'camera_right'/f'{index}.png';target=cache/f'{index}.npz'
            raw=None
            if target.exists():
                try:
                    with np.load(target) as f:raw=f['depth']
                except (OSError,ValueError,EOFError,zipfile.BadZipFile):
                    target.rename(target.with_suffix('.interrupted-npz'))
            if raw is None:
                if model is None:model=load_depth(CHECKPOINT)
                raw=model.infer_image(cv2.imread(str(source)),input_size=518)
            if not np.isfinite(raw).all() or raw.min()<0:raise ValueError('Invalid native ReLU depth prediction')
            if mode=='cache':
                if not target.exists():
                    temporary=target.with_suffix('.tmp')
                    with temporary.open('wb') as stream:np.savez(stream,depth=raw.astype('float32'))
                    temporary.replace(target)
                rows.append(dict(frame=index,rgb_sha256=digest(source),minimum=float(raw.min()),maximum=float(raw.max())))
            else:
                normalized=(raw-cal['depth_min'])/(cal['depth_max']-cal['depth_min'])
                encoded=np.clip(normalized*255,0,255).astype('uint8')
                Image.fromarray(encoded).save(ep/'camera_depth'/f'{index}.png')
                clipped=(normalized<0)|(normalized>1);upper={}
                for kind in ('sock','foot'):
                    mask=np.array(Image.open(ep/f'camera_right_mask/{kind}_mask/{index}.png'))>0
                    Image.fromarray(np.where(mask,encoded,0).astype('uint8')).save(ep/f'depth_mask/{kind}_depth/{index}.png')
                    upper[kind]=float((raw[mask]>cal['depth_max']).mean())
                    if upper[kind]>.02:raise ValueError(f'Full-train depth target upper clipping: {ep.name}/{index}/{kind}: {upper[kind]}')
                    if not encoded[mask].any():raise ValueError('Empty re-encoded target depth')
                row=report['rows'][index];row.update(depth_clip_fraction=float(clipped.mean()),target_depth_upper_clip_fraction=upper)
                row['checks'].update(depth_finite=True,depth_nonzero=True,depth_target_upper_clip=all(v<=.02 for v in upper.values()))
                row['passed']=all(row['checks'].values())
                if target.exists():target.unlink()  # Disposable prediction cache; original data remain archived.
            if index%50==0:
                write_json(cache/f'{mode}-progress.json',dict(processed=index+1,total=count,elapsed_s=time.monotonic()-started))
                print(mode.upper(),ep.name,index+1,'/',count,flush=True)
    if mode=='cache':
        write_json(stats_path,dict(episode=str(ep),number=meta['workbook_pattern']['number'],split='train',count=count,
            minimum=min(r['minimum'] for r in rows),maximum=max(r['maximum'] for r in rows),rows=rows,checkpoint_sha256=digest(CHECKPOINT)))
    else:
        # Isolated SAM corrections remain in place; archive their old depth hashes before updating encoding.
        repairs=ep/'perception/head-repairs.json'
        if repairs.exists():
            old_repairs=ep/'perception/head-repairs-before-full-depth-range.json'
            if not old_repairs.exists():shutil.copy2(repairs,old_repairs)
            log=json.loads(repairs.read_text())
            for r in log['rows'].values():
                r['depth_sha256']=digest(ep/f'camera_depth/{r["frame"]}.png')
                for change in r['changes']:change['corrected_sha256']=digest(ep/change['path'])
                r['depth_encoding_updated_by']='recalibrate_head_depth.py; original depth estimates/encoding archived separately'
            write_json(repairs,log)
        report.update(depth_calibration=cal,depth_reprocessing_source_sha256=digest(__file__),depth_calibration_version=VERSION,
            passed=all(r['passed'] for r in report['rows']))
        write_json(path,report)
        meta.update(perception_passed=report['passed'],reaching_learning_ready=report['passed'],depth_recalibration_pending=False,
            depth_calibration_version=VERSION,depth_encoding='uint8 relative depth; native ReLU floor 0, full-training-frame upper calibration')
        write_json(ep/'metadata.json',meta)
        write_json(ep/'perception/progress.json',dict(completed=True,processed=count,total=count,passed=report['passed'],last=report['rows'][-1]))
    print('DEPTH',mode,'PASSED',ep.name,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/reaching-shareset-head-v2')
    ap.add_argument('--episode',type=Path);ap.add_argument('--mode',choices=['cache','apply']);ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args();out=a.output.resolve()
    if a.episode is not None:worker(out,a.episode.resolve(),a.mode);return
    report_path=out/'report.json';old=json.loads(report_path.read_text())
    snapshot=out/'report-before-full-depth-range.json'
    if not snapshot.exists():shutil.copy2(report_path,snapshot)
    old['completed']=False;old['depth_recalibration_pending']=True;write_json(report_path,old)
    rows=json.loads((out/'collection.json').read_text())['rows'];episodes=[Path(r['path']) for r in rows]
    for ep in episodes:
        m=json.loads((ep/'metadata.json').read_text());m.update(reaching_learning_ready=False,depth_recalibration_pending=True);write_json(ep/'metadata.json',m)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def run(ep,mode):
        with (out/f'{ep.name}-depth-{mode}.log').open('w') as f:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(out),'--episode',str(ep),'--mode',mode],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    train=[Path(r['path']) for r in rows if r['split']=='train']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(lambda ep:run(ep,'cache'),train))
    samples=[json.loads((out/'depth-raw-cache'/ep.name/'stats.json').read_text()) for ep in train]
    maximum=max(s['maximum'] for s in samples)
    cal=dict(camera_profile=PROFILE,depth_min=0.,depth_max=maximum*(1+1/255),training_maximum=maximum,
        checkpoint_sha256=digest(CHECKPOINT),calibration_version=VERSION,training_frames=sum(s['count'] for s in samples),
        source='All frames from every training episode only; native ReLU floor 0; one-bin upper headroom',samples=samples)
    target=out/'data/data_sock_reaching_v1/train/depth.json';backup=target.with_name('depth-before-full-training-range.json')
    if not backup.exists():shutil.copy2(target,backup)
    write_json(target,cal);print('FULL TRAINING CALIBRATION',cal['training_frames'],cal['depth_min'],cal['depth_max'],flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(lambda ep:run(ep,'apply'),episodes))
    print('ALL HEAD DEPTH REENCODED',flush=True)


if __name__=='__main__':main()
