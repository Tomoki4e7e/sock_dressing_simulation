#!/usr/bin/env python3
"""Process collected stereo head episodes, validate and train ShareSet smoke models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from collect_reaching_shareset import split_for
from sock_dressing_simulation.reaching_collection import sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/reaching-shareset-head-v2')
    ap.add_argument('--ml-python',default='/usr/bin/python3');ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLCONFIGDIR='/tmp/sock-collection-mpl',PYTHONUNBUFFERED='1')
    rows=json.loads((ROOT/'artifacts/diverse-left-reaching/corrected-collection/manifest.json').read_text())['rows']
    episodes={r['id']:out/'data/data_sock_reaching_v1'/split_for(r['number'])/r['id'] for r in rows}
    def ready(ep):
        path=ep/'metadata.json'
        if not path.exists():return False
        try:m=json.loads(path.read_text())
        except json.JSONDecodeError:return False
        if m.get('error'):raise RuntimeError(f'Raw head collection failed: {ep}: {m["error"]}')
        return m.get('raw_validation_passed',False) and m.get('camera_profile')=='head-stereo-zero-offset-v2'
    def run(cmd,log):
        with log.open('w') as stream:subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    while not all(ready(episodes[r['id']]) for r in rows if split_for(r['number'])=='train'):time.sleep(10)
    run([a.ml_python,str(ROOT/'scripts/process_head_reaching_shareset.py'),'--output',str(out),'--calibrate-only'],out/'head-depth-calibration.log')
    print('HEAD DEPTH CALIBRATION READY',flush=True)
    def process(ep):
        path=ep/'perception/validation.json'
        if path.exists():
            d=json.loads(path.read_text())
            current_source=d.get('processing_source_sha256')==sha256(ROOT/'scripts/process_head_reaching_shareset.py')
            depth_reprocessed=(d.get('depth_reprocessing_source_sha256')==sha256(ROOT/'scripts/recalibrate_head_depth.py')
                               and d.get('depth_calibration_version')=='full-training-frames-v3')
            if d.get('passed') and d.get('camera_profile')=='head-stereo-zero-offset-v2' and (current_source or depth_reprocessed):return
        print('PROCESS HEAD',ep.name,flush=True)
        try:
            run([a.ml_python,str(ROOT/'scripts/process_head_reaching_shareset.py'),'--output',str(out),'--episode',str(ep)],out/(ep.name+'-head-perception.log'))
        except subprocess.CalledProcessError:
            # Only a complete current-source semantic report is eligible for isolated correction.
            if not path.exists():raise
            failed=json.loads(path.read_text())
            if failed.get('passed') or failed.get('processing_source_sha256')!=sha256(ROOT/'scripts/process_head_reaching_shareset.py'):raise
            run([a.ml_python,str(ROOT/'scripts/repair_head_reaching_perception.py'),str(ep)],out/(ep.name+'-isolated-repair.log'))
        print('HEAD QUALIFIED',ep.name,flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending=set(episodes);futures=[]
        while pending:
            for key in sorted(pending):
                if ready(episodes[key]):futures.append(pool.submit(process,episodes[key]));pending.remove(key)
            for future in futures:
                if future.done():future.result()
            if pending:time.sleep(10)
        for future in futures:future.result()
    # Raw collection also writes the complete split manifest.
    while not (out/'collection.json').exists():time.sleep(5)
    share=ROOT.parent/'ShareSet';config=share/'share_folder/configs/reaching_head_simulation.yaml'
    import yaml
    cfg=yaml.safe_load(config.read_text());cfg['dataset'].update(root=str(out/'data'),manifest=str(out/'data/dataset_reaching.yaml'))
    config=out/'training-head-config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    for representative in (True,False):
        name='model-representative-smoke' if representative else 'model-all12-smoke'
        cmd=[a.ml_python,str(share/'share_folder/src/main/SAMDAMSARNN_learning_reaching.py'),'--config',str(config),'--output',str(out/name),'--epochs','1']
        if representative:cmd.append('--representative-only')
        print('HEAD TRAINING START',name,flush=True);run(cmd,out/(name+'.log'));print('HEAD TRAINING PASSED',name,flush=True)
    results=[]
    for r in rows:
        ep=episodes[r['id']];m=json.loads((ep/'metadata.json').read_text());p=json.loads((ep/'perception/validation.json').read_text())
        if not (m['raw_validation_passed'] and p['passed'] and m['reaching_learning_ready']):raise RuntimeError('Unqualified head episode')
        expected={str(i) for i in range(m['frames'])}
        for folder in ('camera_right','camera_left','camera_overview','camera_depth','camera_right_mask/sock_mask','camera_right_mask/foot_mask','depth_mask/sock_depth','depth_mask/foot_depth'):
            if {path.stem for path in (ep/folder).glob('*.png')}!=expected:raise RuntimeError('Head dataset frame alignment failed')
        results.append(dict(id=r['id'],number=r['number'],split=split_for(r['number']),frames=m['frames'],path=str(ep),
            camera_profile=m['camera_profile'],stereo_baseline_m=m['stereo_baseline_m'],raw_passed=True,perception_passed=True,
            prompt_resets=len(p['resets']),angle_sha256=sha256(ep/'angle.csv'),torque_sha256=sha256(ep/'torque.csv'),
            isolated_repaired_frames=p.get('isolated_repaired_frames',[]),
            review=str((ep/'perception/review.jpg').relative_to(out)),right=str((ep/'camera_right/0.png').relative_to(out)),
            left=str((ep/'camera_left/0.png').relative_to(out)),overview=str((ep/'camera_overview/0.png').relative_to(out))))
    report=dict(completed=True,camera_profile='head-stereo-zero-offset-v2',count=len(results),total_frames=sum(r['frames'] for r in results),
        rows=results,learning_cameras=['camera_right','camera_left'],model_rgb_input='camera_right',
        confirmation_camera='camera_overview',depth_and_masks_source='camera_right',reference_format='reference-format.json',
        scope='Frozen cloth reaching/geometric closure only',dynamic_grasp_validated=False,model_accuracy_validated=False,
        training='Representative and all-route one-epoch head-camera smoke checks',scripts={str(p.relative_to(ROOT)):sha256(p) for p in
            [Path(__file__),ROOT/'scripts/process_head_reaching_shareset.py',ROOT/'scripts/repair_head_reaching_perception.py',ROOT/'scripts/recalibrate_head_depth.py',ROOT/'sock_dressing_simulation/head_cameras.py',ROOT/'sock_dressing_simulation/head_camera_profile.py']})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    cards=''
    for row in results:
        cards+='<article><h2>'+html.escape(row['id'])+'</h2><p>'+f"{row['split']} / {row['frames']} synchronized frames / baseline {row['stereo_baseline_m']*1000:.3f} mm"+'</p><div class="views">'
        for key,label in [('right','頭部右：camera_right'),('left','頭部左：camera_left'),('overview','確認用：camera_overview')]:
            cards+='<figure><figcaption>'+label+'</figcaption><a href="'+row[key]+'"><img src="'+row[key]+'"></a></figure>'
        cards+='</div><a href="'+row['review']+'">右頭部画像のSAM2検査画像</a></article>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>ShareSet 頭部左右カメラ</title>'
        '<style>body{font:16px sans-serif;margin:2rem;background:#f5f7fa}article{background:white;padding:1rem;margin:1rem 0}.views{display:flex;gap:1rem}figure{margin:0;flex:1}img{width:100%}</style>'
        '<h1>Dry-AIREC 頭部左右カメラの学習データ</h1><p>12経路・'+str(report['total_frames'])+'フレーム。訓練8／検証2／テスト2経路。</p>'
        '<p>camera_right／camera_leftはURDFの頭部See3CAM取付フレーム、位置・角度オフセット0。camera_overviewは確認用です。'
        '深度とマスクは頭部右画像から生成しました。実機レンズの精密な内部較正を再現したものではありません。</p>'
        '<p>固定布への接近・幾何学的閉鎖が対象です。学習確認は各1 epochで、動作成功率は未検証です。</p>'
        '<p>追跡の再初期化と、孤立フレームのSAM2内部1024での修正を記録しています。元の予測・検査結果・ハッシュは各経路のperceptionに保持しています。</p>'
        '<p><a href="report.json">JSON</a> / <a href="reference-format.json">参照ShareSet形式</a> / <a href="data/dataset_reaching.yaml">分割</a></p>'+cards)
    print('HEAD PIPELINE COMPLETE',len(results),report['total_frames'],flush=True)


if __name__=='__main__':main()
