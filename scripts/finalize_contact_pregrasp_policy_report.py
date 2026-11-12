#!/usr/bin/env python3
"""Validate all test artifacts, rebuild the canonical report and make a portable ZIP."""
import argparse,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
from PIL import Image


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run.resolve();out=run/'live-test-motion'
    suite=json.loads((out/'report.json').read_text());assert suite['completed'] and suite['expected']==12
    expected=[r for r in json.loads((run/'manifest.json').read_text())['rows'] if r['split']=='test'];checkpoint=run/'best-test-inference/SARNN_best_evaluated.pth';sha=digest(checkpoint);validation=[];raw=[];total_steps=0
    for item in expected:
        dest=out/item['id']/'policy';r=json.loads((dest/'report.json').read_text());steps=(item['frames']+1)//2-1
        assert r['completed'] and r['steps']==steps and r['checkpoint_sha256']==sha
        assert r['device']['device']=='cuda' and all(v.startswith('cuda') for v in r['device']['model_devices'].values())
        assert r.get('physics_batch_steps',1)==1
        if 'actual_physics_steps' in r:assert r['actual_physics_steps']==steps*10
        assert r['cloth_maximum_drift_m']<2e-5
        assert set(r['geometry'])=={'left','right'}
        total_steps+=steps;checks=[]
        for filename,frames in [('overview.mp4',steps+1),('head-input.mp4',steps)]:
            p=dest/filename;probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,r_frame_rate','-of','json',str(p)],text=True))['streams'][0]
            assert probe['codec_name']=='h264' and int(probe['nb_frames'])==frames and probe['r_frame_rate']=='5/1'
            already_decoded=any(c.get('decoded') and c.get('frames')==frames and Path(c.get('path','')).name==filename for c in r.get('video_validation',[]))
            if not already_decoded:subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','null','-'],check=True)
            check=dict(path=str(p.relative_to(out)),frames=frames,decoded=True,probe=probe);checks.append(check);validation.append(check)
        for filename in ['overview-final.png']+[f'{s}-{v}.png' for s in ('left','right') for v in ('gap-side','gap-opposite')]:
            with Image.open(dest/filename) as im:im.verify()
        r['execution_completed']=r['completed'];r['goal_geometry_passed']=all(g['contact_geometry_checks']['passed'] for g in r['geometry'].values());r['passed']=r['goal_geometry_passed'];r['video_validation']=checks
        trace=[json.loads(l) for l in (dest/'motion-trace.jsonl').read_text().splitlines()];assert len(trace)==steps
        r['maximum_input_out_of_training_range_fraction']=max(x['prediction']['input_out_of_range_fraction'] for x in trace)
        (dest/'report.json').write_text(json.dumps(r,indent=2)+'\n');raw.append(r)
        initial=out/item['id']/'native-route-initial/result.json'
        if initial.exists():
            v=json.loads(initial.read_text());v.setdefault('initialization_passed',v['passed']);v.update(passed=r['goal_geometry_passed'],execution_completed=True,motion_replayed=False,policy_executed=True)
            if isinstance(v.get('motion'),dict):v['motion']=r
            initial.write_text(json.dumps(v,indent=2)+'\n')
    subprocess.run(['/usr/bin/python3',str(Path(__file__).with_name('report_contact_pregrasp_policy_tests.py')),'--run',str(run)],check=True)
    terminal=json.loads((out/'terminal-summary.json').read_text())
    old=out/'execution-suite-report.json'
    if not old.exists():shutil.copy2(out/'report.json',old)
    report=dict(completed=True,expected=12,completed_episodes=12,checkpoint_epoch=4780,checkpoint_sha256=sha,total_policy_commands=total_steps,total_physics_steps=total_steps*10,total_simulated_duration_s=total_steps*.2,device='cuda',mode='Live closed-loop model inference and native joint-drive execution; no teacher trajectory commands',both_cloth_present=terminal['both_cloth_present'],both_target_contact_geometry_passed=terminal['both_target_contact_geometry_passed'],rows=terminal['rows'],video_validation=validation,scope='Frozen cloth/human; finite convex-jaw geometric occupancy, no retention/pulling validation',torque_input='Causal endpoint smoothing of past 10Hz samples, up to window 29. Offline training used centered smoothing.',measurement='Actual measured Unity finger transforms and cloth vertices; convex aperture clipping gives surface area and opening-rim length',diagnostics='pilot-* and diagnostic-* directories are separate checks; canonical C-10 was rerun with per-tick exchange, matching the other 11 cases.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    text=f'''GPU閉ループ推論・俯瞰動画と爪間の布検証

SARNN_best.pth / epoch 4780。テスト12状態をすべて完走。
SAMDAMSARNN・SAM2・Depth AnythingはCUDAで実行。
各時刻の実測頭部RGB・関節角・駆動力を入力し、予測角をモーターへ指令。
教示軌道の関節角は指令に使用していません。
物理刻み20 ms、モデル指令周期200 ms。全{total_steps}指令。

終端で左右とも爪間に布が存在した状態: {terminal['both_cloth_present']}/12。
左右の目標接触幾何条件を満たした状態: {terminal['both_target_contact_geometry_passed']}/12。
report.html: 俯瞰動画・接写・測定表をまとめた閲覧ページ。
terminal-measurements.csv: 状態別・左右別の布面積[mm²]、開口縁長[mm]、布帯との距離[mm]。
<状態>/policy/overview.mp4: 1280x960、5fps、H.264。動作開始から終端まで。
<状態>/policy/head-input.mp4: GPUモデルに渡した実測頭部画像。
<状態>/policy/*gap*.png: 終端の左右の爪の両側からの接写。
<状態>/policy/terminal-cloth-between-fingers.json: 詳細測定値・実測爪形状。
<状態>/policy/terminal-geometry.json.gz: 実測布頂点・三角形・材料深さ・リンク姿勢。
<状態>/policy/motion-trace.jsonl: 観測・予測・制限後の指令・実測結果。

人体と布は教示と同じ固定条件。爪間の布面積と開口縁長は、有限な爪の
凸包と実測布メッシュから測定しています。布が爪間にあることと、閉鎖・
摩擦保持・引き上げの成功は別です。今回は追加の閉鎖・保持試験をしていません。
ライブ入力の駆動力は過去サンプルだけを使う因果的平滑化です。
学習時の前後サンプルを使う平滑化との違いを残しています。

全24本のMP4のフレーム数・H.264形式・全フレームのデコードを確認済み。
面積既知の対照試験3件と、採用済み教示A-04の実測正常終端による陽性対照を実施。
pilot-* と diagnostic-* は別の動作確認で、12ケースの集計には含めていません。
'''
    (out/'README.txt').write_text(text)
    programs=out/'programs/final';programs.mkdir(exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    for p in [root/'scripts'/n for n in ['contact_pregrasp_policy_scene.py','contact_pregrasp_policy_worker.py','run_contact_pregrasp_policy_tests.py','report_contact_pregrasp_policy_tests.py','finalize_contact_pregrasp_policy_report.py']]+[root/'sock_dressing_simulation/pregrasp_gap_measurement.py',root/'tests/test_pregrasp_gap_measurement.py']:
        shutil.copy2(p,programs/p.name)
    package=[]
    for name in ['report.html','report.json','README.txt','terminal-measurements.csv','terminal-summary.json','teacher-positive-control.json']:
        package.append(out/name)
    for item in expected:
        dest=out/item['id']/'policy'
        for pattern in ['*.mp4','*gap*.png','overview-final.png','terminal*.json','terminal*.json.gz','motion-trace.jsonl','device.json','report.json']:
            package.extend(dest.glob(pattern))
    package.extend(p for p in (out/'programs').rglob('*') if p.is_file());package=sorted(set(package))
    records=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=digest(p)) for p in package]
    (out/'integrity.json').write_text(json.dumps(dict(files=records,video_count=24,checkpoint_sha256=sha),indent=2)+'\n');package.append(out/'integrity.json')
    archive=out/'videos-and-measurements.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in package:z.write(p,str(p.relative_to(out)))
        z.write(checkpoint,'SARNN_best_evaluated.pth')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    progress=json.loads((run/'progress.json').read_text());progress.update(completed=True,phase='early_stopped_best_offline_and_live_tests_complete',live_test_completed=True,live_test_report=str(out/'report.json'),live_test_cases=12,both_grippers_cloth_present=terminal['both_cloth_present']);(run/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    print(json.dumps(dict(completed=True,cases=12,commands=total_steps,both_cloth_present=terminal['both_cloth_present'],zip_bytes=archive.stat().st_size),indent=2))

if __name__=='__main__':main()
