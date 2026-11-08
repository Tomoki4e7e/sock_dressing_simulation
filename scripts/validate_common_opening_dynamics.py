#!/usr/bin/env python3
"""Replay all immutable common-opening routes in fresh native PhysX players."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write
from common_opening_replay_support import verify_source
from replay_diverse_left_reaching import compress_intermediates


def export_trace(destination):
    path=destination/'native-route-initial/motion-trace.jsonl'
    if not path.exists():return
    names=['time_s','stage']+[f'{s}_arm/joint_{i}_rad' for s in ('right','left') for i in range(1,8)]+['right_gap_m','left_gap_m','tcp_x_m','tcp_y_m','tcp_z_m']
    with path.open() as source,(destination/'measured.csv').open('w') as target:
        writer=csv.writer(target);writer.writerow(names)
        for line in source:
            r=json.loads(line);writer.writerow([(r['step']+1)*.02,r['stage']]+r['actual_right_q_rad']+r['actual_left_q_rad']+[r['actual_right_gap_m'],r['actual_left_gap_m']]+r['tcp_world_m'])


def run_one(path,out,initial,port,timeout):
    record=json.loads(path.read_text());state=ROOT/record['condition']['state'];verify_source(path,initial,state)
    destination=out/'replays'/record['side']/record['condition']['id'];destination.mkdir(parents=True,exist_ok=True)
    status_path=destination/'batch-status.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        motion_path=destination/'native-route-initial/replay-result.json'
        prior_criteria=previous.get('criteria_sha256')
        if prior_criteria is None and motion_path.exists():
            prior_criteria=json.loads(motion_path.read_text()).get('actual_closed_grasp',{}).get('criteria_sha256')
        if previous.get('source_sha256')==hashlib.sha256(path.read_bytes()).hexdigest() and previous.get('finished') and prior_criteria==hashlib.sha256((ROOT/'config/geometric_closure.yaml').read_bytes()).hexdigest():return previous
    start=time.monotonic();env=dict(os.environ,SOCK_REGISTERED_PLAYER_LOG=str(destination/'player.log'),SOCK_REGISTERED_BATCHMODE='1',LIBGL_ALWAYS_SOFTWARE='1',__GLX_VENDOR_LIBRARY_NAME='mesa')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.deps/RCareWorld/pyrcareworld'),'/opt/ros/noetic/lib/python3/dist-packages',env.get('PYTHONPATH','')])
    cmd=[sys.executable,str(ROOT/'scripts/capture_fixed_torso_initial.py'),'--initial-configuration',str(initial),'--state',str(state),'--common-route',str(path),'--output',str(destination),'--port',str(port)]
    status=dict(side=record['side'],condition=record['condition'],source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),finished=False,approach_passed=False,grasp_passed=False,criteria_sha256=hashlib.sha256((ROOT/'config/geometric_closure.yaml').read_bytes()).hexdigest())
    with (destination/'replay.log').open('w') as log:
        process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:status['returncode']=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM);process.wait();status.update(returncode=-15,error='Native replay timeout')
    motion_path=destination/'native-route-initial/replay-result.json'
    if motion_path.exists():
        motion=json.loads(motion_path.read_text())
        if motion.get('source_sha256')==status['source_sha256']:
            status.update(approach_passed=motion['approach']['passed'],grasp_passed=motion['grasp_passed'],motion_result=str(motion_path.relative_to(ROOT)),closure_completed=motion['closure_completed'])
    else:status['error']=status.get('error','Replay did not produce a completed result; see replay.log')
    export_trace(destination);compress_intermediates(destination)
    status.update(finished=True,elapsed_s=time.monotonic()-start);write(status_path,status)
    print('DONE',status['side'],status['condition']['id'],'approach',status['approach_passed'],'grasp',status['grasp_passed'],'return',status['returncode'],'seconds',round(status['elapsed_s']),flush=True)
    return status


def report(out,rows):
    rows=[dict(r) for r in rows]
    for r in rows:
        mp=out/'replays'/r['side']/r['condition']['id']/'native-route-initial/replay-result.json'
        if not mp.exists():continue
        m=json.loads(mp.read_text());reasons=[]
        a=m['approach']
        if not a['passed']:
            if any(c['stage'] in ('opening','reach','settle') for c in m['actual_configuration_failures']):reasons.append('接近時の必要離隔不足')
            if a['completed'] and a['endpoint_position_error_m']>1e-4:reasons.append('接近終端位置誤差')
            if a['completed'] and a['endpoint_orientation_error_deg']>.5:reasons.append('接近終端姿勢誤差')
            if a['maximum_fixed_drift_m']>=2e-5:reasons.append('固定側の変位')
            if a['maximum_velocity_fraction']>1 or a['maximum_acceleration_fraction']>1:reasons.append('関節速度・加速度')
            if not a['actual_hard_joint_limits_passed']:reasons.append('関節角度範囲')
            if not reasons:reasons.append('接近再生条件')
        if m['virtual_closure_passed'] is False:reasons.append('閉鎖途中の離隔')
        grasp=m.get('planned_endpoint_grasp',m['actual_closed_grasp'])
        area=grasp['actual_projection'];r['capture_area_at_1mm_mm2']=area['area_m2']*1e6 if 'planned_endpoint_grasp' in m else None
        if min(area['area_m2'],area['largest_component_area_m2'])<4e-6 or area['area_fraction']<.05 or min(area['span_x_m'],area['span_z_m'])<.0005:reasons.append('布捕捉領域不足')
        if grasp['geometric_check_reason'] not in ('accepted','area'):reasons.append('把持形状: '+grasp['geometric_check_reason'])
        if m['virtual_closure_passed'] and not m['closure_dynamics_passed']:reasons.append('閉鎖の動力学条件')
        r.update(closure_completed=m['closure_completed'],failure_reasons=reasons,measured_steps=m['steps'],approach_metrics=m['approach'],endpoint_error_mm_text=f"{a['endpoint_position_error_m']*1000:.4f}",fixed_drift_um_text=f"{a['maximum_fixed_drift_m']*1e6:.3f}",closure_not_executed_reason=m.get('closure_not_executed_reason'))
        planned=mp.parent.parent/'planned-goal-grasp.json'
        if planned.exists():
            g=json.loads(planned.read_text());r.update(planned_goal_grasp_passed=g['passed'],planned_goal_capture_area_mm2=g['actual_projection']['area_m2']*1e6,planned_goal_reason=g['geometric_check_reason'])
            if g['geometric_check_reason']=='closed_foot':r['failure_reasons'].append('計画閉鎖時の皮膚離隔不足')
    groups=[]
    for side in ('right','left'):
        for stage in 'ABCDE':
            rs=[r for r in rows if r['side']==side and r['condition']['stage']==stage]
            groups.append(dict(side=side,stage=stage,total=len(rs),approach_passed=sum(r['approach_passed'] for r in rs),grasp_passed=sum(r['grasp_passed'] for r in rs),runtime_errors=sum(r['returncode']!=0 for r in rs),closure_completed=sum(r.get('closure_completed',False) for r in rs),closure_clearance_failed=sum('閉鎖途中の離隔' in r.get('failure_reasons',[]) for r in rs),capture_region_failed=sum('布捕捉領域不足' in r.get('failure_reasons',[]) for r in rs)))
    result=dict(expected=140,completed=len(rows),all_completed=len(rows)==140,approach_passed=sum(r['approach_passed'] for r in rows),grasp_passed=sum(r['grasp_passed'] for r in rows),groups=groups,rows=rows,criteria_file='config/geometric_closure.yaml',criteria_sha256=hashlib.sha256((ROOT/'config/geometric_closure.yaml').read_bytes()).hexdigest(),scope='PhysX arm/jaw replay and previous A geometric grasp surrogate with frozen cloth. Friction, cloth deformation and retention unvalidated.',learning_ready=False)
    result['physical_closure_completed']=sum(r.get('closure_completed',False) for r in rows)
    result['closure_skipped_by_capture_gate']=sum(bool(r.get('closure_not_executed_reason')) for r in rows)
    write(out/'summary.json',result)
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['side','stage','total','approach_passed','grasp_passed','runtime_errors','closure_completed','closure_clearance_failed','capture_region_failed']);w.writeheader();w.writerows(groups)
    with (out/'cases.csv').open('w') as f:
        fields=['side','id','stage','number','approach_passed','approach_completed','endpoint_position_error_mm','fixed_frame_drift_um','maximum_velocity_fraction','maximum_acceleration_fraction','planned_goal_grasp_passed','planned_capture_area_mm2','physical_closure_completed','closure_not_executed_reason','failure_reasons','source','source_sha256']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in sorted(rows,key=lambda r:(r['side'],r['condition']['id'])):
            a=r.get('approach_metrics',{})
            w.writerow(dict(side=r['side'],id=r['condition']['id'],stage=r['condition']['stage'],number=r['condition']['number'],approach_passed=r['approach_passed'],approach_completed=a.get('completed'),endpoint_position_error_mm=a.get('endpoint_position_error_m',0)*1000,fixed_frame_drift_um=a.get('maximum_fixed_drift_m',0)*1e6,maximum_velocity_fraction=a.get('maximum_velocity_fraction'),maximum_acceleration_fraction=a.get('maximum_acceleration_fraction'),planned_goal_grasp_passed=r.get('planned_goal_grasp_passed'),planned_capture_area_mm2=r.get('planned_goal_capture_area_mm2'),physical_closure_completed=r.get('closure_completed'),closure_not_executed_reason=r.get('closure_not_executed_reason'),failure_reasons='; '.join(r.get('failure_reasons',[])),source=r['source'],source_sha256=r['source_sha256']))
    planned_path=out/'planned-grasp-summary.json'
    planned=json.loads(planned_path.read_text()) if planned_path.exists() else None
    goal_text='' if planned is None else f"<p>元の計画終点での静的な把持条件：{planned['passed']}/{planned['total']}本が合格。捕捉面積の最大値 {planned.get('maximum_capture_area_mm2',0):.4f}mm²。閉鎖時の皮膚離隔で不合格 {planned.get('reason_counts',{}).get('closed_foot',0)}本。<a href='planned-grasp-summary.json'>全140計画終点の独立した把持判定</a>。</p>"
    table=''.join(f"<tr><td>{g['side']}</td><td>{g['stage']}</td><td>{g['total']}</td><td>{g['approach_passed']}</td><td>{g['grasp_passed']}</td><td>{g['closure_completed']}</td><td>{g['closure_clearance_failed']}</td><td>{g['capture_region_failed']}</td><td>{g['runtime_errors']}</td></tr>" for g in groups)
    details=''.join(f"<tr><td>{r['side']}</td><td>{r['condition']['id']}</td><td>{r['approach_passed']}</td><td>{r['grasp_passed']}</td><td>{r.get('endpoint_error_mm_text','未実施')}</td><td>{r.get('fixed_drift_um_text','未実施')}</td><td>{'、'.join(r.get('failure_reasons',[]))}</td><td><a href='replays/{r['side']}/{r['condition']['id']}/native-route-initial/replay-result.json'>再生結果</a> / <a href='replays/{r['side']}/{r['condition']['id']}/measured.csv'>実測CSV</a> / <a href='replays/{r['side']}/{r['condition']['id']}/replay.log'>ログ</a></td></tr>" for r in sorted(rows,key=lambda r:(r['side'],r['condition']['id'])))
    (out/'report.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>140軌道 動力学再生・把持判定</title><style>body{{font:16px/1.7 sans-serif;margin:30px;}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:5px 12px}}</style><h1>全軌道の動力学再生・把持条件検証</h1><p>完了 {len(rows)}/140。接近再生合格 {result['approach_passed']}、Aと同じ把持条件合格 {result['grasp_passed']}。</p><p>各軌道の保存形状・初期姿勢・補間・時間配分を維持。PhysX関節・指の実測値、接触記録、固定側変位、皮膚・全身の離隔、終端誤差、速度・加速度を検査。判定のために目標や姿勢を変更していない。</p>{goal_text}<p>物理閉鎖の完了 {result['physical_closure_completed']}本、把持条件ゲートによる閉鎖省略 {result['closure_skipped_by_capture_gate']}本。初期の試行には、離隔が許す場合に捕捉面積0でも閉鎖した記録を含みます。その後はAと同じ幾何把持ゲートを閉鎖の必要条件にしています。接近軌道とその判定基準は全件共通です。</p><p>入力軌道の終点は、指先中央面が開口縁の2mm外側に来る接近点です。この終点を保持して、従来Aの指腹の捕捉領域に布が入るかを評価しました。</p><p>把持条件は既存 config/geometric_closure.yaml と同じ。閉じ幅1mm、皮膚離隔1mm、捕捉面積・最大連結面積4mm²以上、面積比5%以上、両方向の幅0.5mm以上。布は保存形状で固定されているため、摩擦による保持・牽引性能は評価対象外。ロボット関節の重力は従来Aの再生設定と同じ無効で、重力荷重・駆動力上限の評価は含まない。全身距離検査は接近中0.1秒ごと、閉鎖・静止中0.02秒ごと、ネイティブ接触監視は全ステップで実施。サンプル間の連続衝突の証明は含まない。</p><table><tr><th>腕</th><th>開口</th><th>軌道数</th><th>接近再生合格</th><th>把持合格</th><th>1mm閉鎖完了</th><th>閉鎖離隔不足</th><th>捕捉領域不足</th><th>実行エラー</th></tr>{table}</table><p><a href="summary.json">全結果JSON</a> / <a href="summary.csv">集計CSV</a> / <a href="cases.csv">140軌道の判定一覧CSV</a></p><table><tr><th>腕</th><th>条件</th><th>接近再生</th><th>把持</th><th>接近終了時誤差 mm</th><th>固定側変位 µm</th><th>不合格理由</th><th>記録</th></tr>{details}</table></html>''')
    print('SUMMARY',len(rows),'/140','approach',result['approach_passed'],'grasp',result['grasp_passed'],flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/'artifacts/common-opening-approach/negative-wrist-start');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/common-opening-dynamics');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--port',type=int,default=20600);ap.add_argument('--timeout',type=int,default=3600);ap.add_argument('--match',default='');ap.add_argument('--stages',default='ABCDE');ap.add_argument('--report-only',action='store_true');a=ap.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    paths=sorted(a.source.resolve().glob('routes/*/*.json'));paths=[p for p in paths if a.match in str(p.relative_to(a.source.resolve())) and p.stem[0] in a.stages];rows=[]
    if a.report_only:
        rows=[json.loads(p.read_text()) for p in sorted((out/'replays').glob('*/*/batch-status.json'))];report(out,rows);return
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_one,p,out,a.source.resolve()/'initial.yaml',a.port+2*i,a.timeout) for i,p in enumerate(paths)]
        for future in as_completed(futures):
            rows.append(future.result());report(out,rows)

if __name__=='__main__':main()
