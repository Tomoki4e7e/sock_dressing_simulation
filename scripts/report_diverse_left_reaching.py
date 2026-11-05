#!/usr/bin/env python3
"""Summarize verified shared-start routes, workbook coverage, and spatial diversity."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-diversity-matplotlib')
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import zipfile
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write
from sock_dressing_simulation.pose_patterns import read_patterns
from sock_dressing_simulation.reaching_dataset import file_sha256,verify_dataset_inputs,spatial_route_distance


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/final');a=ap.parse_args();out=a.output.resolve()
    rows=[];plans={};groups=defaultdict(list)
    for path in sorted((out/'routes').glob('*-trajectory.json')):
        plan=json.loads(path.read_text());verify_dataset_inputs(plan,ROOT)
        validation=json.loads(path.with_name(path.name.replace('trajectory','validation')).read_text())
        if validation['trajectory_sha256']!=file_sha256(path):raise ValueError('Stale validation: '+str(path))
        pattern=plan['workbook_pattern'];dest=out/'replays'/path.stem
        native_path=dest/'native-route-initial/replay-result.json';initial_path=dest/'native-route-initial/result.json';closed_path=dest/'actual-closed-grasp-recheck.json'
        native=json.loads(native_path.read_text()) if native_path.exists() else {}
        initial=json.loads(initial_path.read_text()) if initial_path.exists() else {}
        closed=json.loads(closed_path.read_text()) if closed_path.exists() else {}
        current=native.get('source_sha256')==file_sha256(path)
        offline=validation['offline_passed'];executed=bool(current and initial.get('passed') and native.get('passed'));captured=bool(executed and closed.get('passed'))
        row=dict(id=path.stem,number=plan['number'],pitch_deg=pattern['pitch_deg'],roll_deg=pattern['roll_deg'],stage=plan['applied_opening_stage'],workbook_stage=pattern['stage'],
            offline_passed=offline,native_replay_passed=executed,actual_closed_capture_passed=captured,
            trajectory=str(path.relative_to(ROOT)),duration_s=plan['duration_s'],minimum_joint_margin_deg=validation['minimum_joint_margin_deg'],minimum_slack_mm=validation['minimum_slack_m']*1000,
            planned_area_mm2=validation['actual_grasp_geometry']['projection']['area_m2']*1e6 if validation['actual_grasp_geometry'] else None,
            actual_area_mm2=closed.get('actual_projection',{}).get('area_m2',0)*1e6 if captured else None,
            actual_skin_clearance_mm=closed.get('closed_skin_clearance_m',0)*1000 if captured else None,
            endpoint_error_mm=native.get('endpoint_position_error_m',0)*1000 if executed else None,
            maximum_fixed_drift_um=native.get('maximum_fixed_drift_m',0)*1e6 if executed else None,
            measured_angles=str((dest/'angle.csv').relative_to(ROOT)) if captured else None)
        rows.append(row);plans[row['id']]=plan
        if offline:groups[(row['pitch_deg'],row['roll_deg'],row['stage'])].append(row)
    diversity=[];measured_paths={}
    def measured_path(row):
        if row['id'] not in measured_paths:
            points=[]
            with (out/'replays'/row['id']/'native-route-initial/motion-trace.jsonl').open() as stream:
                for line in stream:
                    sample=json.loads(line)
                    if sample['stage']=='reach':points.append(sample['tcp_world_m'])
            measured_paths[row['id']]=points
        return measured_paths[row['id']]
    for key,values in groups.items():
        for i,first in enumerate(values):
            for second in values[i+1:]:
                distance=spatial_route_distance(plans[first['id']]['tcp_world_m'],plans[second['id']]['tcp_world_m'])
                measured=spatial_route_distance(measured_path(first),measured_path(second)) if first['actual_closed_capture_passed'] and second['actual_closed_capture_passed'] else None
                if measured is not None and measured<.005:raise ValueError('Measured routes lack 5 mm spatial diversity')
                diversity.append(dict(condition=list(key),first=first['id'],second=second['id'],rms_tcp_separation_mm=distance*1000,measured_rms_tcp_separation_mm=None if measured is None else measured*1000,passed=distance>=.005 and (measured is None or measured>=.005)))
    patterns=read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
    verified={(r['pitch_deg'],r['roll_deg'],r['stage']) for r in rows if r['actual_closed_capture_passed']}
    coverage=[dict(number=p.number,pitch_deg=p.pitch_deg,roll_deg=p.roll_deg,stage=p.stage,angle_and_stage_covered=(p.pitch_deg,p.roll_deg,p.stage) in verified) for p in patterns]
    searched=[]
    initial_file=ROOT/'config/diverse_left_reaching_shared.yaml'
    for folder in out.parent.iterdir():
        if not folder.is_dir() or 'probe' not in folder.name:continue
        for path in sorted(folder.glob('pose-*.json')):
            probe=json.loads(path.read_text())
            if probe.get('initial_sha256')!=file_sha256(initial_file):continue
            searched.append(dict(number=probe['number'],stage='A' if probe['stage']=='held-A' else probe['stage'],endpoint_found=bool(probe['accepted']),initial_passed=probe['initial_check']['passed'],counts=probe.get('counts',{}),elapsed_s=probe.get('elapsed_s'),record=str(path.relative_to(ROOT))))
    summary=dict(shared_initial_configuration=next(iter(plans.values()))['shared_initial_configuration'] if plans else None,
        offline_routes=sum(r['offline_passed'] for r in rows),native_verified_routes=sum(r['actual_closed_capture_passed'] for r in rows),
        verified_conditions=len(verified),verified_ankle_angles=len({c[:2] for c in verified}),verified_opening_stages=sorted({c[2] for c in verified}),
        rows=rows,spatial_diversity=diversity,workbook_coverage=coverage,searched_conditions=searched,
        scope='Common robot posture and scene placement. Workbook pitch/roll and opening stage only; sock material, shifts, diagonal, wrinkles, folds and opening collapse are not reproduced. Frozen cloth geometric closure, not friction retention or pull-up.',learning_ready=False)
    write(out/'summary.json',summary)
    measured_audit=[]
    for row in rows:
        if not row['actual_closed_capture_passed']:continue
        dest=out/'replays'/row['id'];angles=np.loadtxt(dest/'angle.csv',delimiter=',',ndmin=2)
        motion=json.loads((dest/'native-route-initial/replay-result.json').read_text())
        frame_rows=list(csv.DictReader((dest/'frames.csv').open()))
        times=np.array([float(f['time_s']) for f in frame_rows])
        passed=bool(angles.shape==(motion['steps'],18) and len(frame_rows)==len(angles) and np.isfinite(angles).all() and np.allclose(angles[:,8],.5*angles[:,7],rtol=0,atol=1e-14) and np.allclose(angles[:,17],.5*angles[:,16],rtol=0,atol=1e-14) and np.allclose(np.diff(times),.02,rtol=0,atol=1e-12))
        measured_audit.append(dict(id=row['id'],passed=passed,rows=len(angles),columns=angles.shape[1],angle_sha256=file_sha256(dest/'angle.csv'),frames_sha256=file_sha256(dest/'frames.csv')))
        if not passed:raise ValueError('Recorded demonstration failed schema/time audit: '+row['id'])
    write(out/'validation/recorded-data-audit.json',dict(passed=all(r['passed'] for r in measured_audit),records=measured_audit))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['id']);writer.writeheader();writer.writerows(rows)
    with (out/'demonstrations.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['id']);writer.writeheader();writer.writerows(r for r in rows if r['actual_closed_capture_passed'])
    if plans:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(111,projection='3d')
        for row in rows:
            if not row['offline_passed']:continue
            tcp=np.array(plans[row['id']]['tcp_world_m']);ax.plot(tcp[:,0],tcp[:,2],tcp[:,1],label=f"{row['number']:02d} {row['stage']} v{row['id'].split('-v')[-1].split('-')[0]}")
        ax.set(xlabel='Unity X (m)',ylabel='Unity Z (m)',zlabel='Unity Y (m)',title='Shared-start left-hand reaching paths')
        ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.,1.));fig.tight_layout();fig.savefig(out/'trajectories.png',dpi=170);plt.close(fig)
    cells=[];gallery=[]
    for row in rows:
        state='実再生・閉鎖検査合格' if row['actual_closed_capture_passed'] else ('幾何検査合格・実再生未合格' if row['offline_passed'] else '不合格')
        cells.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [row['id'],row['pitch_deg'],row['roll_deg'],row['stage'],state,round(row['duration_s'],2),round(row['minimum_slack_mm'],3),None if row['actual_area_mm2'] is None else round(row['actual_area_mm2'],3)])+'</tr>')
        image=out/'replays'/row['id']/'native-route-initial/endpoint-close.png'
        if image.exists():gallery.append(f'<figure><img src="{image.relative_to(out)}"><figcaption>{html.escape(row["id"])}</figcaption></figure>')
    separation=', '.join(f'{d["measured_rms_tcp_separation_mm"]:.2f} mm（実測）' if d['measured_rms_tcp_separation_mm'] is not None else f'{d["rms_tcp_separation_mm"]:.2f} mm（計画）' for d in diversity) or '比較対象なし'
    content=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>共通初期姿勢からの左手リーチング</title>
<style>body{{font:16px sans-serif;max-width:1250px;margin:40px auto;padding:0 20px;line-height:1.7}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #ccd;padding:6px}}img{{max-width:100%}}.gallery{{display:flex;flex-wrap:wrap}}figure{{width:45%;margin:2%}}code{{word-break:break-all}}</style>
<h1>共通初期姿勢からの左手リーチング</h1>
<p>幾何検査合格 {summary['offline_routes']} 本、PhysX実再生・実測閉鎖検査合格 {summary['native_verified_routes']} 本。実再生済み条件 {summary['verified_conditions']} 種、足首角度 {summary['verified_ankle_angles']} 種、開口位置 {', '.join(summary['verified_opening_stages'])}。</p>
<p>共通設定: <code>{summary['shared_initial_configuration']}</code>。成功例から両腕の第2関節だけ−3°。人体・台座・ロボットの配置、torso・head・baseは成功例と共通。全軌道に同じ開始関節角・初期開き幅を要求し、入力ファイルのSHA-256を検証する。</p>
<p>同一条件内の経路差（時間を除き、弧長を正規化した手先位置のRMS）: {separation}。別軌道の採用閾値は5 mm。</p>
<img src="trajectories.png" alt="手先の三次元軌道"><table><tr><th>軌道</th><th>pitch°</th><th>roll°</th><th>開口位置</th><th>検証状態</th><th>接近時間 s</th><th>最小余裕 mm</th><th>実測捕捉面積 mm²</th></tr>{''.join(cells)}</table>
<div class="gallery">{''.join(gallery)}</div>
<h2>適用範囲</h2><p>足首角度と開口位置A〜Eを参照した探索。Aは成功例と同じ保存済み形状、B〜Eは既存の解剖学的経路による正常開口形状を使用。Eは踵・足首位置から下腿方向50 mmという既存定義。Excelの素材・前後ずらし・斜め角・しわ・ねじれ・折り返し・開口のつぶれは未再現。</p>
<p>開口位置C〜Eも探索したが、今回の探索範囲では、閉鎖条件と腕IK・全身衝突条件を併せて満たす軌道は得られていない。探索記録はsummary.jsonのsearched_conditionsに保存する。表の条件とExcel行の足首角度・開口位置の対応はworkbook_coverageに記録する。足首角度だけを参照した追加条件も含むため、Excelの全50行を再現した結果ではない。</p>
<p>足との離隔1 mm以上、捕捉面積4 mm²以上、全身・自己干渉、関節余裕5°以上、速度・加速度を検査。保存布形状での接近・幾何学的な閉鎖検査であり、布変形、摩擦保持、牽引は未検証。従来と同じ人体左足の床貫通約96.6 mmと150 mmの明示的許容を継承する。その他の床検査は維持する。</p>
<p>実再生合格軌道の各フォルダにangle.csv（実測18列）、frames.csv、demonstration.jsonを出力。合格分の関節データは<a href="demonstrations.zip">demonstrations.zip</a>にもまとめる。トルクと同期画像は未収集のため、学習可能データとするフラグはfalse。軌道・検証・再生ログ・Excel対応はsummary.jsonとsummary.csvを参照。</p>
<p>関連する57テストは合格。全体試験は511合格・2未通過（既存の再生フック試験の引数不足1件、検証環境のtorch未導入1件）。ログと環境情報はvalidation/に保存。</p></html>'''
    (out/'report.html').write_text(content)
    with zipfile.ZipFile(out/'demonstrations.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(initial_file,'shared_initial_configuration.yaml')
        archive.write(out/'summary.json','summary.json');archive.write(out/'summary.csv','summary.csv')
        archive.write(out/'demonstrations.csv','demonstrations.csv')
        for row in rows:
            if not row['actual_closed_capture_passed']:continue
            dest=out/'replays'/row['id']
            for name in ('angle.csv','frames.csv','demonstration.json','actual-closed-grasp-recheck.json'):
                archive.write(dest/name,row['id']+'/'+name)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('rows','spatial_diversity','workbook_coverage','searched_conditions')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
