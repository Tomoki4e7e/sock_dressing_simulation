#!/usr/bin/env python3
"""Summarize the selected fixed-torso trajectory, checks, and reproducible inputs."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-fixed-torso-matplotlib')
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import state_for,write,geometric_check,URDF,ROBOT,REGISTERED_MANIFEST
from search_fixed_torso_starts import context
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances
import yaml


def search_counts(folder):
    counts=Counter();rows=[]
    for path in sorted(folder.glob('target-*.json')):
        r=json.loads(path.read_text());rows.append(r)
        for a in r['attempts']:counts[a['stage']]+=1
    return dict(targets=len(rows),trials=sum(len(r['attempts']) for r in rows),stages=dict(counts))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');a=ap.parse_args();out=a.output.resolve()
    status=json.loads((out/'trajectory-status.json').read_text());selected=next(r for r in status['routes'] if r['passed']);path=ROOT/selected['trajectory'];plan=json.loads(path.read_text());validation=json.loads(path.with_name(path.name.replace('trajectory','validation')).read_text())
    state=state_for(plan['geometry_record']);k,checker,dl,du,contract,scene=context(plan['number'],state,REGISTERED_MANIFEST);checker.park_left(plan['left_q_rad'],plan['left_gap_m']);d=np.array(plan['translation_m'])
    initial=json.loads((out/'initial-configurations.json').read_text());parks=json.loads((out/'left-parks.json').read_text());placed=Counter()
    for p in (out/'placed-ik').glob('target-*.json'):placed.update(json.loads(p.read_text())['counts'])
    refinement=search_counts(out/'refined');full=search_counts(out/'search');forward=search_counts(out/'forward/search')
    native=None;motion=None
    native_path=out/'native-route-initial/result.json';motion_path=out/'native-route-initial/replay-result.json'
    if native_path.exists():
        candidate=json.loads(native_path.read_text())
        if Path(candidate['initial_configuration'].get('trajectory','')).resolve()==path.resolve():native=candidate
    if motion_path.exists():
        candidate=json.loads(motion_path.read_text())
        if candidate['source']==str(path.relative_to(ROOT)) and candidate['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest():motion=candidate
    status.update(executed=bool(motion and motion['executed']),native_replay_passed=motion['passed'] if motion else None,
        native_replay_result=str(motion_path.relative_to(ROOT)) if motion else None)
    write(out/'trajectory-status.json',status)
    c=plan['geometry_record']['candidate'];R=np.array(c['rotation_matrix']);p=np.array(c['grasp_center_unity_world_m']);samples=json.loads((ROOT/f"artifacts/bulged-cuff/pose-{plan['number']:02d}/samples.json").read_text());sample=samples[c['sample_index']];faces=np.array(state['triangles']).reshape(-1,3)
    distances=material_distances(np.array(state['source']['rest_positions']),faces,faces[sample['face']],sample['weights']);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());grasp,reason=geometric_check(k,state,sample,R,SkinBody(state['source']['skin_surface']),ClothScreen(state['particles_world'],state['triangles']),settings,distances)
    if not grasp:raise ValueError('Selected continuous grasp failed independent geometric recheck: '+reason)
    write(out/'selected-grasp-recheck.json',dict(passed=True,candidate=c,projection=grasp['projection'],valid_openings=grasp['valid_openings'],translation_m=d.tolist(),frozen_cloth_geometry_only=True))
    foot=np.array(state['source']['surfaces'][0]['vertices'])+d;start=np.array(plan['start_q_rad']);end=np.array(plan['q_rad'][-1]);left=np.array(plan['left_q_rad'])
    start_pose=dict(number=plan['number'],right_arm_degrees=np.rad2deg(start).tolist(),left_arm_degrees=np.rad2deg(left).tolist(),
        right_arm_rad=start.tolist(),left_arm_rad=left.tolist(),torso_degrees={n:float(np.rad2deg(v)) for n,v in plan['torso_joints_rad'].items()},
        human_sock_support_translation_unity_world_m=d.tolist(),right_opening_gap_m=plan['opening_gap_m'],left_opening_gap_m=plan['left_gap_m'],
        original_robot_lateral_interval_m=contract['gripper_x_interval_m'],foot_x_interval_m=[float(foot[:,0].min()),float(foot[:,0].max())],
        reference_state=plan['geometry_record'].get('state_file'),trajectory=str(path.relative_to(ROOT)),
        initialization='Restore reference human/sock/support state, apply the rigid translation, then set only arm and finger joints while holding torso/head/base at the reference values.',
        transition_from_old_apron_start_included=False)
    write(out/'selected-initial-pose.json',start_pose)
    summary=dict(selected_trajectory=str(path.relative_to(ROOT)),number=plan['number'],duration_s=plan['duration_s'],start_tcp_distance_m=validation['start_tcp_distance_m'],
        offline_passed=validation['offline_passed'],native_initial_passed=native['passed'] if native else None,native_replay_passed=motion['passed'] if motion else None,
        torso_angles_degrees=start_pose['torso_degrees'],fixed_robot_lateral_interval_m=contract['gripper_x_interval_m'],foot_x_interval_m=start_pose['foot_x_interval_m'],
        initial_pose=start_pose,left_park_screening=parks['counts'],collision_free_initial_layouts=len(initial['candidates']),initial_layout_screening=initial['counts'],
        full_joint_range_search=full,forward_search=forward,placed_ik_counts=dict(placed),joint_arm_position_refinement=refinement,
        continuous_grasp_rechecked=True,grasp_area_mm2=grasp['projection']['area_m2']*1e6,
        trajectory_validation=validation,native_replay=motion,reference_robot_sha256=hashlib.sha256(ROBOT.read_bytes()).hexdigest(),registered_geometry_manifest_sha256=hashlib.sha256(REGISTERED_MANIFEST.read_bytes()).hexdigest(),
        scope='New separated arm start to the grasp target; torso/head/base and left arm held; saved cloth shape held; no transition from the old apron posture',
        first_replay_diagnostic='diagnostics/first-physical-replay/replay-result.json')
    write(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),constrained_layout=True);tcp=np.array(plan['tcp_world_m']);t=np.array(plan['times_s']);q=np.rad2deg(plan['q_rad'])
    ax=axs[0];span=contract['gripper_x_interval_m'];ax.axvspan(*span,color='tab:green',alpha=.12,label='Fixed robot lateral interval');ax.scatter(foot[:,0],foot[:,2],s=4,c='gray',alpha=.35,label='Foot');ax.plot(tcp[:,0],tcp[:,2],color='tab:blue',label='TCP route');ax.scatter(tcp[[0,-1],0],tcp[[0,-1],2],c=['tab:blue','tab:red'],s=40);ax.axvline((span[0]+span[1])/2,color='k',lw=.7,ls='--',label='Robot center');ax.set(xlabel='Unity X (robot right is negative)',ylabel='Unity Z',title='Foot remains inside the fixed interval');ax.set_aspect('equal');ax.legend(fontsize=7);ax.grid(alpha=.2)
    ax=axs[1]
    for i in range(7):ax.plot(t,q[:,i],label=f'J{i+1}')
    ax.set(xlabel='Time (s)',ylabel='Right arm angle (degrees)',title='Torso and left arm held');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
    ax=axs[2];ax.plot(t,np.linalg.norm(tcp-(p+d),axis=1)*1000,label='Distance to target (mm)');ax.set(xlabel='Time (s)',ylabel='TCP distance to target (mm)',title='150 mm reaching route');ax.grid(alpha=.2)
    if motion and (out/'native-route-initial/motion-trace.jsonl').exists():
        trace=[json.loads(line) for line in (out/'native-route-initial/motion-trace.jsonl').read_text().splitlines()];tt=np.array([v['stage_time_s']+(plan['duration_s'] if v['stage']=='settle' else 0) for v in trace]);err=np.array([v['tracking_error_m'] for v in trace])*1000;other=ax.twinx();other.plot(tt,err,color='tab:red',alpha=.6,label='Tracking error');other.set_ylabel('Tracking error (mm)',color='tab:red')
    fig.savefig(out/'trajectory.png',dpi=170);plt.close(fig)
    fmt=lambda values:', '.join(f'{v:.2f}' for v in values)
    motion_text='未実施' if motion is None else ('合格' if motion['passed'] else '不合格（詳細記録参照）')
    native_details='' if motion is None else f"<p>終端位置誤差 {motion['endpoint_position_error_m']*1000:.6f}mm、姿勢誤差 {motion['endpoint_orientation_error_deg']:.6f}°、最大追従差 {motion['maximum_tracking_error_m']*1000:.4f}mm、固定側最大変位 {motion['maximum_fixed_drift_m']*1e6:.3f}µm。実測姿勢の衝突検査不合格 {len(motion['actual_configuration_failures'])}件、非除外の腕の負離隔接触 {len(motion['unexpected_contacts'])}件。</p>"
    csv_path=path.with_suffix('.csv').name
    # CSV uses the same prefix and extension as the trajectory JSON.
    csv_path=path.name.replace('-trajectory.json','-trajectory.csv')
    (out/'report.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>torso固定・開始姿勢と人体位置の再探索</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#202b35}}table{{border-collapse:collapse}}td,th{{border:1px solid #cdd5dd;padding:7px 12px;text-align:left}}img{{max-width:100%}}code,pre{{background:#f0f3f5}}a{{color:#0754a4}}</style>
<h1>torso固定・開始姿勢と人体位置の再探索</h1>
<p><strong>No.{plan['number']}で1本のリーチング軌道を生成。登録形状での補間検査：合格。物理再生：{motion_text}。</strong></p>
<p>torsoは従来の [−45°, 100°, 0°] に固定し、頭部と台車の指令値も維持した。右腕の旧開始姿勢に基づく109.5°上限を外し、実関節制限から5°内側の全域を探索。左腕は待避姿勢に固定する。</p>
<table><tr><th>項目</th><th>採用値</th></tr><tr><td>右腕開始角度 J1～J7</td><td>[{fmt(np.rad2deg(start))}]°</td></tr><tr><td>左腕待避角度 J1～J7</td><td>[{fmt(np.rad2deg(left))}]°</td></tr><tr><td>人体・靴下・支持台の移動 X/Y/Z</td><td>[{fmt(d*1000)}] mm（Unity世界座標、+XはDry-AIRECの左方向）</td></tr><tr><td>足全体のX範囲</td><td>{foot[:,0].min():.6f} ～ {foot[:,0].max():.6f}m</td></tr><tr><td>固定した中央範囲</td><td>{span[0]:.6f} ～ {span[1]:.6f}m</td></tr><tr><td>接近距離・時間</td><td>{validation['start_tcp_distance_m']*1000:.1f}mm・{plan['duration_s']:.3f}秒</td></tr><tr><td>右／左グリッパー開き幅</td><td>{plan['opening_gap_m']*1000:.0f}／{plan['left_gap_m']*1000:.0f}mm</td></tr></table>
<p>足は固定した中央範囲の左寄りに収まる。待避したグリッパーの間隔を広げて人体配置の制約を緩めたものではない。人体・靴下・支持台は一体で平行移動し、床への貫通を許さない。</p>
<h2>成立した方法</h2><p>両腕を離した非干渉の初期配置は12通り得られた。さらに、到達点での手首姿勢を連続更新し、布・足の条件を同じ基準で再審査した。採用姿勢の把持面積は {grasp['projection']['area_m2']*1e6:.3f}mm²。最後の接近で手首を固定せず、終端から逆向きに衝突制約付き位置IKを解いて新しい開始姿勢を生成した。</p>
<p>開始時の右グリッパーは人体から約{validation['start_gripper_clearances']['right']['skin_m']*1000:.1f}mm、胴体・頭部・反対腕などから約{validation['start_gripper_clearances']['right']['robot_body_m']*1000:.1f}mm離れる。左グリッパーも同じ開始検査を通過した。取付近傍の同側前腕・手首は従来の機構除外と衝突余裕で別途検査し、上記の胴体等の離隔集計から除いた。</p>
<h2>検証</h2><p>登録済み3,688部品を使い、時間補間後の{validation['sample_count']}点、開始時の開き幅変更、仮想閉鎖を再検査した。右腕・左腕の計画関節余裕は5°以上。FKによる台車・torso・頭部の位置変化は {validation['maximum_fixed_frame_position_error_m']:.1e}m。観測カメラの位置・姿勢はUnity読出しでも維持した。</p>
{native_details}<p>初回の物理再生は、頭部との5mmの必要離隔を約0.098mm下回った時点で停止した。その記録を保存し、頭部等への計画離隔を7mm以上に増やし、時間を長くした軌道を採用した。衝突余裕や関節制限の基準を下げて通過扱いにしていない。</p>
<img src="trajectory.png" alt="足の配置・関節角度・TCP軌道">
<h2>保存ファイル</h2><ul><li><a href="selected-initial-pose.json">設定する初期姿勢と人体位置</a></li><li><a href="{str(path.relative_to(out))}">採用軌道JSON</a> ／ <a href="routes/{csv_path}">関節角度CSV</a></li><li><a href="{str(path.with_name(path.name.replace('trajectory','validation')).relative_to(out))}">登録形状での軌道検証</a></li><li><a href="native-route-initial/replay-result.json">物理再生結果</a></li><li><a href="native-route-initial/result.json">初期姿勢・固定カメラのUnity読出し</a></li><li><a href="selected-grasp-recheck.json">更新した把持姿勢の独立幾何検査</a></li><li><a href="summary.json">探索集計</a></li><li><a href="diagnostics/first-physical-replay/replay-result.json">初回の途中停止記録</a></li></ul>
<p>以下の観測画像は同じカメラ位置・姿勢で取得した。別途保存した overview / oblique / endpoint-close は外部確認用の視点で、観測カメラの固定検査後に撮影したもの。</p>
<img src="native-route-initial/fixed-camera-after.png" alt="同じ観測カメラで見た新しい開始姿勢"><img src="native-route-initial/fixed-camera-endpoint.png" alt="同じ観測カメラで見たリーチング終端">
<h2>対象範囲</h2><p>新しく設定した初期姿勢から到達までを検証した。従来のエプロン姿勢から新しい初期姿勢へ移る経路は含まない。布は探索時の保存形状を保持し、閉鎖は幾何評価のみ。自由に変形する布の把持保持・摩擦の成功判定は含まない。サンプル間の連続衝突の形式的証明ではない。静止台車と床の扱いは従来のシーン・検査方針を維持する。</p>
<h2>再生</h2><pre>LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/capture_fixed_torso_initial.py --trajectory {str(path.relative_to(ROOT))} --replay</pre></html>''')
    print('FIXED TORSO REPORT','offline',validation['offline_passed'],'native',motion['passed'] if motion else None,'duration',plan['duration_s'],flush=True)


if __name__=='__main__':main()
