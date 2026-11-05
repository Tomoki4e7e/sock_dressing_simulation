#!/usr/bin/env python3
"""Publish the photo-derived start and separately labeled pregrasp validation."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-photo-reference-matplotlib')
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,REGISTERED_MANIFEST
from search_photo_reference_reaching import photo_context
from sock_dressing_simulation.grasp_geometry import HANDEDNESS


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/photo-reference-reaching');ap.add_argument('--selection',default='forearms-seeded');a=ap.parse_args();out=a.output.resolve();selected=out/a.selection
    initial=json.loads((selected/'initial-configurations.json').read_text())['candidates'][0]
    status=json.loads((selected/'trajectory-status.json').read_text());path=ROOT/next(r['trajectory'] for r in status['routes'] if r['passed']);plan=json.loads(path.read_text());validation=json.loads(path.with_name(path.name.replace('trajectory','validation')).read_text())
    n=initial['number'];state=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/state.json').read_text());k,c,dl,du,contract,scene=photo_context(n,state,initial['left_foot_floor_allowance_m'],REGISTERED_MANIFEST);c.park_left(initial['left_q_rad'],initial['left_gap_m']);d=np.array(initial['translation_m']);q=np.array(initial['right_q_rad']);frames=k.fk(q,initial['right_gap_m'])
    world=lambda name:k.A@frames[name][:3,3]+k.b
    elbows={s:world(s+'_arm/link_3').tolist() for s in ('right','left')};grippers={s:world(s+'_gripper/grasping_frame').tolist() for s in ('right','left')};span=sorted(p[0] for p in grippers.values());foot=np.array(state['source']['surfaces'][0]['vertices'])+d;foot_span=[float(foot[:,0].min()),float(foot[:,0].max())]
    between=bool(foot_span[0]>=span[0]-1e-9 and foot_span[1]<=span[1]+1e-9)
    if not between:raise ValueError('Foot is not between the newly configured grippers')
    initial_folder=selected/'native-initial' if (selected/'native-initial/result.json').exists() else selected/'native-route-initial'
    initial_image=initial_folder/('oblique.png' if initial_folder.name=='native-initial' else 'fixed-camera-after.png')
    native=json.loads((initial_folder/'result.json').read_text());motion=None
    motion_path=selected/'native-route-initial/replay-result.json'
    if motion_path.exists():
        v=json.loads(motion_path.read_text())
        if v['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest() and v['source']==str(path.relative_to(ROOT)):motion=v
    end_clearance=c.gripper_clearances(np.array(plan['q_rad'][-1]),plan['opening_gap_m'],d)
    human=np.array(scene['human_skin']['vertices'])+d;penetrating=human[human[:,1]<c.floor_top];pen_bounds=None if len(penetrating)==0 else [penetrating.min(0).tolist(),penetrating.max(0).tolist()]
    closure_issues=Counter(tuple(i['pair']) for v in validation['virtual_closure_checks'] for i in v['issues']);stats={}
    for name in ('screen','elbows-out/screen','comfortable/screen','continuous','photo-geometry','offset-geometry','elbows-above-hands/offset-geometry','facing-cuff/photo-geometry','facing-cuff/endpoints-only/photo-geometry'):
        rows=[json.loads(p.read_text()) for p in (out/name).glob('*.json')];cnt=Counter()
        for r in rows:
            if 'counts' in r:cnt.update(r['counts'])
            else:cnt.update(v['stage'] for v in r['attempts'])
        stats[name]=dict(records=len(rows),counts=dict(cnt))
    hand_directions={}
    for side in ('left','right'):
        T=frames[side+'_gripper/gripper_base'];R=k.A@T[:3,:3]@HANDEDNESS;tcp=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b;toward=np.array(initial['cuff_center_world_m'])-tcp;toward/=np.linalg.norm(toward)
        hand_directions[side]=dict(approach_axis_unity_world=R[:,0].tolist(),closing_axis_unity_world=R[:,1].tolist(),angle_to_cuff_center_deg=float(np.rad2deg(np.arccos(np.clip(R[:,0]@toward,-1,1)))))
    initial.update(elbow_world_m=elbows,grasping_frame_world_m=grippers,foot_x_interval_m=foot_span,new_gripper_x_interval_m=span,foot_between_new_grippers=between,hand_directions=hand_directions,
        original_fixed_lateral_interval_m=contract['gripper_x_interval_m'],floor_penetrating_human_bounds_m=pen_bounds,configuration='config/photo_reference_initial_pose.yaml')
    initial['elbow_to_wrist_drop_m']={s:{str(i):float(elbows[s][1]-world(s+'_arm/link_'+str(i))[1]) for i in (5,7)} for s in ('left','right')}
    initial['ankle_world_m']=(np.array(state['source']['ankle'])+d).tolist();initial['robot_base_world_m']=world('base_link').tolist()
    initial['ankle_lateral_error_m']=abs(initial['ankle_world_m'][0]-initial['robot_base_world_m'][0])
    if initial.get('forearms_down_required') and any(v<.02-1e-6 for side in initial['elbow_to_wrist_drop_m'].values() for v in side.values()):raise ValueError('A forearm slopes upward instead of the photographed downward posture')
    write(out/'selected-initial-pose.json',initial)
    summary=dict(number=n,initial=initial,offline_passed=validation['offline_passed'],native_initial_passed=native['passed'],native_replay_passed=motion['passed'] if motion else None,
        selected_trajectory=str(path.relative_to(ROOT)),endpoint_task=plan['endpoint_task'],closed_grasp_accepted=False,duration_s=plan['duration_s'],tcp_travel_m=validation['start_tcp_distance_m'],
        trajectory_validation=validation,native_replay=motion,end_gripper_clearances=end_clearance,
        closure_issue_pairs=[dict(pair=list(pair),count=count) for pair,count in closure_issues.items()],searches=stats,
        interpretation='The photograph is a qualitative reference without calibrated camera/joint data. Both elbows are above the hands and flared outward, both hands are beside the sock. Torso/head/base are fixed.',
        scope='New initial posture and collision-free pregrasp approach only. Saved cloth is held. Closed-jaw capture and transition from the old start are not accepted.')
    write(out/'summary.json',summary)
    status.update(executed=bool(motion and motion['executed']),native_replay_passed=motion['passed'] if motion else None,endpoint_task=plan['endpoint_task']);write(selected/'trajectory-status.json',status)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(13.8,4.4),constrained_layout=True)
    for ax,xy,title in zip(axs[:2],[(0,1),(2,1)],['Front: elbows above hands','Side: torso unchanged']):
        ax.scatter(foot[:,xy[0]],foot[:,xy[1]],s=4,color='gray',alpha=.4,label='Foot')
        for side,color in [('right','tab:blue'),('left','tab:orange')]:
            points=np.array([world(side+'_arm/link_'+str(i)) for i in (1,3,5,7)]+[grippers[side]])
            ax.plot(points[:,xy[0]],points[:,xy[1]],'-o',color=color,label=side+' arm',ms=4)
        tcp=np.array(plan['tcp_world_m']);ax.plot(tcp[:,xy[0]],tcp[:,xy[1]],color='tab:red',lw=3,label='Approach')
        ax.set(xlabel=('Unity X' if xy[0]==0 else 'Unity Z')+' (m)',ylabel='Unity Y (m)',title=title);ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
    ax=axs[2];t=np.array(plan['times_s']);angles=np.rad2deg(plan['q_rad'])
    for i in range(7):ax.plot(t,angles[:,i],label=f'J{i+1}')
    ax.set(xlabel='Time (s)',ylabel='Right arm angle (degrees)',title='Short approach; left arm held');ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
    fig.savefig(out/'pose-and-route.png',dpi=170);plt.close(fig)
    fmt=lambda vs:', '.join(f'{v:.2f}' for v in vs)
    motion_text='未実施' if motion is None else ('合格' if motion['passed'] else '不合格')
    actual='' if motion is None else f"終端誤差 {motion['endpoint_position_error_m']*1000:.4f}mm / {motion['endpoint_orientation_error_deg']:.5f}°、最大追従差 {motion['maximum_tracking_error_m']*1000:.4f}mm、固定側最大変位 {motion['maximum_fixed_drift_m']*1e6:.3f}µm。非除外の腕の負離隔接触 {len(motion['unexpected_contacts'])}件、実測姿勢の衝突検査不合格 {len(motion['actual_configuration_failures'])}件。"
    result_path=str(path.relative_to(out));csv_path=result_path.replace('.json','.csv')
    (out/'report.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>写真を基準にした初期姿勢と接近</title>
<style>body{{font:16px/1.75 sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#202b35}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd5dd;padding:6px 10px;text-align:left}}img{{max-width:100%}}a{{color:#0754a4}}pre{{background:#f0f3f5;padding:12px;white-space:pre-wrap}}</style>
<h1>写真を基準にした初期姿勢と接近</h1>
<p><strong>写真に近い初期配置と、靴下の縁へ向かう {validation['start_tcp_distance_m']*1000:.0f}mm の接近を生成。登録形状検査：合格。PhysXでの関節ドライブ再生：{motion_text}。</strong></p>
<p>閉鎖して布を挟む把持姿勢としては未採用。開いたグリッパーによる非干渉の接近と、終端での閉鎖把持を区別している。</p>
<h2>参照写真と採用した初期姿勢</h2><img src="../sock-pose-pattern/screen_shot 2026-11-05 121319.png" alt="ユーザー指定の参照写真">
<img src="{str(initial_image.relative_to(out))}" alt="両肘を外側・手先より上にしたUnity初期姿勢">
<p>写真のカメラ校正値や実関節角度は与えられていないため、完全な姿勢復元ではなく構えの近似。両肘を左右に張り、手先より50mm以上高く、両手を靴下の左右近くに構える条件を加えた。さらにグリッパーの前方軸を、靴下の縁へ向かう方向から40°以内に保つ条件を加えた。torsoは [−45°, 100°, 0°] に固定し、頭部と台車の指令値も維持した。左右とも、肘から前腕端・手首へ20mm以上下がることをFKで確認した。</p>
<table><tr><th>項目</th><th>採用値</th></tr><tr><td>右腕 J1～J7</td><td>[{fmt(initial['right_q_deg'])}]°</td></tr><tr><td>左腕 J1～J7</td><td>[{fmt(initial['left_q_deg'])}]°</td></tr><tr><td>人体・靴下・支持台の移動 X/Y/Z</td><td>[{fmt(d*1000)}] mm（+Xはロボットの左）</td></tr><tr><td>左足の実際の床貫通</td><td>{initial['grounded_left_foot_penetration_m']*1000:.2f}mm</td></tr><tr><td>初期の最小関節余裕</td><td>{initial['minimum_joint_margin_deg']:.2f}°</td></tr><tr><td>初期の右／左グリッパーと靴下の最短距離</td><td>{validation['start_gripper_clearances']['right']['cloth_m']*1000:.2f} / {validation['start_gripper_clearances']['left']['cloth_m']*1000:.2f}mm（被覆余白を含む保守値）</td></tr><tr><td>開き幅</td><td>左右40mm</td></tr><tr><td>足全体のX範囲</td><td>{foot_span[0]:.6f} ～ {foot_span[1]:.6f}m</td></tr><tr><td>新しい初期グリッパー間のX範囲</td><td>{span[0]:.6f} ～ {span[1]:.6f}m</td></tr></table>
<p>足全体が新しく構えた両グリッパー間にも収まる。床例外は人体の接地した左足について設定し、上げた右脚・靴下・支持台・ロボットの床制約と、人体に対する腕の衝突検査は維持した。探索上限は150mmだが、採用配置の床貫通は上表の値。可動域の余裕を示したもので、関節トルクの測定ではない。</p>
<img src="pose-and-route.png" alt="腕と足の位置関係、接近軌道、関節角度">
<h2>接近と閉鎖の検証</h2><p>靴下の縁への距離が縮まる接近方向を採用。接近距離 {validation['start_tcp_distance_m']*1000:.1f}mm、時間 {plan['duration_s']:.3f}秒、時間補間後 {validation['sample_count']}点を登録済み3,688部品で検査した。必要離隔に対する最小余裕は {validation['minimum_slack_m']*1000:.4f}mm。{actual}</p>
<p>開始姿勢のUnity読出しも合格。torso・頭部・台車の最大位置差は {native['maximum_fixed_frame_position_error_m']*1e6:.3f}µm、観測カメラの位置・姿勢は維持した。下の外部確認画像のカメラ移動は、その比較検査後に行っている。</p>
<p>終端からの仮想閉鎖は不合格。右の指と人体・右脚皮膚の必要離隔が不足する。したがって、この短い接近を「靴下を掴める把持軌道」として採用していない。布上の点や手首姿勢を変える追加探索も記録したが、今回の写真に近い構えから閉鎖条件まで通過した候補は未確認。</p>
<img src="{selected.name}/native-route-initial/fixed-camera-endpoint.png" alt="固定カメラによる接近終端">
<h2>設定と保存データ</h2><ul><li><a href="../../config/photo_reference_initial_pose.yaml">初期姿勢設定 YAML</a> ／ <a href="selected-initial-pose.json">配置と幾何情報 JSON</a></li><li><a href="{result_path}">接近軌道 JSON</a> ／ <a href="{csv_path}">関節角度 CSV</a></li><li><a href="{result_path.replace('trajectory','validation')}">衝突・開口・閉鎖検査</a></li><li><a href="{str((initial_folder/'result.json').relative_to(out))}">初期姿勢のUnity検査</a></li><li><a href="{selected.name}/native-route-initial/replay-result.json">接近の物理再生結果</a></li><li><a href="summary.json">探索・検査の集計</a></li></ul>
<h2>再現</h2><pre>LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/capture_fixed_torso_initial.py --initial-configuration config/photo_reference_initial_pose.yaml --output artifacts/photo-reference-reaching/{selected.name}

LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/capture_fixed_torso_initial.py --output artifacts/photo-reference-reaching/{selected.name} --trajectory {str(path.relative_to(ROOT))} --replay</pre>
<p>対象は保存した布形状を保持した状態で、新しい初期姿勢から接近する部分。旧エプロン姿勢からの移行、自由に変形する布の把持保持、サンプル間の連続衝突の形式的証明は含まない。</p></html>''')
    print('PHOTO REPORT',validation['offline_passed'],motion_text,'travel mm',validation['start_tcp_distance_m']*1000,'duration',plan['duration_s'],flush=True)


if __name__=='__main__':main()
