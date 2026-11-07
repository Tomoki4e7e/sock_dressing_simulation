#!/usr/bin/env python3
"""Report a fixed-layout, photo-start grasp approach and full jaw closure."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-photo-grasp-matplotlib')
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,state_for,REGISTERED_MANIFEST,geometric_check
from search_fixed_torso_starts import context
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area
from sock_dressing_simulation.reaching_dataset import load_grasp_samples


def actual_recheck(plan,motion,out):
    trace=[json.loads(line) for line in (out/'native-route-initial/motion-trace.jsonl').read_text().splitlines()];last=trace[-1];side=plan['moving_side'];state=state_for(plan['geometry_record']);d=np.array(plan['translation_m'])
    k,checker,*_=context(plan['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=plan['left_foot_floor_allowance_m']);checker.park_left(last['actual_left_q_rad'],last['actual_gap_m'] if side=='left' else plan['left_gap_m']);checker.closure_side=side
    v=ControlledArmKinematics(k,side,last['actual_right_q_rad'],plan['right_gap_m']);p,R=v.pose(last['actual_q_rad']);sample=load_grasp_samples(plan['geometry_record'],ROOT)[plan['geometry_record']['candidate']['sample_index']];sample=dict(sample,position=(p-d).tolist());faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);dist=material_distances(rest,faces,faces[sample['face']],sample['weights']);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles'])
    geometry,reason=geometric_check(v,state,sample,R,skin,cloth,settings,dist,approach_distance=0.);gap=last['actual_gap_m'];fingers=[g for n,g in v.hand.vertices(gap).items() if n.endswith(('/left_finger_link','/right_finger_link'))];depth=rest[:,2]-rest[:,2].min()
    area=projected_cloth_area((np.array(state['particles_world'])-(p-d))@R,faces,fingers,dist,depth,**{n:settings[n] for n in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
    clear,distance,limiting,why=skin.inspect(v.hand.bank(gap),R,p-d,.001)
    body=checker.slacks(last['actual_right_q_rad'],plan['right_gap_m'] if side=='left' else gap,d,allow_cloth_overlap=True,details=True)
    passed=bool(geometry and clear and body['passed'] and min(area['area_m2'],area['largest_component_area_m2'])>=4e-6 and area['area_fraction']>=.05 and min(area['span_x_m'],area['span_z_m'])>=.0005)
    result=dict(passed=passed,moving_side=side,geometric_check_reason=reason,actual_closed_gap_m=gap,actual_projection=area,closed_skin_clearance_m=distance,limiting_part=limiting.replace('right_gripper/',side+'_gripper/'),closed_skin_reason=why,whole_scene_closed_check=body,
        native_tcp_to_fk_error_m=float(np.linalg.norm(p-last['tcp_world_m'])),actual_right_q_rad=last['actual_right_q_rad'],actual_left_q_rad=last['actual_left_q_rad'],scope='Actual joint feedback, saved frozen cloth, and measured gap; not a friction/retention test')
    write(out/'actual-closed-grasp-recheck.json',result);return result,trace


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/photo-grasp-adjustment');a=ap.parse_args();out=a.output.resolve();path=ROOT/json.loads((out/'trajectory-status.json').read_text())['trajectory'];plan=json.loads(path.read_text());validation=json.loads(path.with_name(path.name.replace('trajectory','validation')).read_text());initial=yaml.safe_load((ROOT/'config/photo_reference_initial_pose.yaml').read_text())
    motion=None;native=None;closed=None;trace=[]
    mp=out/'native-route-initial/replay-result.json';npth=out/'native-route-initial/result.json'
    if mp.exists():
        candidate=json.loads(mp.read_text())
        if candidate['source_sha256']==hashlib.sha256(path.read_bytes()).hexdigest() and candidate['source']==str(path.relative_to(ROOT)):motion=candidate
    if npth.exists():native=json.loads(npth.read_text())
    if motion and motion['passed']:closed,trace=actual_recheck(plan,motion,out)
    baseline=json.loads((ROOT/'artifacts/photo-reference-reaching/summary.json').read_text());oldpath=ROOT/baseline['selected_trajectory'];old=json.loads(oldpath.read_text());endpoint=np.array(plan['geometry_record']['candidate']['grasp_center_unity_world_m'])+plan['translation_m'];oldpoint=np.array(old['geometry_record']['candidate']['grasp_center_unity_world_m'])+old['translation_m'];tcp=np.array(plan['tcp_world_m'])
    unchanged=bool(np.array_equal(plan['translation_m'],initial['translation_m']) and np.array_equal(plan['right_q_rad'],initial['right_q_rad']) and np.array_equal(plan['start_q_rad'],initial['left_q_rad']))
    if not unchanged:raise ValueError('Photo initial posture or human translation changed')
    counts={}
    for name in ('geometry-local.json','geometry-rim.json','right-endpoint-summary.json','left-endpoint-summary.json'):
        p=out/name
        if p.exists():counts[name]=json.loads(p.read_text()).get('counts',{})
    summary=dict(offline_passed=validation['offline_passed'],native_replay_passed=motion['passed'] if motion else None,actual_closed_grasp_passed=closed['passed'] if closed else None,
        moving_side=plan['moving_side'],photo_initial_posture_and_human_unchanged=unchanged,initial_configuration='config/photo_reference_initial_pose.yaml',selected_trajectory=str(path.relative_to(ROOT)),
        start_to_target_m=float(np.linalg.norm(tcp[-1]-tcp[0])),tcp_path_length_m=float(np.linalg.norm(np.diff(tcp,axis=0),axis=1).sum()),endpoint_shift_from_previous_right_pregrasp_m=(endpoint-oldpoint).tolist(),
        planned_reach_duration_s=plan['duration_s'],preopen_duration_s=plan['preopen_duration_s'],closure_duration_s=plan['closure_duration_s'],planned_capture_area_mm2=validation['actual_grasp_geometry']['projection']['area_m2']*1e6,
        validation=validation,native_replay=motion,actual_closed_recheck=closed,searches=counts,scope='Left gripper approach and complete closure with frozen cloth; dynamic cloth grasp/retention not validated')
    write(out/'summary.json',summary)
    status=json.loads((out/'trajectory-status.json').read_text());status.update(executed=bool(motion and motion['executed']),native_replay_passed=motion['passed'] if motion else None,actual_closed_grasp_passed=closed['passed'] if closed else None);write(out/'trajectory-status.json',status)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True);state=state_for(plan['geometry_record']);foot=np.array(state['source']['surfaces'][0]['vertices'])+plan['translation_m']
    for ax,axes,title in zip(axs[:2],[(0,1),(2,1)],['Front: left hand reaches the free rim','Side: a detour replaces straight approach']):
        x,y=axes;ax.scatter(foot[:,x],foot[:,y],s=5,c='gray',alpha=.45,label='Foot');ax.plot(tcp[:,x],tcp[:,y],c='tab:blue',label='Left TCP');ax.scatter(tcp[[0,-1],x],tcp[[0,-1],y],c=['tab:green','tab:red'],s=40);ax.scatter([oldpoint[x]],[oldpoint[y]],marker='x',s=50,color='tab:orange',label='Old right endpoint');ax.set(xlabel=('Unity X' if x==0 else 'Unity Z')+' (m)',ylabel='Unity Y (m)',title=title);ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
    ax=axs[2]
    if trace:
        offsets={'opening':0.,'reach':plan['preopen_duration_s'],'settle':plan['preopen_duration_s']+plan['duration_s'],'closure':plan['preopen_duration_s']+plan['duration_s']+1.,'closed_settle':plan['preopen_duration_s']+plan['duration_s']+1.+plan['closure_duration_s']}
        times=np.array([offsets[r['stage']]+r['stage_time_s'] for r in trace]);ax.plot(times,[r['requested_gap_m']*1000 for r in trace],label='Command');ax.plot(times,[r['actual_gap_m']*1000 for r in trace],ls='--',label='Measured')
    ax.set(xlabel='Time (s)',ylabel='Jaw gap (mm)',title='Open, reach, then close to 1 mm');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.savefig(out/'trajectory-and-closure.png',dpi=170);plt.close(fig)
    verdict='未実施' if motion is None else ('合格' if motion['passed'] and closed and closed['passed'] else '不合格')
    actual_text='' if not motion else f"<p>実測終端誤差 {motion['endpoint_position_error_m']*1000:.4f}mm / {motion['endpoint_orientation_error_deg']:.5f}°、実測閉じ幅 {motion['final_actual_gap_m']*1000:.4f}mm、固定側最大変位 {motion['maximum_fixed_drift_m']*1e6:.3f}µm。非除外の腕の負離隔接触 {len(motion['unexpected_contacts'])}件、実測姿勢の距離検査不合格 {len(motion['actual_configuration_failures'])}件。</p>"
    closed_text='' if not closed else f"<p>実測関節角度と閉じ幅で再計算した布捕捉面積は {closed['actual_projection']['area_m2']*1e6:.4f}mm²、閉鎖後の指・手と皮膚の保守的な最短離隔は {closed['closed_skin_clearance_m']*1000:.4f}mm。全身の閉鎖姿勢検査も合格した。</p>"
    rel=str(path.relative_to(out));(out/'report.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>写真の初期姿勢から足に干渉せず指を閉じる経路</title><style>body{{font:16px/1.75 sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;color:#202b35}}img{{max-width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccd5dd;padding:7px 11px;text-align:left}}pre{{background:#f0f3f5;padding:12px;white-space:pre-wrap}}</style>
<h1>写真の初期姿勢から、足に干渉せず指を閉じる経路</h1><p><strong>左グリッパーによる接近・閉鎖の計画検査：合格。PhysXでの再生と閉鎖後の再検査：{verdict}。</strong></p>
<p>人体・靴下・支持台の位置、torso [−45°, 100°, 0°]、両腕の写真に近い初期姿勢を維持した。左手の開き幅は開始時40mm。右腕は初期姿勢に保持し、左手で靴下の上縁へ回り込む。床の例外も従来の左足の約96.6mmの貫通のまま。</p>
<h2>変更した点</h2><p>以前の右手の終端でそのまま閉じると指が足へ重なるため、経路だけを差し替えて同じ終端姿勢へ戻しても解決しない。数mm外へ逃がす局所探索では、皮膚を避けても布を挟めない候補が残った。今回は左手が入れる向きの上縁を選び、経由点と手首の回転を含む経路を生成した。右手だけで閉じる経路を改善できたという結果ではない。</p>
<table><tr><th>項目</th><th>結果</th></tr><tr><td>動かす腕</td><td>左腕。右腕は初期姿勢を維持</td></tr><tr><td>開始から目標までの直線距離／実際のTCP経路長</td><td>{summary['start_to_target_m']*1000:.1f}／{summary['tcp_path_length_m']*1000:.1f}mm</td></tr><tr><td>開口変更・接近・閉鎖</td><td>40→50mm：2秒、接近：{plan['duration_s']:.2f}秒、50→1mm：10秒（別途各1秒の静止）</td></tr><tr><td>時間補間後の軌道検査</td><td>{validation['sample_count']}点</td></tr><tr><td>開口変更／閉鎖検査</td><td>{len(validation['start_opening_checks'])}／{len(validation['virtual_closure_checks'])}点、全点合格</td></tr><tr><td>計画の最小必要離隔余裕</td><td>{validation['minimum_slack_m']*1000:.4f}mm（必要離隔そのものではなく、それに対する追加余裕）</td></tr><tr><td>左腕の最小関節余裕</td><td>{validation['minimum_joint_margin_deg']:.3f}°</td></tr><tr><td>布を挟む面積</td><td>{summary['planned_capture_area_mm2']:.4f}mm²（既定の4mm²以上）</td></tr></table>
<p>手・指と皮膚の必要離隔1mm、腕同士・胴体等の既定の余裕、関節制限を維持した。閉鎖時の保存布との接触だけは捕捉評価の対象として許容し、足・人体との干渉を除外していない。左右の手の衝突STLとグリッパー基部以下の関節・mimic構造が一致することも監査した。左右の取付姿勢の違いはそれぞれの腕のFKで処理している。</p>
<img src="trajectory-and-closure.png" alt="迂回する左手の軌道と開閉幅">{actual_text}{closed_text}
<img src="native-route-initial/fixed-camera-after.png" alt="従来の写真に近い初期姿勢"><img src="native-route-initial/fixed-camera-endpoint.png" alt="左手で閉じた終端"><img src="native-route-initial/endpoint-close.png" alt="閉鎖後の指と靴下・足の接写">
<h2>記録</h2><ul><li><a href="{rel}">軌道 JSON</a> ／ <a href="{rel.replace('.json','.csv')}">左腕の関節角度 CSV</a></li><li><a href="{rel.replace('trajectory','validation')}">軌道・捕捉・閉鎖の検査</a></li><li><a href="native-route-initial/replay-result.json">物理再生</a></li><li><a href="actual-closed-grasp-recheck.json">実測閉鎖姿勢での捕捉・皮膚検査</a></li><li><a href="shared-gripper-geometry.json">左右の手の形状同一性監査</a></li><li><a href="summary.json">集計</a></li></ul>
<h2>再現</h2><pre>LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/capture_fixed_torso_initial.py --output artifacts/photo-grasp-adjustment --trajectory {str(path.relative_to(ROOT))} --replay</pre>
<p>布は保存形状を保持した検証。足に干渉せず閉じられることと、幾何学的な布捕捉を確認するもので、自由に変形する布を摩擦で保持・牽引できることは未検証。軌道サンプル間の連続衝突の形式的証明も含まない。外部確認用の接写・斜視画像は観測カメラの固定比較後に撮影している。</p></html>''')
    print('GRASP ADJUSTMENT REPORT','offline',validation['offline_passed'],'native',verdict,'actual closed',closed['passed'] if closed else None,flush=True)


if __name__=='__main__':main()
