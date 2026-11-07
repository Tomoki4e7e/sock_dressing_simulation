#!/usr/bin/env python3
"""Audit saved centered-search results with independent tree FK and summarize."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,state_for,URDF,ROBOT,REGISTERED_MANIFEST
from search_translated_reaching import load
from apron_reaching_common import overlay_robot,AuditedScene
from sock_dressing_simulation.arm_reaching import Kinematics,HANDEDNESS
from sock_dressing_simulation.centered_reaching import placement_bounds


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/centered-reaching')
    args=ap.parse_args();out=args.output.resolve();robot=json.loads(ROBOT.read_text())
    full=json.loads((out/'full-grid/geometry-summary.json').read_text())
    geometry=[]
    for ref in full['accepted']:
        geometry.append(json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']])
    datasets={};contexts={};closest=[];maximum_position_difference=0.;maximum_angle_difference=0.;audited=0;central_failures=[];joint_failures=[];tested_keys=set()
    def geometry_key(r):
        return (r['number'],r['source_kind'],r['candidate']['sample_index'],tuple(np.round(np.asarray(r['candidate']['rotation_matrix']).ravel(),7)))
    folders=['multistart','forward','forward-positive','continuous','optimized','unbuffered']+sorted(p.name for p in out.glob('live-*') if p.is_dir())
    for folder in folders:
        paths=sorted((out/folder/'search').glob('target-*.json'))
        counts=Counter();endpoint_pairs=Counter();start_pairs=Counter()
        for path in paths:
            row=json.loads(path.read_text());r=row['geometry_record'];key=(r['number'],r['source_kind']);state=state_for(r)
            tested_keys.add(geometry_key(r))
            if key not in contexts:
                scene=overlay_robot(load(r['number']),robot);scene['particles_world']=state['particles_world'];k=Kinematics(URDF,scene)
                dl,du,contract=placement_bounds(scene,state);contexts[key]=(k,dl,du,scene,state)
            k,dl,du,scene,_=contexts[key];c=r['candidate'];p=np.asarray(c['grasp_center_unity_world_m']);R=np.asarray(c['rotation_matrix'])
            dl,du,_=placement_bounds(scene,state,edge_margin=row['central_contract'].get('edge_margin_m',.01))
            for i,a in enumerate(row['attempts']):
                q=np.asarray(a['q']);d=np.asarray(a['translation_m']);audited+=1;counts['trials']+=1
                T=k.fk(q,only=['right_gripper/gripper_base'])['right_gripper/gripper_base']
                actual=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b
                rotation=k.A@T[:3,:3]@HANDEDNESS
                pe=float(np.linalg.norm(actual-p-d));ae=float(np.rad2deg(Rotation.from_matrix(rotation@R.T).magnitude()))
                maximum_position_difference=max(maximum_position_difference,abs(pe-a.get('position_error_m',pe)));maximum_angle_difference=max(maximum_angle_difference,abs(ae-a.get('orientation_error_deg',ae)))
                central=bool(np.all(d>=dl-1e-9) and np.all(d<=du+1e-9))
                joints=bool(np.all(q>=k.lower+np.deg2rad(5)-1e-8) and np.all(q<=k.upper-np.deg2rad(5)+1e-8))
                if not central:central_failures.append(dict(file=str(path.relative_to(ROOT)),trial=i))
                if not joints:joint_failures.append(dict(file=str(path.relative_to(ROOT)),trial=i))
                ik=pe<=1e-4 and ae<=.5 and joints
                counts['endpoint_ik_passes']+=ik
                if ik:
                    counts['start_clear_ik_passes']+=a.get('initial_pose',{}).get('passed',False)
                    counts['endpoint_clear_ik_passes']+=a.get('endpoint',{}).get('passed',False)
                    counts['start_and_endpoint_passes']+=a.get('initial_pose',{}).get('passed',False) and a.get('endpoint',{}).get('passed',False)
                    if 'endpoint' in a:
                        score=max(-a['endpoint']['minimum_slack_m'],-a['initial_pose']['minimum_slack_m'])
                        closest.append((score,path,i,row))
                    endpoint_pairs.update(tuple(v['pair']) for v in a.get('endpoint',{}).get('issues',[]))
                    start_pairs.update(tuple(v['pair']) for v in a.get('initial_pose',{}).get('issues',[]))
                counts['sampled_route_passes']+=a.get('trajectory',{}).get('passed',False)
        datasets[folder]=dict(targets=len(paths),counts=dict(counts),
            common_endpoint_failures=[dict(pair=list(p),count=n) for p,n in endpoint_pairs.most_common(12)],
            common_start_failures=[dict(pair=list(p),count=n) for p,n in start_pairs.most_common(12)])
    rechecks=[]
    for score,path,trial,row in sorted(closest,key=lambda r:r[0])[:8]:
        r=row['geometry_record'];k,_,_,scene,state=contexts[(r['number'],r['source_kind'])]
        checker=AuditedScene(k,scene,state['triangles'],REGISTERED_MANIFEST,ROOT);a=row['attempts'][trial]
        rechecks.append(dict(source=str(path.relative_to(ROOT)),trial=trial,number=r['number'],
            translation_m=a['translation_m'],q_rad=a['q'],gap_m=a['gap_m'],
            start=checker.slacks(k.start,a['gap_m'],a['translation_m'],details=True),
            endpoint=checker.slacks(np.asarray(a['q']),a['gap_m'],a['translation_m'],details=True)))
        print('REGISTERED RECHECK',path.parent.parent.name,path.name,trial,rechecks[-1]['start']['passed'],rechecks[-1]['endpoint']['passed'],flush=True)
    continuous=json.loads((out/'continuous/summary.json').read_text())
    rotating=json.loads((out/'rotating-approach-diagnostics.json').read_text())
    angles=Counter((r['base_candidate']['yaw_deg'],r['base_candidate']['tilt_deg'],r['additional_roll_deg']) for r in geometry)
    approach=np.array([r['candidate']['rotation_matrix'] for r in geometry])[:,:,0]
    summary=dict(
        scope='Fixed base, torso and left arm at saved apron start; right arm only; entire foot between starting gripper X planes; primary 10mm margin and supplementary zero-margin search',
        geometry_grid=dict(foot_states=full['records'],circumferential_points=32,depths_m=[0,.005,.01,.015,.02,.025,.03],
            yaw_step_deg=30,tilt_deg=list(range(-75,76,15)),independent_roll_step_deg=15,counts=full['counts']),
        surviving_foot_states=sorted({r['number'] for r in geometry}),
        surviving_approach_axis_min=approach.min(0).tolist(),surviving_approach_axis_max=approach.max(0).tolist(),
        surviving_angle_combinations=[dict(yaw_deg=y,tilt_deg=t,roll_deg=r,count=n) for (y,t,r),n in sorted(angles.items())],
        search_datasets=datasets,continuous_orientation_supplement=continuous,rotating_approach_endpoint_diagnostics=rotating['counts'],
        rotating_approach_valid_endpoints=len(rotating['valid_endpoints']),
        independent_audit=dict(trials=audited,maximum_position_error_difference_m=maximum_position_difference,
            maximum_orientation_error_difference_deg=maximum_angle_difference,central_placement_failures=central_failures,joint_margin_failures=joint_failures,
            full_grid_candidates_with_reaching_trials=sum(geometry_key(r) in tested_keys for r in geometry),
            full_grid_candidates_without_reaching_trials=sum(geometry_key(r) not in tested_keys for r in geometry)),
        registered_geometry_closest_candidate_rechecks=rechecks,
        registered_geometry_manifest_sha256=hashlib.sha256(REGISTERED_MANIFEST.read_bytes()).hexdigest(),
        generated_accepted_trajectories=sum(v['counts'].get('sampled_route_passes',0) for v in datasets.values()),
        executed=False,continuous_collision_certified=False,
        limitations=['Finite grids and local optimizations; not an impossibility proof.',
                     'Whole-foot placement hull is conservative; actual skin is used for collision.',
                     'Local hand/cloth geometry uses the previous five convex hand pieces; registered compound shapes are used in the independent closest-case rechecks.'])
    write(out/'summary.json',summary)
    write(out/'independent-audit.json',dict(summary['independent_audit'],registered_rechecks=rechecks))
    if summary['generated_accepted_trajectories']==0:
        write(out/'trajectory-status.json',dict(generated=False,executed=False,
            reason='No centered endpoint passed both start and endpoint collision gates; no accepted trajectory exported',
            full_grid_candidates=len(geometry),full_grid_candidates_with_reaching_trials=summary['independent_audit']['full_grid_candidates_with_reaching_trials'],
            audited_trials=audited,continuous_collision_certified=False))
    # A static explanation of the positional and directional conflict.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    ax=axes[0];links={r['name']:r for r in robot['links']};right=np.array(links['right_gripper/grasping_frame']['position']);left=np.array(links['left_gripper/grasping_frame']['position'])
    ax.axvspan(right[0],left[0],color='tab:green',alpha=.1,label='Starting gripper lateral interval')
    ax.scatter([right[0],left[0]],[right[2],left[2]],c=['tab:blue','tab:orange'],s=60,label='Starting grippers')
    state=json.loads((ROOT/'artifacts/bulged-cuff/pose-10/state.json').read_text());foot=np.asarray(state['source']['surfaces'][0]['vertices']);delta=np.array([-.02683,0,-.4]);moved=foot+delta
    ax.scatter(moved[:,0],moved[:,2],s=4,color='gray',alpha=.4,label='Centered foot example')
    c=json.loads((ROOT/'artifacts/bulged-cuff/pose-10/search/geometry-0001.json').read_text())['candidate'];p=np.asarray(c['grasp_center_unity_world_m'])+delta;R=np.asarray(c['rotation_matrix'])
    ax.arrow(p[0]-.15*R[0,0],p[2]-.15*R[2,0],.15*R[0,0],.15*R[2,0],width=.003,color='tab:red',length_includes_head=True,label='Accepted local approach direction')
    shoulder=np.array(links['right_arm/link_1']['position']);base=p-R@HANDEDNESS@contexts[(10,'bulged')][0].hand.grasp_center_urdf if (10,'bulged') in contexts else p
    ax.plot([shoulder[0],base[0]],[shoulder[2],base[2]],'--',color='tab:purple',label='Shoulder to required gripper base')
    ax.scatter([shoulder[0],base[0]],[shoulder[2],base[2]],color='tab:purple',s=35)
    ax.set(xlabel='Unity X (robot right is negative)',ylabel='Unity Z',title='Centered placement example, delta Z = -0.4 m');ax.set_aspect('equal');ax.legend(fontsize=7,loc='best');ax.grid(alpha=.2)
    ax=axes[1];labels=['Grid poses','Cloth / foot pass','Endpoint + start pass','Connected routes'];values=[full['counts']['tested'],full['counts'].get('accepted',0),sum(v['counts'].get('start_and_endpoint_passes',0) for v in datasets.values()),summary['generated_accepted_trajectories']]
    ax.barh(labels,[max(1,v) for v in values],color=['gray','tab:blue','tab:orange','tab:green']);ax.set_xscale('log');ax.invert_yaxis();ax.set(xlabel='Count (log scale; zero shown at 1)',title='Search funnel')
    for i,v in enumerate(values):ax.text(max(1,v)*1.1,i,f'{v:,}',va='center')
    ax.set_xlim(.8,max(values)*10);fig.savefig(out/'diagnostics.png',dpi=170);plt.close(fig)
    trial_rows=''.join(f'<tr><td>{name}</td><td>{v["targets"]}</td><td>{v["counts"].get("trials",0)}</td><td>{v["counts"].get("endpoint_ik_passes",0)}</td><td>{v["counts"].get("start_and_endpoint_passes",0)}</td></tr>' for name,v in datasets.items() if v['targets'])
    (out/'report.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8">
<title>人体中央配置のリーチング再探索</title>
<style>body{{font:16px/1.7 sans-serif;max-width:1050px;margin:32px auto;padding:0 20px;color:#202b35}}table{{border-collapse:collapse}}td,th{{padding:7px 14px;border:1px solid #cdd5dd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}code,pre{{background:#f0f3f5}}a{{color:#0754a4}}</style>
<h1>人体中央配置のリーチング再探索</h1>
<p><strong>採用軌道：{summary['generated_accepted_trajectories']}件。</strong>今回の保存試行では、開始姿勢と終端姿勢の衝突条件を両方満たす解が得られなかった。終端合格を前提とする軌道生成・物理再生には進んでいない。</p>
<h2>維持した条件と追加した自由度</h2>
<p>エプロン開始姿勢の左右把持フレームのX範囲 −0.343054～＋0.055059m に、記録された足全体の保守的な形状を収める。基本探索は左右各10mmの余裕付きとし、追加探索ではその余裕を0mmにして範囲の境界まで許した。人体・靴下・支持台を一体で平行移動し、床への貫通を許さない。前後±800mm、上方300mmまで。ロボット台車・胴体・左腕は固定し、右腕7関節を探索した。</p>
<p>実関節制限から5°の余裕、足との1mmの離隔、最大開き50mm、閉鎖隙間1mm、把持面積4mm²等の従来条件を維持した。開始第4関節131°の既存例外は保存する。開始側から接続できない既知の肩の断面を越えないよう、第1関節の探索上限は109.5°とした。</p>
<p>各足首状態に224把持点、方位12通り、傾き11通り（−75～75°）、接近軸まわりの独立したひねり24通り（15°刻み）を設定した。32足首状態で<strong>{full['counts']['tested']:,}姿勢</strong>を検査し、布・足の幾何条件を<strong>{len(geometry)}姿勢、{len(summary['surviving_foot_states'])}足首状態</strong>が通過した。指の割り当てや把持点の違いを含む件数で、独立した接近方向数ではない。</p>
<h2>中央配置での到達性</h2>
<table><tr><th>探索記録</th><th>目標数</th><th>試行数</th><th>終端IK成立</th><th>開始・終端とも衝突条件合格</th></tr>{trial_rows}</table>
<p>全{audited:,}試行の位置・姿勢誤差を独立したURDFツリーFKで再計算した。足の中央配置と関節余裕の違反は{len(central_failures)}件・{len(joint_failures)}件。方向グリッドの合格190姿勢は全件、到達性の試行に含めた。</p>
<p>位置・姿勢のIKが成立する配置では、胴体と人体・脚、固定左腕と支持台などの干渉が残った。人体を前へ出す補助探索では、指定の把持姿勢を維持するIKが成立しなかった。手首の向きを連続値で変える{continuous['counts']['proposals']}試行でも、位置に届くことと、足・布の非干渉と把持条件の両立は確認できなかった。</p>
<p>連続姿勢のうち閉鎖・面積条件を通る18試行については、50mmの姿勢固定直進を外し、終端だけを再検査した。13開き幅×18試行の234組すべてが、終端の足との干渉168組または布との干渉66組で除外された。したがって、これらは接近中の手首回転だけで採用できる候補ではなかった。</p>
<p>違反の小さい保存IK解8例を、登録済み3,688部品の衝突形状でも再検査した。全例で開始・終端の不合格を再確認した。</p>
<img src="diagnostics.png" alt="中央配置例と探索段階ごとの候補数">
<h2>保存した結果</h2><ul><li><a href="summary.json">全体集計・条件・不合格理由</a></li><li><a href="independent-audit.json">独立FK監査と登録形状による再検査</a></li><li><a href="full-grid/geometry-summary.json">32足首状態の方向・ひねり探索集計</a></li><li><a href="continuous/summary.json">連続的な手首姿勢の補助探索</a></li><li><a href="trajectory-status.json">軌道生成ゲート</a></li></ul>
<h2>再実行の入口</h2><p>依存パッケージは既存の <code>/tmp/sock-grasp-deps</code> を使用する。条件を変える場合は別の出力先を指定する。</p>
<pre>PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/run_centered_grid_batch.py --output artifacts/centered-reaching/full-grid --workers 18 --theta-step 1 --roll-step 15 --depths 0 .005 .01 .015 .02 .025 .03 --resume
PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/report_centered_reaching.py</pre>
<p>有限の方向グリッドと局所最適化による結果であり、不可能性の証明ではない。局所的な布・手先の選別には従来の5凸形状を使用し、合格に近い例の独立衝突再検査には登録済み複数凸形状を使用した。既存32状態以外の足首姿勢の生成、靴下初期形状の再設計、胴体・左腕の運動は今回の対象に含めていない。</p></html>''')
    print('SUMMARY',full['counts'],'independently audited',audited,'accepted routes',summary['generated_accepted_trajectories'],flush=True)


if __name__=='__main__':main()
