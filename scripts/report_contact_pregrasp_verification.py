#!/usr/bin/env python3
"""Collect continuous certificates and measured native-drive evidence separately."""
import argparse,csv,hashlib,html,json,shutil,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())if path.exists()else None
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-pregrasp-verification');p.add_argument('--package',action='store_true');a=p.parse_args();out=a.output.resolve();source=ROOT/'artifacts/contact-pregrasp-start-search'
    ids=sorted(p.stem for p in(source/'routes').glob('*.json'));rows=[];table=[]
    hashes={k:sha(ROOT/path)for k,path in dict(replay_source_sha256='scripts/replay_contact_pregrasp.py',checker_source_sha256='sock_dressing_simulation/continuous_pregrasp.py',native_geometry_source_sha256='scripts/native_pregrasp_geometry.py').items()}
    initial_sha=sha(source/'candidate-initial.yaml');auditor_sha=sha(ROOT/'scripts/audit_pregrasp_native_contacts.py');events_sha=sha(ROOT/'scripts/audit_pregrasp_native_contact_events.py')
    manifest_sha=sha(ROOT/'artifacts/compound-reaching/registration/candidate/manifest.json');urdf_sha=sha(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf')
    for id in ids:
        route_sha=sha(source/'routes'/f'{id}.json');cc=read(out/'continuous'/f'{id}.json');supp=read(out/'continuous-shoulder-bound'/f'{id}.json')
        state_fresh=bool(cc and cc['binding']['state_sha256']==sha(ROOT/cc['condition']['state']))
        cc_fresh=bool(state_fresh and cc['binding']['route_sha256']==route_sha and cc['binding']['verifier_source_sha256']==hashes['checker_source_sha256'] and cc['binding']['initial_sha256']==initial_sha and cc['binding']['urdf_sha256']==urdf_sha and cc['binding']['geometry_manifest_sha256']==manifest_sha)
        exact=bool(cc_fresh and cc['passed']);positive=exact or bool(supp and supp['passed']and supp['route_sha256']==route_sha and supp.get('verifier_source_sha256')==hashes['checker_source_sha256'])
        folder=out/'drives'/id/'native-route-initial';rp=folder/'replay-result.json';r=read(rp);status=read(folder.parent/'status.json');audit=read(out/'native-endpoint-audit'/f'{id}.json')
        initialized=bool(r and(r.get('capture_initialization_version')=='paired-frozen-reference-v1'or r.get('capture_source_sha256')=='c5944c27c9111c99d01b9a89946a8a35c136b45946d013974a543bdd65665995'))
        fresh=bool(state_fresh and r and initialized and status and status.get('returncode')==0 and all(r.get(k)==v for k,v in hashes.items())and r['source_sha256']==route_sha and r['initial_configuration_sha256']==initial_sha and r['fixed_step_s']==.02)
        audit_fresh=bool(fresh and audit and audit.get('motion_result_sha256')==sha(rp)and audit.get('auditor_source_sha256')==auditor_sha and audit.get('source_route_sha256')==route_sha)
        events=read(out/'native-contact-event-audit'/f'{id}.json');events_fresh=bool(fresh and events and events.get('motion_result_sha256')==sha(rp)and events.get('auditor_source_sha256')==events_sha)
        completed=bool(fresh and r['completed']);endpoint=bool(completed and audit_fresh and audit['contact_passed']and audit['terminal_scene_passed'])
        reasons=[]
        if not fresh:reasons.append('未完了・実行エラー・旧版のいずれか')
        else:
            if not completed:reasons.append('途中停止')
            if r['actual_configuration_failures']:reasons.append('実測姿勢の幾何条件違反')
            if r['unexpected_contacts']:reasons.append('許容外のPhysX接触')
            if not events_fresh:reasons.append('接触全記録の再検査待ち')
            elif not events['passed']:reasons.append('接触全記録に許容範囲外の通知あり（0.1µmしきい値）')
            if completed and not audit_fresh:reasons.append('終点の独立再検査待ち')
            if completed and audit_fresh and not audit['contact_passed']:reasons.append('終点接触・開口線・足の左右条件の不成立')
            if completed and audit_fresh and not audit['terminal_scene_passed']:reasons.append('独立終点検査の干渉条件違反')
            if r['maximum_fixed_drift_m']>=2e-5:reasons.append('固定部位の変位20µm以上（追加の厳格基準）')
            if r['human_skin_maximum_error_m']>=2e-5:reasons.append('人体変位20µm以上')
            if r['cloth_maximum_error_m']>=5e-6:reasons.append('布変位5µm以上')
            for key,label in [('actual_hard_joint_limits_passed','腕関節範囲'),('actual_finger_joint_limits_passed','爪関節範囲')]:
                if not r.get(key):reasons.append(label)
            for key,label in [('maximum_velocity_fraction','腕速度'),('maximum_acceleration_fraction','腕加速度'),('maximum_finger_velocity_fraction','爪速度')]:
                if r.get(key,float('inf'))>1:reasons.append(label)
        passed=bool(endpoint and not reasons)
        contacts=audit['contacts']if audit_fresh else r.get('final_contact_checks',{})if fresh else{}
        row=dict(id=id,continuous_status=cc['status']if cc_fresh else'missing_or_stale',exact_margin_certified=exact,positive_clearance_certified=positive,
                 supplementary_used=positive and not exact,drive_fresh=fresh,motion_completed=completed,online_passed=r['passed']if fresh else None,
                 independent_endpoint_audited=audit_fresh,endpoint_passed=endpoint,drive_passed=passed,failure_reasons=reasons,contacts=contacts,
                 native_event_audited=events_fresh,native_event_passed=bool(events_fresh and events['passed']),
                 maximum_tracking_error_m=r['maximum_tracking_error_m']if fresh else None,maximum_fixed_drift_m=r['maximum_fixed_drift_m']if fresh else None,
                 cloth_maximum_error_m=r['cloth_maximum_error_m']if fresh else None,maximum_velocity_fraction=r.get('maximum_velocity_fraction')if fresh else None,
                 maximum_acceleration_fraction=r.get('maximum_acceleration_fraction')if fresh else None)
        rows.append(row)
        cclabel='既存離隔で証明'if exact else'補足下限で非交差を証明'if positive else'交差検出'if cc_fresh and cc['status']=='failed'else'未証明'
        label='合格'if passed else'不合格'if fresh else'未確定'
        links=f'<a href="continuous/{id}.json">連続時間</a> · <a href="drives/{id}/native-route-initial/replay-result.json">駆動記録</a>'
        if audit_fresh:links+=f' · <a href="native-endpoint-audit/{id}.json">実測終点</a>'
        if events_fresh:links+=f' · <a href="native-contact-event-audit/{id}.json">全接触通知</a>'
        table.append('<tr>'+''.join(f'<td>{x}</td>'for x in[id,cclabel,'完走'if completed else'未完走',label,html.escape('、'.join(reasons))or'—',links])+'</tr>')
        # Full requested/actual arm joints and gaps; source trace remains intact.
        if fresh:
            csvout=out/'measured-trajectories'/f'{id}.csv';csvout.parent.mkdir(exist_ok=True)
            with(folder/'motion-trace.jsonl').open()as src,csvout.open('w')as dst:
                writer=csv.writer(dst);coords=[f'{s}_joint_{i}_rad'for s in('left','right')for i in range(1,8)]+['left_gap_m','right_gap_m']
                writer.writerow(['time_s','stage']+['requested_'+k for k in coords]+['actual_'+k for k in coords]+['left_tcp_'+k+'_m'for k in'xyz']+['right_tcp_'+k+'_m'for k in'xyz']+['tracking_error_m'])
                for line in src:
                    v=json.loads(line);writer.writerow([v['time_s'],v['stage']]+v['requested_x']+v['actual_x']+v['left_tcp_world_m']+v['right_tcp_world_m']+[v['tracking_error_m']])
    counts=dict(expected=len(rows),continuous_exact_margin_certified=sum(r['exact_margin_certified']for r in rows),continuous_positive_clearance_certified=sum(r['positive_clearance_certified']for r in rows),continuous_failed=sum(r['continuous_status']=='failed'for r in rows),continuous_unresolved=sum(r['continuous_status']=='unresolved'for r in rows),drive_runs_current=sum(r['drive_fresh']for r in rows),drive_motion_completed=sum(r['motion_completed']for r in rows),drive_endpoint_audited=sum(r['independent_endpoint_audited']for r in rows),drive_endpoint_passed=sum(r['endpoint_passed']for r in rows),drive_native_events_audited=sum(r['native_event_audited']for r in rows),drive_online_passed=sum(r['online_passed']is True for r in rows),drive_independently_audited_passed=sum(r['drive_passed']for r in rows))
    from collections import Counter
    failure_counts=dict(Counter(reason for r in rows for reason in r['failure_reasons']))
    comparisons=[]
    for id in ('A-01','E-14'):
        path=out/'timestep-10ms/drives'/id/'native-route-initial/replay-result.json';refined=read(path);audit=read(out/'timestep-10ms/native-endpoint-audit'/f'{id}.json')
        if not refined:continue
        audited=bool(audit and audit.get('motion_result_sha256')==sha(path)and audit.get('auditor_source_sha256')==auditor_sha)
        comparisons.append(dict(id=id,fixed_step_s=refined['fixed_step_s'],completed=refined['completed'],online_passed=refined['passed'],independent_endpoint_audited=audited,
            independent_endpoint_passed=bool(audited and audit['contact_passed']and audit['terminal_scene_passed']),maximum_tracking_error_m=refined['maximum_tracking_error_m'],maximum_fixed_drift_m=refined['maximum_fixed_drift_m'],cloth_maximum_error_m=refined['cloth_maximum_error_m']))
    result=dict(counts=counts,failure_counts=failure_counts,timestep_comparisons=comparisons,all_70_verified=all(r['exact_margin_certified']and r['drive_passed']for r in rows),rows=rows,policy=read(out/'policy.json'),source_initial_sha256=initial_sha,
        drive_observation=dict(physics_step_s=.02,geometric_check_every_steps=5,geometric_numerical_allowance_m=1e-7,native_contact_negative_noise_threshold_m=1e-5,cloth_drift_limit_m=5e-6,human_drift_limit_m=2e-5,additional_fixed_body_drift_limit_m=2e-5),
        scope='Commanded continuous paths and finite-step native joint-drive replay are distinct checks. Final independent endpoint meshes take precedence over online endpoint results. No closure or cloth deformation.')
    save(out/'verification-summary.json',result)
    with(out/'verification-summary.csv').open('w')as f:
        keys=['id','continuous_status','exact_margin_certified','positive_clearance_certified','drive_fresh','motion_completed','online_passed','independent_endpoint_audited','endpoint_passed','native_event_audited','native_event_passed','drive_passed','failure_reasons','maximum_tracking_error_m','maximum_fixed_drift_m','cloth_maximum_error_m','maximum_velocity_fraction','maximum_acceleration_fraction'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k]for k in keys}for r in rows)
    cc=counts
    body=f'''<h1>両グリッパー接近軌道の検証</h1>
<p>保存済み70条件を検証。連続時間で既存離隔まで証明：{cc['continuous_exact_margin_certified']}/70。最大1µm小さい離隔での補足証明を含む非交差：{cc['continuous_positive_clearance_certified']}/70。交差検出：{cc['continuous_failed']}条件。</p>
<p>関節駆動シミュレーション：現行版の試行 {cc['drive_runs_current']}/70、完走 {cc['drive_motion_completed']}、独立実測終点の条件成立 {cc['drive_endpoint_passed']}、追加の固定部位変位基準などを含めた合格 {cc['drive_independently_audited_passed']}。オンライン判定の合格数 {cc['drive_online_passed']} は参考値で、最終判定には実測爪メッシュによる独立再検査を使用。</p>
<h2>許容条件</h2><p>各手先が予定終点から5mm以内に入った区間のみ、爪と開口境界から材料座標30mm以内の布、および対象脚の開口境界近傍30mmの皮膚との局所接触を許容。侵入上限0.5mm。掌・腕・遠方の布・皮膚は元の検査を維持。皮膚領域は各三角形全体が一つの境界辺の半径30mmカプセルに含まれる部分に限定し、保守的に評価する。侵入深さは布・皮膚表面が凸爪の内側へ入った最大深さを内側オフセット形状で検査する定義。</p>
<p>終点条件は一方の爪が材料座標15〜30mmの布、他方が足部〜下腿の皮膚に0.5mm以内で接し、左右グリッパーが足の両側にあり、各有限爪間に開口境界線が長さ1µm以上収まること。最終閉鎖は実施していない。</p>
<h2>連続時間検証の解釈</h2><p>関節空間の各区分線分を対象に、URDFの回転軸・リンク長から形状の移動上限を計算し、中点の距離と移動上限で区間全体を判定。必要な区間のみ細分化する。単なる時刻サンプリングではない。計算上の数値余裕は0.1µm。期限・細分化限界で未証明の区間は合格にしない。証明対象は入力軌道、固定した人体・布と現在の登録衝突形状。</p>
<p>補足対象 C-47、D-02、D-16、D-34、E-32、E-44 は肩の離隔を5mmから4.999mmにした下限で検証。E-32では他の正の離隔にも最大1µmの差を認めた。形状カバー用の追加幅は保持。これらは非交差の証明であり、元の厳密な5mm等の条件の合格には含めない。</p>
<p>A-37：左爪と布が、終点から9.515mmの位置で約0.771µm交差した。許容開始の5mmより手前なので不合格。モデル上の微小交差であり、実機の衝突規模を示すものではない。FCLの生値 -1 は交差を示す番兵値で、1mの侵入を意味しない。<a href="crossing-diagnostic.json">交差の独立診断</a></p>
<h2>駆動検証の解釈</h2><p>Unity/PhysXで両腕14関節と爪関節のドライブに指令し、動作中の関節位置の直接設定は使用しない。初期化時に設定した固定基部・胴体用の保持設定は継承。人体姿勢を固定し、布は全粒子の逆質量を0にして変形を凍結。物理刻み20ms、実接触を毎刻み記録、実測リンク姿勢の幾何検査を5刻みごとおよび区間端で実施。予定軌道を速めず、各区間を刻みの整数倍に延長。終点で開いたまま1秒保持する。</p>
<p>布変位5µm以下、人体変位20µm以下、関節範囲・速度・加速度の上限を検査。基部・胴体等の最大変位20µm未満という追加の厳格基準も別に記録。違反時はその試行を途中で停止するため未到達の終点を成功としない。終点は初期の実測爪メッシュ座標と最後の実測個別爪リンク姿勢から再構成し、微小な基準フレーム較正差を除去して再検査する。</p>
<p>オンライン停止には10µmを超える負のNative接触分離量を使用する。保存した接触の全記録をさらに0.1µmのしきい値で再検査し、その結果も最終合否に含める。許容外の通知は native-contact-event-audit/ に保存。接触点の判定と、三角形・内側爪形状を使う幾何検査は別の検査である。</p>
<p>駆動中の実軌道そのものの連続時間証明は行っていない。物理刻みでの検証と予定経路の連続時間証明を合わせても、実機や布変形下の安全性・保持力・引き上げ成功を保証しない。元設定の立脚側左足の床侵入許容量150mm（参照状態約96.6mm）を継承しており、人体と床の完全非貫通を証明した結果ではない。</p>
<p><a href="verification-summary.json">集計JSON</a> · <a href="verification-summary.csv">一覧CSV</a> · <a href="policy.json">判定条件</a> · 実測関節CSVは measured-trajectories/、全時刻記録は drives/条件名/native-route-initial/motion-trace.jsonl に保存。元の探索入力・経路は変更していない。</p>
<h2>条件別結果</h2><table><thead><tr><th>条件</th><th>連続時間</th><th>駆動</th><th>独立判定</th><th>不成立理由</th><th>証拠</th></tr></thead><tbody>{''.join(table)}</tbody></table>'''
    if comparisons:
        body+='<h2>10msでの追加比較</h2><p>代表2条件のみの刻み幅比較であり、全条件の時間刻み収束証明ではない。</p><table><tr><th>条件</th><th>完走</th><th>実測終点条件</th><th>最大手先追従誤差 mm</th><th>固定部位変位 µm</th></tr>'
        for c in comparisons:body+=f'<tr><td>{c["id"]}</td><td>{c["completed"]}</td><td>{c["independent_endpoint_passed"] if c["independent_endpoint_audited"] else "独立再検査待ち"}</td><td>{c["maximum_tracking_error_m"]*1000:.4f}</td><td>{c["maximum_fixed_drift_m"]*1e6:.3f}</td></tr>'
        body+='</table>'
    (out/'report.html').write_text('<!doctype html><html lang="ja"><meta charset="utf-8"><title>接近軌道検証</title><style>body{font-family:system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 24px;line-height:1.7;color:#18232d}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #cbd5df;padding:8px;text-align:left}th{background:#eaf0f5}tr:nth-child(even){background:#f7f9fb}a{color:#1565a5}</style>'+body+'</html>')
    if a.package:
        programs=out/'programs';programs.mkdir(exist_ok=True)
        for pattern in ['scripts/*contact_pregrasp*.py','scripts/*pregrasp*bound.py','scripts/native_pregrasp_geometry.py','scripts/audit_pregrasp_native_contacts.py','scripts/audit_pregrasp_native_contact_events.py','scripts/capture_fixed_torso_initial.py','sock_dressing_simulation/continuous_pregrasp.py','tests/test_continuous_pregrasp.py','tests/test_native_pregrasp_events.py','unity/RuntimeExtensions/SockCompoundReplay.cs','RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockPosePatterns.cs']:
            for src in ROOT.glob(pattern):shutil.copy2(src,programs/src.name)
        runtime_paths=['Build/SockDressingPlayer/RegisteredReachingV3/Player.x86_64','Build/SockDressingPlayer/RegisteredReachingV3/Player_Data/Managed/Assembly-CSharp.dll','Build/SockDressingPlayer/RegisteredReachingV3/Player_Data/Managed/RCareWorld.dll','assets/generated/dry_airec3_gripper_v2_rcareworld.urdf','artifacts/compound-reaching/registration/candidate/manifest.json','config/arm_reaching_safe_start.yaml']
        save(out/'runtime-provenance.json',dict(scope='Post-run fingerprint of the existing runtime and configuration; native mesh snapshots and per-case source bindings provide run evidence.',files={v:dict(sha256=sha(ROOT/v),size=(ROOT/v).stat().st_size,mtime_ns=(ROOT/v).stat().st_mtime_ns)for v in runtime_paths if(ROOT/v).exists()}))
        inputs=out/'source-inputs';inputs.mkdir(exist_ok=True);shutil.copy2(source/'candidate-initial.yaml',inputs/'candidate-initial.yaml')
        (inputs/'routes').mkdir(exist_ok=True)
        for src in(source/'routes').glob('*.json'):shutil.copy2(src,inputs/'routes'/src.name)
        evidence=[p for p in out.rglob('*')if p.is_file()and p.name in('motion-trace.jsonl','native-scene.json.gz','before-scene.json.gz','assets.tar.gz','result.json')]
        save(out/'external-evidence-manifest.json',dict(scope='Large native evidence retained beside this compact archive, not embedded in it.',files={str(p.relative_to(out)):dict(sha256=sha(p),size=p.stat().st_size)for p in evidence}))
        files=[p for p in out.rglob('*')if p.is_file()and p.name!='package-manifest.json'and (p.parent==out and p.suffix in('.json','.csv','.html','.txt')or p.parts[len(out.parts)]in('programs','source-inputs','continuous','continuous-shoulder-bound','native-endpoint-audit','native-contact-event-audit','measured-trajectories')or p.name in('replay-result.json','status.json')or p.parts[len(out.parts)]=='timestep-10ms'and p.suffix=='.json')]
        save(out/'package-manifest.json',{str(p.relative_to(out)):sha(p)for p in files})
        with zipfile.ZipFile(out/'verification-results.zip','w',compression=zipfile.ZIP_DEFLATED)as archive:
            for file in files+[out/'package-manifest.json']:archive.write(file,file.relative_to(out))
    print(json.dumps(counts,ensure_ascii=False))

if __name__=='__main__':main()
