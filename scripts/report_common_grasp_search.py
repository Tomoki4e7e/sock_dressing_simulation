#!/usr/bin/env python3
"""Audit and report the bounded search without publishing an unverified start."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,csv,hashlib,json,sys,zipfile,shutil
from collections import Counter
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,URDF,REGISTERED_MANIFEST,geometric_check
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/common-grasp-start-search');a=ap.parse_args();out=a.output.resolve();inputs=inventory(ROOT);lookup={r['id']:r for r in inputs};criteria_path=ROOT/'config/geometric_closure.yaml';criteria_sha=sha256(criteria_path);settings=yaml.safe_load(criteria_path.read_text());hand=RightHand(URDF);model=type('HandOnly',(),dict(hand=hand))();poses={};sources=[];counts=Counter()
    for folder in [out,out/'transport',out/'transfer-all',out/'symmetry',out/'foot-reflection']:
        for path in sorted((folder/'geometry').glob('*.json')):
            d=json.loads(path.read_text());row=d['condition'];assert row==lookup[row['id']] and d['criteria_sha256']==criteria_sha
            counts['pose_proposals']+=d.get('trials',d.get('counts',{}).get('tested',0));counts['geometry_records']+=1;sources.append(path)
            for i,p in enumerate(d['accepted']):poses.setdefault(row['id'],[]).append(dict(p,source=str(path.relative_to(ROOT)),source_index=i))
    endpoints={};ik_counts=Counter()
    for row in inputs:
        for side in ('left','right'):
            path=out/'arm-space-search'/side/f'{row["id"]}.json'
            if not path.exists():raise ValueError('Arm-space search is unfinished: '+str(path))
            d=json.loads(path.read_text());assert d['condition']==row and d['criteria_sha256']==criteria_sha
            endpoints[row['id'],side]=d
            if not d.get('reused_verified_endpoint'):ik_counts.update(d['counts'])
            for i,e in enumerate(d['accepted']):
                g=e['geometry'];poses.setdefault(row['id'],[]).append(dict(position=g['candidate']['grasp_center_unity_world_m'],rotation=g['candidate']['rotation_matrix'],sample=e['sample'],geometry=g,source=str(path.relative_to(ROOT)),source_index=i))
    # Recheck every distinct accepted pose against the actual saved cloth and
    # sealed skin, with the same 1 mm close and 4 mm2 thresholds.
    verified={};audits=[]
    for identifier,proposals in poses.items():
        row=lookup[identifier];state=load_state(ROOT,row);skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);seen=set();distances={}
        for p in proposals:
            key=tuple(np.round(np.r_[p['position'],np.array(p['rotation']).ravel()],10))
            if key in seen:continue
            seen.add(key);sample=p['sample'];anchor=(sample['face'],tuple(sample['weights']))
            if anchor not in distances:distances[anchor]=material_distances(rest,faces,faces[sample['face']],sample['weights'])
            openings=[g['gap_m'] for g in p['geometry']['valid_openings']]
            geometry,reason=geometric_check(model,state,dict(sample,position=p['position']),np.array(p['rotation']),skin,cloth,settings,distances[anchor],approach_distance=0,opening_gaps=openings)
            if geometry is None:raise ValueError('Accepted pose no longer passes A criteria: '+identifier+' '+reason)
            verified.setdefault(identifier,[]).append(p);audits.append(dict(id=identifier,source=p['source'],source_index=p['source_index'],passed=True,capture_area_mm2=geometry['projection']['area_m2']*1e6,opening_gaps_m=[g['gap_m'] for g in geometry['valid_openings']]))
            if len(hand._banks)>256:hand._banks={k:v for k,v in hand._banks.items() if k[0]==.001}
    write(out/'accepted-pose-audit.json',dict(passed=True,distinct_poses=len(audits),criteria_sha256=criteria_sha,rows=audits))
    starts=json.loads((out/'start-candidates/start-search.json').read_text());assert starts['shared_state_count']==70 and all(p['check']['passed'] for p in starts['candidate_pairs']);initial_path=out/'start-candidates/initial.yaml';initial_sha=sha256(initial_path)
    for d in endpoints.values():assert d['initial_sha256']==initial_sha
    optimizations=[json.loads(p.read_text()) for p in sorted((out/'optimization').glob('*.json'))];assert len(optimizations)==10
    for d in optimizations:assert d['criteria_sha256']==criteria_sha and d['condition']==lookup[d['condition']['id']]
    rows=[]
    for row in inputs:
        left=bool(endpoints[row['id'],'left']['accepted']);right=bool(endpoints[row['id'],'right']['accepted']);reasons=[]
        if row['id'] not in verified:reasons.append('把持条件を満たすグリッパー終点を未取得')
        if not left:reasons.append('左腕の合格IK終点を未取得')
        if not right:reasons.append('右腕の合格IK終点を未取得')
        rows.append(dict(**row,tool_grasp_found=row['id'] in verified,distinct_tool_poses=len(verified.get(row['id'],[])),left_endpoint_found=left,right_endpoint_found=right,approach_route_verified=False,native_verified=False,reasons='; '.join(reasons)))
    groups=[dict(stage=stage,total=sum(r['stage']==stage for r in rows),tool_grasp_found=sum(r['stage']==stage and r['tool_grasp_found'] for r in rows),left_endpoint_found=sum(r['stage']==stage and r['left_endpoint_found'] for r in rows),right_endpoint_found=sum(r['stage']==stage and r['right_endpoint_found'] for r in rows)) for stage in 'ABCDE']
    old_protocol=json.loads((ROOT/'artifacts/common-opening-dynamics/protocol.json').read_text());published=ROOT/'config/common_opening_initial_pose.yaml';old_sha=next(r['sha256'] for r in old_protocol['files'] if r['path']=='config/common_opening_initial_pose.yaml');assert sha256(published)==old_sha
    summary=dict(search_finished=True,goal_met=False,verified_universal_initial_configuration=None,initial_configuration_adopted=False,published_initial_unchanged=True,initial_candidate_pairs=len(starts['candidate_pairs']),candidate_pair_source=str(initial_path.relative_to(ROOT)),candidate_initial_sha256=initial_sha,conditions=70,tool_grasp_states=sum(r['tool_grasp_found'] for r in rows),left_endpoint_states=sum(r['left_endpoint_found'] for r in rows),right_endpoint_states=sum(r['right_endpoint_found'] for r in rows),groups=groups,rows=rows,missing_tool_grasp_states=[r['id'] for r in rows if not r['tool_grasp_found']],geometry_proposals=counts['pose_proposals'],optimization_evaluations=sum(d['counts']['evaluations'] for d in optimizations),optimization_accepted=sum(len(d['accepted']) for d in optimizations),arm_space_counts=dict(ik_counts),criteria=settings,criteria_sha256=criteria_sha,geometry_manifest_sha256=sha256(REGISTERED_MANIFEST),workbook_sha256=sha256(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx'),scope='Bounded finite search with frozen unchanged shapes. Missing candidates are not a proof of nonexistence. Initial-pair union geometry and endpoint IK do not certify approach paths, native motion, friction or retention.')
    write(out/'summary.json',summary)
    fields=['id','stage','number','pitch_deg','roll_deg','tool_grasp_found','distinct_tool_poses','left_endpoint_found','right_endpoint_found','approach_route_verified','native_verified','reasons']
    with (out/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    write(out/'candidate-initial-status.json',dict(initial_source=str(initial_path.relative_to(ROOT)),initial_sha256=initial_sha,initial_union_geometry_passed=True,all_70_grasp_routes_verified=False,adopted=False,reason='Mandatory grasp endpoints are missing; no verified universal initial pose is published'))
    table=''.join(f"<tr><td>{g['stage']}</td><td>{g['total']}</td><td>{g['tool_grasp_found']}</td><td>{g['left_endpoint_found']}</td><td>{g['right_endpoint_found']}</td></tr>" for g in groups)
    case_table=''.join(f"<tr><td>{r['id']}</td><td>{r['tool_grasp_found']}</td><td>{r['left_endpoint_found']}</td><td>{r['right_endpoint_found']}</td><td>{r['reasons']}</td></tr>" for r in rows)
    (out/'report.html').write_text(f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>把持終点を条件にした共通初期姿勢の再探索</title><style>body{{font:16px/1.7 sans-serif;max-width:1250px;margin:30px auto;padding:0 20px;color:#223}}table{{border-collapse:collapse;margin:20px 0}}td,th{{border:1px solid #bbb;padding:6px 12px}}img{{max-width:100%}}</style><h1>全70状態で把持へ接近する共通初期姿勢の再探索</h1><p><strong>70状態すべてを満たす共通初期姿勢は未取得です。初期干渉のない候補24組を得ましたが、把持終点と接近軌道の条件を満たす解としては採用していません。</strong></p><p>初期位置に依存しないグリッパー単体の把持終点は{summary['tool_grasp_states']}/70状態で取得。候補初期姿勢1組を固定した腕のIK・全身閉鎖検査では、左腕{summary['left_endpoint_states']}状態、右腕{summary['right_endpoint_states']}状態に合格終点を取得しました。接近経路と動力学再生は未認証です。</p><p>終点を得られない状態は、足・布との干渉、捕捉面積、腕のIKの段階で候補が落ちています。初期姿勢だけを変更しても、同じ終点の幾何干渉は変わらないため、全条件を満たす設定の発行には進んでいません。有限範囲・時間内の探索結果であり、不可能性の証明ではありません。</p><h2>段階別の結果</h2><p>IK欄の0は今回の探索で未取得という意味です。グリッパー単体の合格は腕の到達可能性を意味しません。</p><table><tr><th>開口</th><th>状態数</th><th>単体の把持終点</th><th>左腕IK終点</th><th>右腕IK終点</th></tr>{table}</table><h2>維持した条件</h2><p>保存された靴下70形状、足首姿勢、人体・支持台の配置、ロボットの基部・胴体、グリッパーの形状を維持。閉じ幅1mm、皮膚離隔1mm、捕捉面積と最大連結面積4mm²以上、面積比5%以上、捕捉幅0.5mm以上、固定8×8mm窓、30mm材料帯を従来Aと同じにしています。Aの既知の合格例で判定・面積・幅・皮膚離隔の一致を確認しました。</p><h2>今回の探索</h2><p>布面の周方向64点×深さ13段階と姿勢・位置の変化、成功したA・Cの姿勢の材料座標での移送、足の反対側への反射、指の入れ替え、接近方向反転、開き幅の連続探索を実施。形状探索の提案{summary['geometry_proposals']:,}件、B全10状態の連続最適化{summary['optimization_evaluations']:,}評価。最適化の代理目的値だけで合格にせず、合格候補{len(audits)}姿勢を保存形状に対して独立に再検査しました。</p><p>両腕の初期姿勢は各48候補、全70状態の布と全32足首姿勢の人体形状を重ねた幾何検査で24組が合格。目標付近の実際のFK姿勢を新たな終点として判定する腕空間の探索も全70状態・左右腕で実施しました。提案の位置・向きを変えても、把持の合否条件は変更していません。</p><h2>代表候補の開き幅</h2><p>A-No.10の合格例と、Bの有望だった閉鎖姿勢を比較しています。Bでは皮膚を避ける開き幅が約1.20〜1.43mm以下に限られ、その範囲で指と布の干渉を解消できませんでした。下図はこれらの固定姿勢に対する幅の走査で、他の全姿勢の非存在を示すものではありません。</p><img src="opening-tradeoff.png" alt="Aの合格姿勢とBの有望姿勢での皮膚離隔・布干渉の開き幅依存性"><p><a href="summary.json">全結果JSON</a> ／ <a href="cases.csv">70状態の一覧CSV</a> ／ <a href="accepted-pose-audit.json">合格単体姿勢の再検査</a> ／ <a href="opening-tradeoff.json">開き幅の診断値</a> ／ <a href="controls/A10-default-equivalence.json">旧Aの陽性対照</a> ／ <a href="tests.txt">テスト結果</a></p><p><a href="start-candidates/start-search.json">未採用の初期姿勢24組</a> ／ <a href="candidate-initial-status.json">採用状態</a>。既存の公開初期姿勢設定は更新していません。布の変形・摩擦保持・牽引の検証は含みません。</p><h2>状態ごとの結果</h2><table><tr><th>状態</th><th>単体把持終点</th><th>左腕IK終点</th><th>右腕IK終点</th><th>未成立の段階</th></tr>{case_table}</table></html>''')
    program_names=['scripts/search_common_grasp_geometry.py','scripts/optimize_common_grasp_geometry.py','scripts/check_common_grasp_arm_endpoints.py','scripts/expand_common_grasp_symmetries.py','scripts/diagnose_grasp_opening_tradeoff.py','scripts/report_common_grasp_search.py','scripts/search_centered_reaching.py','tests/test_continuous_grasp_opening.py']
    for name in program_names:
        target=out/'programs'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    write(out/'programs/sources.json',dict(files=[dict(source=name,sha256=sha256(ROOT/name)) for name in program_names],note='Source snapshots; execute in the original project with its referenced model assets.'))
    # Package the reviewable result, not a claimed valid new initial setting.
    members=sorted(p for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.html','.png','.yaml','.txt','.py') and p.name not in ('archive.json','manifest.json'))
    manifest=dict(goal_met=False,search_finished=True,files=[dict(path=str(p.relative_to(out)),sha256=sha256(p)) for p in members],inputs=[dict(id=r['id'],state=r['state'],sha256=r['state_sha256']) for r in inputs],criteria_sha256=criteria_sha,urdf_sha256=sha256(URDF),published_initial_sha256=old_sha)
    write(out/'manifest.json',manifest);members.append(out/'manifest.json');archive=out/'search-records.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in members:z.write(p,str(p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    write(out/'archive.json',dict(path=archive.name,bytes=archive.stat().st_size,sha256=sha256(archive),members=len(members)))
    print('SEARCH REPORT',summary['tool_grasp_states'],'tool states',summary['left_endpoint_states'],'left',summary['right_endpoint_states'],'right; goal met',False,flush=True)

if __name__=='__main__':main()
