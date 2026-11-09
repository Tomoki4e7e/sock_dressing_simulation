#!/usr/bin/env python3
"""Publish a hash-bound all-70 result only when every independent check passes."""
import argparse,csv,hashlib,html,json,sys,zipfile,shutil
from pathlib import Path
import numpy as np,yaml
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from search_centered_reaching import write,URDF,REGISTERED_MANIFEST
BASE=ROOT/'artifacts/contact-placement-search';SOURCE=BASE/'skin-mesh-far';VERIFY=BASE/'skin-far-verification'
def read(p):return json.loads(p.read_text()) if p.exists() else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--package',action='store_true');a=ap.parse_args();rows=inventory(ROOT);initial=yaml.safe_load((SOURCE/'candidate-initial.yaml').read_text());initial_hash=sha256(SOURCE/'candidate-initial.yaml');checker_hash=sha256(ROOT/'sock_dressing_simulation/continuous_pregrasp.py');urdf_hash=sha256(URDF);manifest_hash=sha256(REGISTERED_MANIFEST);data=[];poses=[]
 for row in rows:
  id=row['id'];path=SOURCE/'routes'/f'{id}.json';route=read(path);route_hash=sha256(path);cc=read(VERIFY/'continuous'/f'{id}.json');supp=read(VERIFY/'continuous-shoulder-bound'/f'{id}.json');ep=read(VERIFY/'endpoint-audit'/f'{id}.json')
  for s in ('left','right'):np.testing.assert_array_equal(route['initial_'+s+'_q_rad'],initial[s+'_q_rad'])
  np.testing.assert_array_equal(route['translation_m'],initial['translation_m']);assert route['condition']==row
  fresh=bool(cc and cc['binding']['route_sha256']==route_hash and cc['binding']['initial_sha256']==initial_hash and cc['binding']['state_sha256']==row['state_sha256'] and cc['binding']['verifier_source_sha256']==checker_hash and cc['binding']['urdf_sha256']==urdf_hash and cc['binding']['geometry_manifest_sha256']==manifest_hash)
  exact=bool(fresh and cc['passed']);positive=exact or bool(fresh and supp and supp['passed'] and supp['route_sha256']==route_hash and supp['geometry_digest']==cc['binding']['geometry_digest'] and supp['verifier_source_sha256']==checker_hash and supp['supplement_source_sha256']==sha256(ROOT/'scripts/verify_pregrasp_shoulder_bound.py'))
  endpoint=bool(ep and ep['passed'] and ep['route_sha256']==route_hash and ep['initial_sha256']==initial_hash and ep['state_sha256']==row['state_sha256'])
  data.append(dict(id=id,stage=row['stage'],exact_margin_certified=exact,positive_clearance_certified=positive,endpoint_passed=endpoint,passed=positive and endpoint,continuous_status=cc['status'] if fresh else 'pending_or_stale',route_sha256=route_hash))
  state=load_state(ROOT,row);foot=np.array(state['source']['surfaces'][0]['vertices'])+initial['translation_m'];record=dict(id=id,stage=row['stage'],pitch_deg=row['pitch_deg'],roll_deg=row['roll_deg'])
  for axis,v in zip('xyz',(foot.min(0)+foot.max(0))/2):record['foot_hull_center_'+axis+'_m']=float(v)
  for side in ('left','right'):
   e=route['terminal'][side]
   for axis,v in zip('xyz',e['tcp_world_m']):record[side+'_tcp_'+axis+'_m']=v
   for axis,v in zip('xyzw',Rotation.from_matrix(e['rotation']).as_quat()):record[side+'_quaternion_'+axis]=float(v)
   record[side+'_opening_gap_m']=e['opening_gap_m'];record[side+'_cloth_finger']=e['contact_check']['cloth_finger']
   for j,q in enumerate(e['q_rad'],1):record[f'{side}_joint_{j}_rad']=q
  poses.append(record)
 complete=len(data)==70 and all(r['passed'] for r in data)
 summary=dict(all_70_verified=complete,counts=dict(expected=70,endpoint_passed=sum(r['endpoint_passed'] for r in data),continuous_exact_margin=sum(r['exact_margin_certified'] for r in data),continuous_positive_clearance=sum(r['positive_clearance_certified'] for r in data),jointly_passed=sum(r['passed'] for r in data)),groups=[dict(stage=s,total=sum(r['stage']==s for r in data),passed=sum(r['stage']==s and r['passed'] for r in data)) for s in 'ABCDE'],rows=data,source_initial_sha256=initial_hash,collision_model='Actual saved human skin, cloth and registered robot collision geometry. Auxiliary human capsules excluded.',native_joint_drive_verified=False,closure_performed=False,local_contact_intrusion_allowance_m=.0005,standing_left_leg_floor_allowance_m=.15,standing_left_foot_reference_penetration_m=initial['grounded_left_foot_penetration_m'],strict_proxy_search=dict(hand_geometry_states=70,individual_arm_endpoint_candidate_states=60,all_70_verified=False),scope='Continuous commanded joint polylines; frozen cloth/human. Supplementary certificates prove positive clearance, not exact original nominal margins. This is not a native-drive, cloth-deformation, closure, holding-force or pulling validation.')
 write(BASE/'result-summary.json',summary)
 accepted={k:v for k,v in initial.items() if k not in ('reachability_verified','verified_paired_state_count','continuous_collision_certified','native_dynamics_verified')};accepted.update(continuous_collision_certified=complete,verified_paired_state_count=summary['counts']['jointly_passed'],native_dynamics_verified=False,proof_input_configuration='skin-mesh-far/candidate-initial.yaml',proof_input_sha256=initial_hash,verification_summary='result-summary.json',collision_model=summary['collision_model']);write(BASE/'verified-initial.json',accepted)
 with(BASE/'grasp-poses-70.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(poses[0]));w.writeheader();w.writerows(poses)
 write(BASE/'grasp-poses-70.json',dict(units='metres, radians; Unity world +Y up; quaternion xyzw maps canonical hand axes to world; foot centre is the saved conservative foot-hull AABB centre',conditions=poses))
 table=''
 for r in data:
  id=r['id'];status='元の離隔で証明' if r['exact_margin_certified'] else '正の離隔下限で証明' if r['positive_clearance_certified'] else r['continuous_status']
  links=f'<a href="skin-mesh-far/routes/{id}.json">姿勢・軌道JSON</a> / <a href="skin-mesh-far/routes/{id}.csv">軌道CSV</a> / <a href="skin-far-verification/continuous/{id}.json">連続時間検査</a> / <a href="skin-far-verification/endpoint-audit/{id}.json">終点再検査</a>'
  if r['positive_clearance_certified'] and not r['exact_margin_certified']:links+=f' / <a href="skin-far-verification/continuous-shoulder-bound/{id}.json">補足証明</a>'
  table+=f'<tr><td>{id}</td><td>{r["endpoint_passed"]}</td><td>{status}</td><td>{links}</td></tr>'

 initial_table=''.join('<tr><td>'+s+'</td><td>'+', '.join(f'{x:.6f}' for x in initial[s+'_tcp_world_m'])+'</td><td>'+f'{initial["tcp_to_foot_aabb_lower_bound_m"][s]*100:.1f} cm'+'</td></tr>' for s in ('left','right'))
 gallery=''.join('<figure><img style="width:100%" src="skin-mesh-far/figures/'+id+'.png"><figcaption>'+id+'</figcaption></figure>' for id in ['A-10','A-37','B-19','C-17','D-01','E-14','E-50'])
 status='全70状態で成立' if complete else '検証中・全70状態の成立は未確認'
 body=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>共通初期姿勢・全70状態の接近軌道</title><style>body{{font:16px/1.7 sans-serif;max-width:1250px;margin:32px auto;padding:0 20px;color:#18232d}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:7px}}th{{background:#eaf0f5}}figure{{margin:24px 0}}</style><h1>共通初期姿勢と全70状態の把持姿勢・接近軌道</h1><p><strong>{status}</strong>。独立終点検査 {summary['counts']['endpoint_passed']}/70、連続時間の正の離隔証明 {summary['counts']['continuous_positive_clearance']}/70。共通の初期位置・回転・関節角を使用します。</p><p><strong>判定範囲：</strong>人体の実皮膚メッシュを基準とした、凍結した人体・靴下に対する予定軌道です。補助カプセルは含みません。関節駆動の全70合格を示す結果ではありません。既存の終点近傍における最大0.5 mmの局所侵入許容と、立脚側左脚の床例外150 mm（参照侵入量約96.6 mm）を継承しています。これらを含む全物体の完全非貫通の証明ではありません。</p><h2>初期位置</h2><p>座標はUnity世界座標、単位m、+Yが上です。距離はTCPから全70状態の足包絡箱までの保守的な下限であり、爪表面からの距離ではありません。</p><table><tr><th>グリッパー</th><th>TCP (X, Y, Z) [m]</th><th>足包絡箱まで</th></tr>{initial_table}</table><p>人体・椅子・靴下の共通平行移動は {initial['translation_m']} m。探索の結果、この実皮膚メッシュ基準の候補では従来の人体配置を採用し、グリッパー初期姿勢を遠方へ変更しました。足全70状態の包絡箱は {initial['initial_foot_bounds_world_m']} mで、全て左右の初期TCPのX座標の間に収まります。足姿勢と靴下形状は保存済みの70条件を維持しています。</p><p><a href="verified-initial.json">検証結果付き初期姿勢（位置・回転・全関節角）</a> / <a href="skin-mesh-far/candidate-initial.yaml">証明に結び付いた固定入力YAML</a> / <a href="grasp-poses-70.csv">70状態の足位置・把持姿勢CSV</a> / <a href="grasp-poses-70.json">同JSON</a></p><p>固定入力YAMLの探索時の未検証フラグは、証明のハッシュを保つため変更していません。最新の検証状態は verified-initial.json と result-summary.json に記録しています。</p><h2>把持条件と動作</h2><p>左右TCPは足の矢状面の反対側に位置します。各手で片爪を開口円周から材料座標15〜30 mmの帯へ、他方を実皮膚面へ接近させ、距離0〜0.5 mm・終点貫通なしを検査します。各手の有限な爪間に実際の局所開口境界線が収まることを検査します。初期側で必要な開口幅を設定し、終点で閉じて把持確認する動作は行いません。</p><p>A-37の途中の布干渉は、関節空間で0.1度の迂回点を追加して修正し、全経路を再検査しました。新しい初期位置から旧開始姿勢までの接続を70本へ追加し、追加後の軌道を各状態で再検査しています。名目離隔を厳密に証明できない一部は、既存と同じ補足検査で最大1 µm小さい正の離隔を証明しています。接触条件を緩和したものではありません。</p><h2>条件別証拠</h2><table><tr><th>条件</th><th>終点条件</th><th>連続時間</th><th>記録</th></tr>{table}</table><p><a href="result-summary.json">ハッシュ付き集計JSON</a> / <a href="teaching-trajectories-70.zip">教示用データ・検証記録ZIP</a></p><h2>代表条件の形状</h2>{gallery}<h2>探索の記録</h2><p>人体・椅子の平行移動45通りの粗探索、補助Colliderを含む高低2群82組の初期非干渉配置、全70状態の補助Colliderを避ける手単体姿勢、選定配置で左右それぞれの腕付き終点候補がある状態60/70（両腕同時・軌道成立の証明ではない）を記録しました。補助カプセル併用の探索は全70状態未成立です。実皮膚メッシュ基準の結果と混同しません。</p></html>'''
 (BASE/'report.html').write_text(body)
 if a.package:
  if not complete:raise ValueError('Refuse to package an incomplete all-70 result')
  (BASE/'PACKAGE-NOTES.txt').write_text('Actual saved skin-mesh collision model; auxiliary capsules excluded. No native-drive or final-closure verification. Existing local contact and standing-left-leg floor exceptions retained. Source shape meshes remain at the repository paths and SHA256 hashes in each route/certificate. Initial configuration hash is bound to skin-mesh-far/candidate-initial.yaml; verified-initial.json provides the updated acceptance metadata for the identical pose.\n')
  programs=BASE/'programs';programs.mkdir(exist_ok=True)
  for path in ['scripts/verify_contact_pregrasp_continuous.py','scripts/verify_pregrasp_shoulder_bound.py','scripts/verify_placed_contact_endpoints.py','scripts/connect_distant_contact_start.py','scripts/repair_contact_continuous_segment.py','scripts/package_contact_placement_result.py','sock_dressing_simulation/continuous_pregrasp.py','sock_dressing_simulation/contact_pregrasp.py']:
   shutil.copy2(ROOT/path,programs/Path(path).name)
  write(BASE/'programs/manifest.json',{p.name:sha256(p) for p in programs.glob('*.py')})
  files=list(programs.glob('*'))+[BASE/'report.html',BASE/'result-summary.json',BASE/'verified-initial.json',BASE/'grasp-poses-70.csv',BASE/'grasp-poses-70.json',BASE/'PACKAGE-NOTES.txt']+list(SOURCE.rglob('*'))+list(VERIFY.rglob('*.json'))
  with zipfile.ZipFile(BASE/'teaching-trajectories-70.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
   for p in files:
    if p.is_file():z.write(p,p.relative_to(BASE))
 print(summary['counts'], 'all_70_verified',complete,flush=True)
if __name__=='__main__':main()
