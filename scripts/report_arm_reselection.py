#!/usr/bin/env python3
"""Report bounded IK reselection and the repaired initial posture separately."""
import json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'artifacts/arm-reaching-reselection';old=json.loads((out/'existing-summary.json').read_text());start=json.loads((ROOT/'artifacts/arm-reaching-safe-start/validation.json').read_text())
    searches=[]
    for name,label in [('summary.json','32姿勢：足裏法線から60°以内'),('wide-summary.json','32姿勢：足裏寄り側方まで85°以内'),('dense-summary.json','No.3・12：角度と把持点の密探索'),('offset--4mm-summary.json','No.3・12：TCPを接近軸方向に−4mm'),('refined-summary.json','No.3・12：余裕境界付近の局所探索')]:
        rows=json.loads((ROOT/'artifacts/sole-grasp-search'/name).read_text());counts=Counter()
        for r in rows:counts.update(r['counts'])
        searches.append(dict(label=label,source=name,poses=len(rows),counts=dict(counts),tested=counts.get('orientations',counts.get('tested',0))))
    guided=[json.loads((out/f'arm-guided-{n:02d}.json').read_text()) for n in [3,12]]
    assert all(r['counts'].get('geometry_accepted',0)==0 for r in searches), 'New geometry candidates need a fresh review'
    assert all(not r['accepted'] for r in guided), 'New arm-guided candidates need a fresh review'
    result=dict(existing=dict(configurations=60,unique_target_poses=52,ik_seeds_per_target=48,ik_trials=2496,accepted=old['configurations_with_endpoint_ik']),
        safe_start=start,sole_searches=searches,arm_guided_diagnostics=guided,
        new_grasp_candidates=0,reaching_or_closure_executed=False,
        note='Searches overlap; test counts are not counts of unique configurations. No global impossibility proof.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 腕IKによる再選別・開始姿勢修正・足裏側の探索','',
        '**開始姿勢の修正は完了。既存60配置は腕IKで全件不採用。足裏側の追加探索でも、現在の幾何条件を満たす新候補は得られなかった。**',
        '台座・胴体・左腕は変えず、右腕のみを対象にした。爪間隔1mm、射影面積／連結面積4mm²、足との余裕1mmは変更していない。','',
        '## 既存60配置の再選別','',
        '- 60配置に含まれる52個の異なる目標位置・姿勢を、各48初期値（計2,496試行）で評価した。開き幅だけが異なる配置では同じIK結果を再利用した。',
        '- 最初の初期値は修正後の開始姿勢、残り47初期値は固定乱数によるもの。全試行を保存した。',
        '- 位置0.1mm・姿勢0.5°以内の解は0件。したがって現状の60件からリーチング用の候補を選び出すことはできない。',
        '- 全候補でURDFとUnityの実関節制限を使用した。無理な姿勢の除外条件として、候補が成立した場合には端点・50mm手前とも制限から5°以上の余裕を要求する。今回はその前のIK条件で除外された。',
        '- 失敗した最小誤差の姿勢を、元の把持候補の代わりとして実行していない。',
        '- 多初期値探索の失敗であり、数学的な到達不可能性の証明ではない。','',
        '各目標の全試行は `existing/*.json`、集計は [existing-summary.json](existing-summary.json)。','',
        '## 開始姿勢の修正','',
        '修正した設定は [arm_reaching_safe_start.yaml](../../config/arm_reaching_safe_start.yaml)。',
        '右腕第1〜7関節を **[0, −60, 0, 90, 0, −60, 55]°**、実爪間隔を50mmとした。その他の関節・台座・人体・靴下の初期形状は維持した。',
        '同プロファイルの第7関節のソフトウェア制限も、URDF／Unityの−10〜90°へ合わせた。共通のdefault.yamlは書き換えていない。','',
        'No.3・No.12をUnityで新規初期化し、実際の関節位置と全衝突形状を読み戻した。',
        '- 検査対象の自己干渉、人体全体の皮膚、足の皮膚、椅子、床、凍結布との幾何条件を通過。',
        '- 関節制限までの最小余裕：約30°。',
        '- 最も近い検査組は右腕link_2とtorso/link_3で距離約6.254mm。要求5mmに対する余裕は約1.254mm。',
        '- 爪間隔の読出し：約50.000001mm。布の復元誤差は0、FKとNative座標の最大誤差は約7.39µm。',
        '- グリッパー基部と前腕link_5の重なり、および左右の爪の重なりはUnityのComputePenetrationでも解消。','',
        '[開始姿勢の全景](../arm-reaching-safe-start/pose-03/start-overview.png) ／ [手首の拡大](../arm-reaching-safe-start/pose-03/wrist-close.png) ／ [検証JSON](../arm-reaching-safe-start/validation.json)','',
        '### SRDFに基づく判定の訂正','',
        '前回の「非隣接の手首干渉」に含めた `right_arm/link_5` と `right_arm/link_7` は、対応するDry-AIREC3 gripper_v2のSRDFで `reason="Default"` の除外対象だった。前回はこのモデル固有の除外を反映できていなかった。',
        '姿勢を54通り変えてもこの組の重なりが残ること、元メッシュでも重なりがあることを調べたうえで、SRDFを確認して判定を訂正した。',
        'この組の衝突形状の重なり自体を消したとはしていない。機構モデルに従って除外する。',
        '一方、グリッパー基部とlink_5は除外していない。こちらの干渉は姿勢変更によって解消した。',
        'Unityの自己衝突無視設定をそのまま全組へ適用したものではない。元SRDFの場所・ハッシュ・採用した1組を [model-self-collision-policy.json](model-self-collision-policy.json) に記録した。','',
        '## 足裏側からの新しい方向探索','',
        '足裏法線は踵→足首ベクトルのうち足首→つま先軸に直交する成分を使って近似した。世界座標の「下」とは別で、各足首姿勢に追従する。',
        'TCPから50mm手前へ戻る方向を「接近してくる側」として足裏法線との角度を測る。足裏側の布だけに把持点を限定せず、側面・甲側の布へ足裏寄りから接近する方向も対象にした。',
        '周方向・深さ・接近方向・指の割り当てを探索し、4mm²の布が把持面間に残ることと、1mmでの閉鎖位置で足から1mm以上離れることを要求した。','',
        '| 探索 | 延べ方向・配置検査数 | 合格 |','| --- | ---: | ---: |']
    for r in searches:lines.append(f"| {r['label']} | {r['tested']:,} | 0 |")
    lines+=['',
        '各段階のグリッドは重複するため、表の合計を独立した配置数とはしない。',
        '粗探索の把持点は周方向45°刻み、開口から0・10・20mm。密探索は周方向22.5°刻み、深さ0・5・10・15・20・25mmで、旧モデルの接触帯に属する面を使った。',
        '密探索の接近方位は15°刻み、傾きは−75〜75°を15°刻みで探索した。',
        '局所探索は足との余裕境界付近の例から方位・傾き±5°、接近軸回りの指の回転±30°、TCPの接近軸方向のずらし±4mmを試した。TCPをずらしても「実先端から手のひら側10mm」の機械的な基準点は変更していない。','',
        '**主な不採用理由は、射影面積不足、または閉鎖位置での足との1mm余裕不足。**',
        'たとえば足裏60°以内の32姿勢粗探索では、42,488方向中15,642が面積条件で、残る26,846が閉鎖位置の足の条件で除外された。',
        '足と実際に交差する例に加え、交差はしていなくても要求した1mmにわずかに足りない例も含む。',
        '閉鎖端点の条件で除外されたため、これらを腕のIKや接近経路の合格候補として数えていない。余裕や面積の閾値を下げて通過させることはしていない。',
        '結果は試した有限の探索範囲についてのものであり、足裏側からの把持が一般に不可能という意味ではない。','',
        '詳細な各姿勢のレポートは `../sole-grasp-search/pose-XX/` に保存した。','',
        '## 腕の姿勢から逆に作る補助探索','',
        'No.3・No.12では、既存候補の把持位置を保ち、全関節に5°以上の余裕を持たせた腕姿勢から、近いグリッパー姿勢を逆に提案する方法も試した。',
        '元の候補の姿勢を満たすものとして扱わず、新しい候補として同じ布・足・接近条件へ通した。開き幅も1.5〜50mmまで試した。',
        'この48提案でも、位置拘束・足との余裕・面積・接近の全条件を通過する候補は得られなかった。詳細は `arm-guided-03.json`、`arm-guided-12.json`。','',
        '## 現在の状態と次の判断','',
        '- 開始姿勢とモデル固有の自己衝突判定は修正済み。',
        '- 既存60配置からの採用候補は0。新しい足裏寄り候補も今回の範囲では0。',
        '- したがって、リーチングや閉鎖の実行に進んだとはしていない。',
        '- 次の探索では、既存の甲側候補に対して胴体を含めたIKを試す、または足裏側の布に余裕を作った初期状態を用意することが候補になる。今回その条件変更は行っていない。','',
        '## 再現','', '```bash',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/reselect_arm_candidates.py --workers 4',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/verify_arm_reaching.py --snapshot-only --config config/arm_reaching_safe_start.yaml --output artifacts/arm-reaching-safe-start',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_sole_grasps.py --dense',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/refine_sole_grasps.py',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/validate_arm_reselection.py',
        'python3 scripts/report_arm_reselection.py', '```', '',
        '`search_sole_grasps.py --numbers ... --workers 4` で対象の32姿勢を指定して粗探索を行い、`--minimum-sole-alignment 0.0871557427` で85°の側方まで広げられる。',
        'この実装は候補選別・開始姿勢検証を行うもので、汎用経路計画器や実把持保持モデルが完成したという意味ではない。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
