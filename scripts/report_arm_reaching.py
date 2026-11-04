#!/usr/bin/env python3
"""Summarize the two requested right-arm feasibility gates without calling them executions."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-arm-reaching-mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'artifacts/arm-reaching';rows=[]
    lines=['# 代表2候補の右腕リーチング事前検証','',
        '2026-11-04。No.3／候補8、No.12／候補9について、提案済みの初期設定で検証した。',
        '**両候補とも位置・姿勢の同時拘束を満たすIK解が見つからず、目標へのリーチングと閉鎖は実行していない。**',
        'また、既定の開始姿勢には手首周辺の衝突形状の重なりがある。台座・胴体を動かしたり、関節制限を広げたり、目標姿勢を緩和して通過扱いにはしていない。','',
        '## 初期設定','',
        '- 開始姿勢：現在の設定から再現したShareSet ka姿勢。右腕7関節のみを動かす前提。',
        '- 他の腕・胴体・台座は固定する計画。布は各候補の保存済み初期形状を全頂点固定で復元。',
        '- 接近前位置：把持位置から既存の接近軸に沿って50mm手前。目標位置誤差0.1mm以内・姿勢誤差0.5°以内。',
        '- 計画上の開き幅20mm、最終隙間1mm。手と人体の余裕1mm、腕の他部位・他の物体・非隣接自己衝突の余裕5mm。',
        '- 計画上の関節速度はURDF上限の10%、閉鎖速度5mm/s。経路実行に進んでいないため、この速度での実行を検証した結果ではない。','',
        '## 運動学とIK','',
        'Unityから165リンクの座標、全38個のロボット衝突形状、人体全体の実皮膚、足の実皮膚、椅子、床を読み出した。',
        'URDFのFKをUnityのグリッパー基部座標へ校正し、全165リンクの座標・姿勢を照合した。最大位置誤差は約5.23µm、最大姿勢誤差は約0.000084°。',
        '関節制限を守り、固定シードで目標位置・姿勢に96初期値、接近前位置・姿勢に48初期値からIKを解いた。',
        '加えて24初期値から、位置だけを合わせた後、その位置を等式拘束として姿勢誤差を最小化した。','',
        '| 候補 | 目標IK | 50mm手前IK | 位置を一致させた探索解の最小姿勢誤差 | 位置・姿勢を重み付きで近づけた解 |',
        '| --- | --- | --- | ---: | --- |']
    for n in [3,12]:
        directory=out/f'pose-{n:02d}';d=json.loads((directory/'report.json').read_text());scene=json.loads((directory/'initial-scene.json').read_text())
        constrained=[r for r in d['position_constrained_orientation'] if r['position_error_m']<=.0001]
        best=min(constrained,key=lambda r:r['orientation_error_deg']) if constrained else None
        b=d['best_weighted_ik']
        row=dict(number=n,candidate_index=d['candidate_index'],status=d['status'],
            ik_solutions=d['ik_solution_count'],ik_seed_count=d['ik_seed_count'],
            preapproach_solutions=d['preapproach_ik_solution_count'],
            position_matched_best_orientation_error_deg=best['orientation_error_deg'] if best else None,
            best_weighted_position_error_mm=b['position_error_m']*1000,best_weighted_orientation_error_deg=b['orientation_error_deg'],
            native_start_overlap=scene['self_overlap_audit'],reaching_executed=False,closure_executed=False)
        rows.append(row)
        lines.append(f"| No.{n}／{d['candidate_index']} | {d['ik_solution_count']}/96 | {d['preapproach_ik_solution_count']}/48 | {best['orientation_error_deg']:.2f}° | 位置 {b['position_error_m']*1000:.2f}mm・姿勢 {b['orientation_error_deg']:.2f}° |")
    lines+=['',
        '位置だけなら両候補とも一致する解が得られた。問題は位置と姿勢の同時拘束である。',
        '重み付きの近似解ではNo.3の第6関節は−90°、No.12は＋90°、両候補の第7関節は−10°の制限値に達している。',
        '位置だけを一致させた診断姿勢や重み付き近似解は、元の把持候補の姿勢を満たさないため、実行候補へ置き換えていない。',
        '設定ファイルの右手首第7関節の範囲は約−170〜170°だが、URDFとUnityの実制限は−10〜90°で一致する。今回のIKは実制限を使った。設定範囲の広さを理由に、実機構の制限外の解を採用していない。',
        '同じ材料上の把持位置で左右の指を入れ替えた既存合格候補（No.3候補10、No.12候補8）も、各48初期値ではIK解が見つからなかった。これは診断のみで実行していない。',
        '多初期値の局所最適化で解が見つからなかった結果であり、連続空間全体での到達不可能性を数学的に証明したものではない。','',
        '## 開始姿勢の自己干渉','',
        'Unityの実凸コライダーのComputePenetrationでも、次の重なりが検出された。以下は衝突形状の重なり深さであり、実機部品の貫通や接触力を直接示す値ではない。','',
        '| 組み合わせ | 開始時の重なり深さ | Unityの該当ペアの無視設定 |',
        '| --- | ---: | --- |']
    for r in rows[0]['native_start_overlap']:
        lines.append(f"| {r['a']} / {r['b']} | {r['penetration_depth_m']*1000:.2f}mm | {r['unity_pair_ignored']} |")
    lines+=['',
        '機構上の隣接リンク・固定アセンブリ内は検査から除外する一方、上記の手首の非隣接組は自動的に除外していない。',
        '開始姿勢の指関節値0は、現在の実先端断面での爪間隔0を意味しない。爪間隔20mmに相当する指令は約0.0286546mで、爪間の初期重なりを解消する。',
        '初期開き幅を設定して1物理ステップ進めた補助診断では、爪同士の重なりは消えたが手首の2組は残った。',
        '同時に関節の追従ずれとmimic指の中心位置のずれも見られたため、その状態を正しい実行開始姿勢として採用していない。',
        'この補助診断は `opening-one-step-diagnostic-scene.json` に保存し、最終検証は移動指令・物理更新より前で停止する構成に改めた。',
        'mimicの物理追従は今後の開閉実行前に読み戻して検証する必要がある。','',
        '## 実行した範囲と次の手順','',
        '- 最終検証では元の開始姿勢からロボットへの移動指令を出していない。布の形状も固定した。',
        '- 腕全体を含む初期衝突検査と、目標・接近前のIKまで実施した。IKが成立しないため経路探索、最後の50mmの腕動作、低速閉鎖、到達後の射影面積再評価は未実施。',
        '- まず既存60候補へ腕のIK条件を追加し、台座・胴体固定で到達可能な姿勢を含む候補を選び直す。今回の2候補へ経路探索だけを追加しても、目標姿勢の問題は解決しない。',
        '- 開始姿勢の手首の重なりは、衝突メッシュの過大近似・機構上の除外対象・姿勢の問題を確認してから修正する。現状の無視設定をそのまま安全性の証拠にしない。',
        '- 制御設定の第7関節の範囲も実制限と整合させる必要がある。この検証では共通設定ファイルを書き換えていない。',
        '- 到達可能候補と干渉のない開始姿勢が得られてから、衝突回避経路・速度制限付き再生・終端の幾何再評価へ進む。','',
        '## 出力・再現','',
        '- 各 `pose-XX/manifest.json`：要求候補、初期関節値、設定。',
        '- 各 `pose-XX/initial-scene.json`：Unityから読み出した形状・関節情報。',
        '- 各 `pose-XX/report.json`：全IK試行、FK照合、関節制限、初期衝突判定。',
        '- 各 `pose-XX/start-overview.png`：実際の開始姿勢のUnity画像。',
        '- [診断グラフ](diagnostics.png)、[検証結果](validation.json)。','',
        '```bash','PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/verify_arm_reaching.py',
        'python3 scripts/report_arm_reaching.py','```','',
        'Playerは `scripts/run_arm_reaching_player.sh` と `Build/SockDressingPlayer/ArmReachingV1` を使用。',
        'この実装は事前検証までを扱い、汎用の衝突回避経路生成器を実装済みとするものではない。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(completed_feasibility_checks=True,executed_reaching_trials=0,executed_closure_trials=0,poses=rows),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    x=np.arange(2)
    axes[0].bar(x,[r['position_matched_best_orientation_error_deg'] for r in rows],color=['#bb5544','#cc8833'])
    axes[0].axhline(.5,color='green',ls='--',label='Required <= 0.5 deg')
    axes[0].set(xticks=x,xticklabels=['No.3 / 8','No.12 / 9'],ylabel='Orientation error (deg)',title='Best found with target position constrained');axes[0].legend()
    axes[1].bar(x,[r['best_weighted_position_error_mm'] for r in rows],color=['#bb5544','#cc8833'])
    axes[1].axhline(.1,color='green',ls='--',label='Required <= 0.1 mm')
    axes[1].set(xticks=x,xticklabels=['No.3 / 8','No.12 / 9'],ylabel='Position error (mm)',title='Best weighted position + orientation fit');axes[1].legend()
    fig.suptitle('Right arm only; base/torso fixed; bounded multistart IK. No reaching/closure executed.')
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
