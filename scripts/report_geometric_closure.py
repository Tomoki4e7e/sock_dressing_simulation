#!/usr/bin/env python3
"""Publish force-free closure results and projected-area inspection figures."""
import json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-geometric-matplotlib')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'artifacts/geometric-closure';summary=json.loads((out/'summary.json').read_text());rows=[]
 for p in summary['poses']:rows.extend(json.loads((out/f"pose-{p['number']:02d}/report.json").read_text())['candidates'])
 good=[r for r in rows if r['accepted']];settings=summary['settings']
 lines=['# 摩擦保持を使わない幾何学的閉鎖の再評価','',f"旧候補 {len(rows)} 配置をすべて再評価。合格 {len(good)}、不合格 {len(rows)-len(good)}。候補が残る足首姿勢は {summary['poses_with_candidates']}/32。",'',
 '## 現在の判定と閉鎖構成','',
 '1. 開いた手の50mm直線接近全体で、布との交差なし・実皮膚から1mm以上の余裕を要求する。',
 '2. 内側・外側の指の配置、内側の指が皮膚へ向いていること、全手形状の足との余裕により挿入空間を判定する。',
 '3. 布を初期形状のまま固定し、正の評価隙間1mmまで仮想閉鎖する。途中を含め足との余裕1mmと、左右の指同士の非交差を要求する。',
 '4. 実際の凸指形状の対向面で区切った有限領域へ布を3Dで切り取り、閉鎖方向へ正射影した面積の和集合を求める。',
 '5. 8×8mm窓内で面積4mm²以上・面積率5%以上・連結面積4mm²以上・両方向の広がり0.5mm以上を要求する。',
 '6. 全条件が通った配置だけに開→閉の実行プランを有効化する。不合格の配置は実行しない。摩擦力、接触ソルバー値、保持時間、滑りは合否に使わない。','',
 '評価隙間1mmは実厚みではなく、厚みゼロのメッシュの評価用の設定。1mm・4mm²は5候補の画像レビュー後に本幾何タスクの測定定義として固定したが、物理的な把持成功の閾値として較正したものではない。対象の布は元の把持点から材料座標の辺経路距離20mm以内、開口から30mm帯以内へ限定する。','',
 '三角形の重複・両面・射影重なりを二重計上せず、穴を保存する。連結判定のみ1nmの数値的継ぎ目を同一視し、測定面積を膨張させない。','',
 '## 姿勢別結果','', '| No. | pitch ° | roll ° | 入力 | 合格 |','| --- | ---: | ---: | ---: | ---: |']
 for p in summary['poses']:lines.append(f"| {p['number']} | {p['pitch_deg']:g} | {p['roll_deg']:g} | {p['input_count']} | {p['accepted_count']} |")
 lines+=['','## 条件ごとの不合格件数（重複あり）','', '| 条件 | 件数 |','| --- | ---: |']
 for key,count in summary['failed_checks_nonexclusive'].items():lines.append(f'| {key} | {count} |')
 lines+=['','## 出力','', '- [全436配置のCSV](candidates.csv)','- [足首姿勢別CSV](poses.csv)','- [設定・集計JSON](summary.json)','- [独立検証](validation.json)','- [射影領域と結果の図](diagnostics.png)',
 '- `pose-XX/report.json`：各配置の射影ポリゴン、指の内外関係、接近・閉鎖の皮膚余裕、合否、開閉の関節値と5メッシュの姿勢。','',
 '## 仮想閉鎖の範囲','',
 f"閉鎖位置では全{summary['closed_cloth_overlap_count']}配置で凍結布と爪の交差も検出される。これは閉鎖時の棄却条件にしていない。",
 '閉鎖は布を変形させない幾何評価であり、爪と重なる布も含む初期メッシュのうち、爪の間に残る領域を測る。閉鎖後の布全体が爪を貫通しない物理状態や、実把持を再現したという意味ではない。',
 '接近中の布交差は許容しない。足との交差は接近・閉鎖を通して許容しない。',
 'たるみは指を挿入できる空間によって代理評価しており、布の張力や伸びから余長を推定する処理ではない。','',
 '## 再現','', '```bash', 'PYTHONPATH=/tmp/sock-grasp-deps python3 scripts/reevaluate_geometric_closure.py --workers 4', 'python3 scripts/validate_geometric_closure.py', 'python3 scripts/report_geometric_closure.py', '```', '',
 '[設定ファイル](../../config/geometric_closure.yaml)で閉鎖隙間と閾値を変更できる。以前の摩擦保持計測は履歴として別フォルダに保持する。']
 lines+=['','## 合格候補の実画像レビュー','',
 '[5候補・3段階・4視点の撮影と判定値のレビュー](../geometric-closure-photos-clean/README.md)。',
 '検査表示では評価メッシュにない太い開口縁の表示線を隠した。対象布・皮膚・指の配置をUnityの読出しで照合している。',
 '次の動作確認はNo.3候補8・No.12候補9を優先する。候補集合は60配置のまま。']
 (out/'README.md').write_text('\n'.join(lines)+'\n')
 fig=plt.figure(figsize=(15,9),layout='constrained');grid=fig.add_gridspec(2,3)
 ax=fig.add_subplot(grid[0,:2]);positions=np.arange(len(summary['poses']));ax.bar(positions,[p['input_count'] for p in summary['poses']],color='.8',label='Input configurations');ax.bar(positions,[p['accepted_count'] for p in summary['poses']],color='#228833',label='Geometry accepted')
 ax.set_xticks(positions,[str(p['number']) for p in summary['poses']],fontsize=8);ax.set(xlabel='Workbook row number (32 ankle states)',ylabel='Configurations',title='Force-free geometric closure, fixed positive terminal gap');ax.legend()
 ax=fig.add_subplot(grid[0,2]);ax.scatter([r['opening_gap_m']*1000 for r in rows],[r['projection']['area_m2']*1e6 for r in rows],c=['#228833' if r['accepted'] else '#999999' for r in rows],s=12,alpha=.7);ax.axhline(settings['minimum_area_m2']*1e6,color='red',ls='--');ax.set(xlabel='Initial opening (mm)',ylabel='Projected cloth union area (mm²)',title=f"{len(good)} / {len(rows)} accepted at 1 mm")
 examples=[max(good,key=lambda r:r['projection']['area_m2']),min(good,key=lambda r:r['projection']['area_m2']),next(r for r in rows if r['number']==1 and r['candidate']['sample_index']==9 and r['candidate']['yaw_deg']==330 and r['candidate']['closing_sign']==1 and abs(r['opening_gap_m']-.03)<1e-9)]
 for i,r in enumerate(examples):
  ax=fig.add_subplot(grid[1,i]);q=r['projection'];h=settings['half_extent_m']*1000;ax.add_patch(Rectangle((-h,-h),2*h,2*h,fill=False,ls='--',edgecolor='.4'))
  for polygon in q['polygons_xz_m']:
   v=np.asarray(polygon['exterior'])*1000;ax.fill(v[:,0],v[:,1],color='#228833' if r['accepted'] else '#dd8800',alpha=.8)
   for hole in polygon['holes']:
    v=np.asarray(hole)*1000;ax.fill(v[:,0],v[:,1],color='white')
  ax.plot(0,0,'k+',ms=7);ax.set(xlim=(-h-.5,h+.5),ylim=(-h-.5,h+.5),aspect='equal',xlabel='Approach coordinate x (mm)',ylabel='Jaw-width coordinate z (mm)',title=f"Pose {r['number']}, candidate {r['candidate_index']}: {'accepted' if r['accepted'] else 'rejected'}\nUnion {q['area_m2']*1e6:.3f} mm²; gap 1 mm");ax.grid(alpha=.2)
 fig.suptitle('Actual facing-facet clipping + projected polygon union; frozen cloth; no friction or retention criterion')
 fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()
