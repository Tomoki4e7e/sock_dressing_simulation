#!/usr/bin/env python3
"""Summarize the +5 mm initial-shape experiment with its exact scope."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-bulged-cuff-mpl')
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'artifacts/bulged-cuff';search=json.loads((out/'search-summary.json').read_text());validation=json.loads((out/'validation.json').read_text());native=json.loads((out/'native/validation.json').read_text());margin=json.loads((out/'margin-optimization.json').read_text());closest=json.loads((out/'closest-ik-diagnostics.json').read_text());pre=json.loads((out/'closest-preapproach-ik.json').read_text());counts=search['counts'];old=[];geometric=[]
    for pose in search['poses']:
        folder=out/f"pose-{pose['number']:02d}";r=json.loads((folder/'search/report.json').read_text());old.extend(r['old60_parameter_recheck'])
        for path in (folder/'search').glob('geometry-*.json'):geometric.append(json.loads(path.read_text()))
    assert counts.get('arm_accepted_configurations',0)==0 and pre['solution_count']==0
    old_counts=Counter(r['stage'] for r in old);best=margin[0]['best'];edge_min=min(r['peak_mesh_edge_radius_increment_mm_min'] for r in validation['poses']);edge_max=max(r['peak_mesh_edge_radius_increment_mm_max'] for r in validation['poses']);native_error=max(r['cloth_error_m'] for r in native['poses'])
    summary=dict(shape=dict(material_band_width_mm=30,center_depth_mm=15,peak_vertex_radius_increment_mm=5,section_edge_radius_increment_mm_range=[edge_min,edge_max],outside_band_unchanged=True,vertices=1057,faces=2080),
        ankle_states=32,geometry_checks_passed=32,native_checks_passed=len(native['poses']),counts=counts,
        ankle_states_with_geometry_candidates=len([p for p in search['poses'] if p['counts'].get('geometry_accepted_poses',0)]),
        target_poses_with_endpoint_ik=sum(r['endpoint_ik_solution_count']>0 for r in geometric),final_accepted_configurations=0,
        old60_parameter_recheck=dict(old_counts),torso_ik_enabled=False,executed_reaching=False,executed_closure=False,
        closest_target=dict(number=39,geometry_index=3,best_margin_deg=best['minimum_joint_margin_deg'],preapproach_ik_solutions=pre['solution_count'],whole_arm_open_clear_trials=sum(r['open_collision']['passed'] for r in closest['results'])))
    (out/'summary-results.json').write_text(json.dumps(summary,indent=2)+'\n')
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';canvas=Image.new('RGB',(1400,1180),'white');draw=ImageDraw.Draw(canvas)
    for row,n in enumerate([3,12]):
        for col,(name,title) in enumerate([('before-close.png','変更前'),('bulge-close.png','変更後：15mm中心、半径+5mm')]):
            image=Image.open(out/f'pose-{n:02d}'/name).convert('RGB').resize((700,550));top=row*590;draw.text((col*700+12,top+4),f'No.{n} ／ {title}',font=ImageFont.truetype(font,24),fill='black');canvas.paste(image,(col*700,top+40))
    canvas.save(out/'before-after.jpg',quality=95)
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    d=np.linspace(0,30,301);axes[0].plot(d,5*np.sin(np.pi*d/30)**2,label='Prescribed vertex displacement field')
    profile=json.loads((out/'pose-03/profile.json').read_text());z=np.array(profile['depth_m'])*1000;delta=np.array(profile['radius_increment_m'])*1000;active=z<=30+1e-6;axes[0].scatter(z[active],delta[active],s=16,c='orange',label='Mesh vertices');axes[0].axvline(15,color='.5',ls=':');axes[0].set(xlabel='Material depth from opening (mm)',ylabel='Radial increment (mm)',title='Local radial bulge: peak +5 mm at 15 mm',xlim=(-1,32));axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].bar(['Geometric\nconfigurations','Target poses\nwith bounded IK','Final\naccepted'],[counts['geometry_accepted_configurations'],1,0],color=['#4477aa','#ddaa33','#aa4455']);axes[1].set(ylabel='Count (different stages have different units)',title='Torso/base fixed; right arm only');axes[1].text(1,3,f"Closest joint margin\n{best['minimum_joint_margin_deg']:.2f}° < required 5°",ha='center',fontsize=9)
    fig.savefig(out/'diagnostics.png',dpi=150);plt.close(fig)
    lines=['# 開口から15mmを中心に半径を5mm増やした初期形状での再探索','',
        '**32足首姿勢の布形状を変更し、実際のObiメッシュとの一致を確認して再探索した。胴体・台座・左腕は固定したまま。最終採用候補は0件。**','',
        '## 形状変更','',
        '[変更前後のUnity画像](before-after.jpg) ／ [形状プロファイルと結果の図](diagnostics.png)','',
        '- 従来と同じ材料座標で、開口を0mm、探索帯の端を30mmとした。中心15mmで元の断面中心線からの半径を5mm増やす。',
        '- 半径増分は `5 × sin²(π d / 30)` mm（0〜30mm）。0mmと30mmで0、15mmで5mm。30mmよりつま先側の元の形状は変えていない。',
        '- 既存の断面中心線に直交する半径方向へ移動する。元の靴下は非円形なので、円へ置き換えたり全体を拡大したりしていない。',
        '- 元の長さ方向12.5mm刻みには15mm断面がないため、7.5・15・22.5・30mmで元の三角形を切り分けた。変形前の面を保ったまま801頂点・1568面から1057頂点・2080面へ細分化した。',
        '- 15mm断面の64頂点では半径増分5.000mm。面内補間を含む輪郭では、有限の直線エッジによる近似で約'+f'{edge_min:.3f}〜{edge_max:.3f}mm。',
        '- 32姿勢すべてで、足の実皮膚との交差・布の自己交差がないことを確認。帯の外側の変形前後座標は完全一致する。',
        '- 固定初期形状として再検証した。布の物理静定や摩擦保持を新たに検証した結果ではない。','',
        '変更量は [config/cuff_bulge.yaml](../../config/cuff_bulge.yaml)。各 `pose-XX/state.json` に新形状、`before-refined.json` に同じ細分化を施した変形前形状、`profile.json` に元の中心線・半径方向・変形量を保存した。','',
        '## シミュレータとの一致','',
        '細分化したメッシュを実際のObiClothとして読み込み、Nativeの頂点順へ対応付けて復元した。表示だけ別の布へ差し替えたものではない。',
        f'32姿勢すべてで三角形形状が一致し、布のNative読出し誤差は最大{native_error*1e6:.3f}µm。足の皮膚の読出し誤差は0。',
        'OBJインポータのX反転と面の向きの反転を補正している。`rest.obj` はこのインポータ用、`deformed-world.obj` は通常の世界座標のメッシュ。',
        '修正済みの開始姿勢も、新しい布と全32足首姿勢で衝突条件を再確認した。右腕以外のIKは有効にしていない。','',
        '## 維持した判定','',
        '- Dry-AIRECの指・手のひらの実衝突形状、最大開き幅50mm、TCPは実先端から手のひら側10mm。',
        '- 開いた状態の最後の50mm接近で布と非交差。接近・閉鎖中とも足から1mm以上の余裕。',
        '- 仮想閉鎖隙間1mm。8×8mm窓内の射影面積と連結面積は各4mm²以上、面積率5%以上、両方向の広がり0.5mm以上。',
        '- 右腕7関節のみ。位置誤差0.1mm以内、姿勢誤差0.5°以内。実関節制限内で、端点・接近前位置とも制限から5°以上の余裕を要求。',
        '- 自己衝突のモデル固有のSRDF除外は前回修正した同じものを使用。人体全体・椅子・床・他のロボット部位の検査を含む。',
        '- 閉鎖中の布は凍結した幾何評価で、布と爪の重なりを許容する。保持・摩擦は合否に使用しない。','',
        '## 探索結果','',
        '各姿勢について周方向32点、深さ0・5・10・15・20・25・30mm、接近方位30°刻み、傾き−75〜75°の15°刻み、左右の指の割り当て2通りを探索した。',
        '開き幅は1.5・2・3・4・5・7.5・10・15・20・25・30・40・50mmで、50mmの上限は維持した。',
        '足との閉鎖端点条件などで除外した方向は、その後の開き幅・IK評価を省略する。','',
        '| 段階 | 結果 |','| --- | ---: |',
        f"| 方向・姿勢の検査 | {counts['orientations_tested']:,} |",
        f"| 閉鎖端点の足条件で除外 | {counts['closed_foot_rejected']:,} |",
        f"| 面積条件で除外 | {counts['area_rejected']:,} |",
        f"| 開き幅・挿入・接近・閉鎖掃引条件で除外 | {counts['opening_or_approach_rejected']:,} |",
        '| 布と手先の幾何条件を通過 | 34配置（異なる手先姿勢32、足首姿勢10種類） |',
        '| 位置・姿勢の右腕IKを通過 | 手先姿勢1種類 |',
        '| 従来の関節余裕等を含む最終採用 | 0配置 |','',
        '幾何条件を通った32の手先姿勢は各48初期値、計1536試行で右腕IKを調べた。残り31姿勢は位置・姿勢のIKが成立せず、1姿勢は関節余裕の基準を満たさなかった。',
        '幾何条件を通過した配置の接近方向はすべて甲側寄りで、今回の足裏側寄りの方向は残らなかった。',
        '上記は有限のグリッドでの結果であり、連続した全姿勢での不可能性を証明したものではない。','',
        '### 最も近いNo.39の新候補','',
        '[No.39／新形状geometry-0003](pose-39/search/geometry-0003.json) は、開口から15mm、初期開き30mm、閉鎖時射影面積4.085mm²、足との余裕約1.334mm。',
        '位置・姿勢のIKに成功した14試行（近い解をまとめると7解）があるが、元の探索での最大関節余裕は約1.08°。',
        f"右腕の冗長性と既存の位置・姿勢許容誤差の範囲で余裕を増やす追加探索でも、得られた最大値は約{best['minimum_joint_margin_deg']:.2f}°で、5°基準に届かなかった。第6・第7関節が制限付近にある。",
        'さらに、元のIK試行解を全身幾何モデルで診断すると、グリッパー・手首と頭部や前腕などの自己干渉が残った。50mm手前の接近開始姿勢も48初期値でIK解が見つからなかった。',
        'したがって、関節余裕の基準だけを外せばそのまま実行できる候補ではない。追加最適化した姿勢を、把持成功した姿勢として扱っていない。',
        '詳細は [余裕の追加探索](margin-optimization.json)、[全身幾何診断](closest-ik-diagnostics.json)、[接近前のIK](closest-preapproach-ik.json)。','',
        '### 以前の60配置との対応','',
        '同じ材料上の把持点・方位・傾き・指割り当てを新形状上で再構成すると、60件中16件が閉鎖位置の足条件、18件が面積、26件が開き幅・接近等の条件で除外された。',
        '旧配置の世界座標をそのまま固定した比較ではなく、変形後の位置・法線へ追従したパラメータの再評価である。',
        '新形状の探索はメッシュ・方向グリッド・開き幅の刻みも細かくしているため、旧60件と新34件の件数差だけで膨らみの効果を評価しない。','',
        '### 方向グリッド以外の補助探索','',
        'No.3・No.12では、関節制限から5°以上離れた実際の腕姿勢から手先の向きを提案する方法も試した。位置IK768試行から延べ1044手先姿勢を検査したが、1042姿勢が閉鎖端点の足条件、2姿勢が面積条件で除外された。',
        '把持中心断面での指の厚みは衝突形状上で約17.1mmある。半径を5mm増やしても足側の指と布の空間が常に十分になるわけではなく、これが挿入条件の制約の一因と考えられる。','',
        '## 検証・再現','',
        '[独立検証結果](validation.json) は32形状の半径増分・帯外不変・細分化前後の面積・Native一致・開始姿勢、32件の幾何レコード、1536 IK試行を再計算して確認する。',
        '到達から閉鎖までの動作実行には進んでいない。','',
        '```bash',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/prepare_bulged_cuffs.py',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/capture_bulged_cuffs.py',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_bulged_cuffs.py --workers 4 --tilts -75 -60 -45 -30 -15 0 15 30 45 60 75',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/search_bulged_arm_guided.py',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/optimize_bulged_ik_margin.py',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/diagnose_bulged_ik.py',
        'PYTHONPATH=/tmp/sock-grasp-deps:$PYTHONPATH python3 scripts/validate_bulged_cuffs.py',
        'python3 scripts/report_bulged_cuffs.py','```','',
        'Nativeの写真比較を撮り直す場合は `capture_bulged_cuffs.py --numbers 3 12` を使用できる。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
