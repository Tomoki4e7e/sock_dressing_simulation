#!/usr/bin/env python3
"""Audit and report the fixed-shape B endpoint and trajectory exploration."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/b-grasp-search-matplotlib')
import argparse,json,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.reaching_dataset import file_sha256,load_endpoint_samples
from sock_dressing_simulation.fitted_b_cuff import PROFILE


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/b-natural-toe/grasp-search');a=ap.parse_args();out=a.output.resolve();rows=[];initial_sha=file_sha256(ROOT/'config/diverse_left_reaching_shared.yaml');shape_rows=json.loads((out.parent/'shape-summary.json').read_text())['rows'];frozen={r['number']:r for r in shape_rows}
    for mode in ('probe','dense-probe','fine-opening-probe','wide-pose-probe'):
        for path in sorted((out/mode).glob('pose-*.json')):
            d=json.loads(path.read_text());n=d['number'];assert d['shape_profile']==PROFILE and d['initial_sha256']==initial_sha and d['initial_check']['passed']
            source=ROOT/frozen[n]['state_file'];assert file_sha256(source)==frozen[n]['state_sha256']
            s=json.loads((out/mode/'states'/path.name).read_text());canonical=json.loads(source.read_text());np.testing.assert_array_equal(s['particles_world'],canonical['particles_world']);assert s['source']==canonical['source']
            if d.get('frozen_source_sha256'):assert d['frozen_source_sha256']==file_sha256(source)
            if d.get('samples_file'):load_endpoint_samples(d,s,ROOT)
            c=d.get('counts',{});trials=d.get('pose_trials')
            if trials is None:trials=sum(v for k,v in c.items() if k in ('closed_foot','area','opening_or_approach','accepted') or k.startswith('fine_rotation_'))
            rows.append(dict(number=n,mode=mode,pitch_deg=canonical['pattern']['pitch_deg'],roll_deg=canonical['pattern']['roll_deg'],pose_trials=trials,accepted_endpoints=len(d['accepted']),best_capture_mm2=d.get('best_connected_capture_area_mm2'),closed_foot=c.get('closed_foot',0),capture_area=c.get('area',0)+c.get('fine_rotation_area',0),opening_or_approach=c.get('opening_or_approach',0),opening_skin_gap_trials=c.get('opening_skin',0),opening_cloth_gap_trials=c.get('opening_cloth',0),numerical_rejected=c.get('numerical_geometry_rejected',0),elapsed_s=d['elapsed_s'],record=str(path.relative_to(ROOT))))
    polish=[]
    for path in sorted((out/'local-polish').glob('pose-*.json')):
        d=json.loads(path.read_text());assert d['shape_profile']==PROFILE
        polish.append(dict(number=d['number'],accepted_geometry=len(d['accepted_geometry']),evaluations=d['counts']['evaluations'],solver_trials=d['counts']['solver_trials'],record=str(path.relative_to(ROOT))))
    baseline={r['number'] for r in rows if r['mode']=='probe'};assert baseline=={3,8,10,15,19,22,23,31,38,41}
    shapes={n:frozen[n]['state_sha256'] for n in sorted(baseline)}
    routes=[]
    for p in (out/'routes').glob('*-validation.json'):
        d=json.loads(p.read_text())
        if d.get('offline_passed'):routes.append(str(p.relative_to(ROOT)))
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());assert settings['closing_gap_m']==.001 and settings['body_clearance_m']==.001 and settings['minimum_area_m2']==4e-6 and settings['half_extent_m']==.004
    summary=dict(shape_profile=PROFILE,shape_and_initial_configuration_unchanged=True,common_initial_sha256=initial_sha,shape_sha256=shapes,unique_ankle_conditions=10,workbook_B_rows_covered_by_angle=10,duplicate_workbook_B_row='No.43 uses the same angle as No.3',additional_B_angle='No.10: spreadsheet stage A, used here only as an extra ankle angle at stage B',pose_trials=sum(r['pose_trials'] for r in rows),accepted_endpoints=sum(r['accepted_endpoints'] for r in rows),local_polish_geometry=sum(r['accepted_geometry'] for r in polish),local_optimization_evaluations=sum(r['evaluations'] for r in polish),offline_qualified_routes=len(routes),native_verified_routes=0,rows=rows,local_polish=polish,policy=settings,policy_sha256=file_sha256(ROOT/'config/geometric_closure.yaml'),policy_review_sha256=file_sha256(ROOT/'config/geometric_closure_review.yaml'),scope='Bounded finite search under the existing 1 mm / 4 mm2 / fixed 8x8 mm window convention. No proof of impossibility. Shapes, placement and shared initial posture unchanged. No deformable-cloth grasp claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import csv
    with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4.5));numbers=sorted(baseline);dense={r['number']:r for r in rows if r['mode']=='dense-probe'}
    bottom=np.zeros(len(numbers))
    for key,label,color in [('closed_foot','Closed hand vs skin','#ba6557'),('capture_area','Capture rule','#e6b967'),('opening_or_approach','Open hand / insertion','#6293b5')]:
        values=np.array([dense[n][key]/dense[n]['pose_trials']*100 for n in numbers]);ax.bar([str(n) for n in numbers],values,bottom=bottom,label=label,color=color);bottom+=values
    ax.set(xlabel='Workbook ankle-angle No.',ylabel='Percent of dense pose proposals',ylim=(0,100),title='B: rejection stages under unchanged geometric policy');ax.legend(loc='lower center',ncol=3,fontsize=8);fig.tight_layout();fig.savefig(out/'rejection-stages.png',dpi=160);plt.close(fig)
    table=''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in [r['number'],r['mode'],r['pose_trials'],r['accepted_endpoints'],'未集計' if r['best_capture_mm2'] is None else round(r['best_capture_mm2'],3),r['closed_foot'],r['capture_area'],r['opening_or_approach'],round(r['elapsed_s'],1)])+'</tr>' for r in rows)
    (out/'report.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><title>最新版Bの把持軌道探索</title><style>body{font:16px/1.7 sans-serif;max-width:1250px;margin:35px auto;padding:0 16px}table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:6px}img{max-width:100%}</style>
<h1>最新版Bの把持軌道探索</h1><p>足首10条件、'''+f"{summary['pose_trials']:,}"+'''姿勢候補を探索。条件をすべて満たす把持目標姿勢 '''+str(summary['accepted_endpoints'])+''' 件、局所連続最適化での合格姿勢 '''+str(summary['local_polish_geometry'])+''' 件、合格軌道 '''+str(summary['offline_qualified_routes'])+''' 本。</p>
<p>靴下は最新版の連続したつま先形状に固定。人体・台座・ロボットの配置、両腕・グリッパーを含む共通初期設定を固定し、保存形状のSHA-256と頂点配列を照合した。No.3/43の足首角度は重複するため、No.3で探索。No.10は追加の足首角度としてBに適用した。</p>
<p>固定条件：閉じ幅1mm、手と皮膚の離隔1mm以上、実際の対向指面間に連続した布面積4mm²以上、固定の8×8mm評価窓、30mm材料帯。指先の衝突形状・評価窓は変更していない。</p>
<p>従来の材料位置・姿勢格子に加え、周方向64点×帯深さ13点、手先位置のずれ、連続した姿勢・位置の修正、把持前の開き幅の細分化を探索した。開き幅の候補には1.01〜1.25mmも追加したが、最終閉じ幅と離隔・捕捉面積の条件は維持した。幾何判定が数値的に不確定な候補は採用していない。</p>
<p>一部の候補は閉鎖状態の足との離隔と捕捉面積を満たしたが、把持前の指と布・足の非干渉を同時に満たせなかった。把持目標姿勢が成立しなかったため、腕の軌道生成・PhysXでの軌道再生には進めていない。探索範囲内で未成立という結果であり、不可能性の証明ではない。</p>
<img src="rejection-stages.png"><table><tr><th>No.</th><th>探索</th><th>姿勢候補数</th><th>合格把持姿勢</th><th>最大捕捉面積mm²</th><th>閉鎖時の足干渉</th><th>捕捉条件不合格</th><th>開口・挿入不合格</th><th>探索時間s</th></tr>'''+table+'''</table>
<p>最大捕捉面積は、閉鎖時に足との離隔が通った候補の診断値であり、開口・挿入・全身・軌道条件を通ったことを意味しない。表の不合格数は段階別の診断数で、従来格子の局所回転再探索等を含む。詳細は<a href="summary.json">summary.json</a>と<a href="summary.csv">summary.csv</a>、各probeフォルダ・local-polishを参照。</p>
<p><a href="../shape-report.html">使用した最新版Bの形状</a> / <a href="../report.html">採用軌道の集計</a> / <a href="validation/final-audit.json">最終監査・テスト結果</a></p></html>''')
    print(json.dumps({k:summary[k] for k in ['unique_ankle_conditions','pose_trials','accepted_endpoints','local_polish_geometry','local_optimization_evaluations','offline_qualified_routes']}),flush=True)


if __name__=='__main__':main()
