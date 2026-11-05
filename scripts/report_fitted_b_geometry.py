#!/usr/bin/env python3
"""Compare B toe-section continuity, actual surface fit and native photographs."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sock-natural-b-matplotlib')
import argparse,csv,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.fitted_b_cuff import PROFILE
from sock_dressing_simulation.natural_b_shape import section_perimeters
from sock_dressing_simulation.reaching_dataset import file_sha256


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/b-natural-toe');a=ap.parse_args();out=a.output.resolve();states={};rows=[]
    for path in sorted((out/'final-shapes').glob('pose-*/state.json')):
        s=json.loads(path.read_text());assert s['shape_profile']==PROFILE and s['cuff_fit']['passed'];n=s['pattern']['number'];states[n]=s;f=s['cuff_fit'];e=s['toe_envelope']
        levels,perimeters=section_perimeters(s['particles_world'],np.array(s['source']['rest_positions']))
        old=json.loads((ROOT/f'artifacts/diverse-left-reaching/b-native-fit-drape/probe/states/pose-{n:02d}-B.json').read_text());old_z,old_c=section_perimeters(old['particles_world'],np.array(old['source']['rest_positions']))
        tip=e['anatomical_tip_material_depth_m']
        rows.append(dict(number=n,pitch_deg=s['pattern']['pitch_deg'],roll_deg=s['pattern']['roll_deg'],anchor_depth_mm=e['toe_envelope_anchor_depth_m']*1000,forefoot_perimeter_mm=e['toe_envelope_anchor_perimeter_m']*1000,previous_toe_perimeter_mm=float(np.interp(tip,old_z,old_c))*1000,new_toe_perimeter_mm=float(np.interp(tip,levels,perimeters))*1000,minimum_toe_forefoot_ratio=f['toe_continuity']['minimum_toe_to_forefoot_perimeter_ratio'],minimum_adjacent_ratio=f['toe_continuity']['minimum_adjacent_perimeter_ratio'],free_sleeve_perimeter_mm=e['free_sleeve_perimeter_m']*1000,flatness=e['free_sleeve_flatness'],bend_radius_mm=e['free_sleeve_bend_radius_m']*1000,contact_skin_median_mm=f['wrapped_skin_distance_m']['median']*1000,peak_minimum_mm=f['peak_radius_increment_min_m']*1000,peak_maximum_mm=f['peak_radius_increment_max_m']*1000,passed=f['passed'],state_file=str(path.relative_to(ROOT)),state_sha256=file_sha256(path)))
    summary=dict(shape_profile=PROFILE,rows=rows,all_passed=all(r['passed'] for r in rows),scope='B-specific frozen geometric reference; cloth deformation, gravity settling, friction retention and pulling are not validated',common_initial_configuration='config/diverse_left_reaching_shared.yaml',common_initial_sha256=file_sha256(ROOT/'config/diverse_left_reaching_shared.yaml'))
    (out/'shape-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'shape-summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[n for n in [8,10,3] if n in states];fig,axes=plt.subplots(1,len(chosen),figsize=(5*len(chosen),4.5),squeeze=False)
    for j,n in enumerate(chosen):
        s=states[n];e=s['toe_envelope'];old=json.loads((ROOT/f'artifacts/diverse-left-reaching/b-native-fit-drape/probe/states/pose-{n:02d}-B.json').read_text());ax=axes[0,j]
        for state,color,label in [(old,'#cf8a29','Previous B'),(s,'#2478b5','B-specific continuous envelope')]:
            levels,perimeters=section_perimeters(state['particles_world'],np.array(state['source']['rest_positions']));ax.plot(levels*1000,perimeters*1000,color=color,label=label)
        ax.axvline(e['anatomical_tip_material_depth_m']*1000,color='#555',ls='--',label='Anatomical toes');ax.axvspan(e['garment_cap_start_m']*1000,300,color='#ddd',alpha=.5,label='Garment closed-end rounding')
        ax.set(title=f'No.{n}: section circumference',xlabel='Material distance from opening (mm)',ylabel='Circumference (mm)',ylim=(0,None));ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'toe-circumference-comparison.png',dpi=160);plt.close(fig)
    fields=['number','forefoot_perimeter_mm','previous_toe_perimeter_mm','new_toe_perimeter_mm','minimum_toe_forefoot_ratio','minimum_adjacent_ratio','contact_skin_median_mm','peak_minimum_mm']
    table=''.join('<tr>'+''.join('<td>'+str(round(r[k],3))+'</td>' for k in fields)+'</tr>' for r in rows)
    old_images='../b-native-fit-drape/initial-views'
    (out/'shape-report.html').write_text('''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Bのつま先形状と外周の連続性</title>
<style>body{font:16px sans-serif;max-width:1300px;margin:40px auto;padding:0 16px;line-height:1.7}img{max-width:100%}.compare{display:flex}.compare figure{width:32%;margin:0.7%}table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccc;padding:6px}</style>
<h1>Bのつま先形状と外周の連続性</h1>
<p>前回は皮膚のつま先断面まで靴下を絞り、細い自由部分へ接続したため、局所的なくびれができていた。今回は開口位置Bの前足部から基準外周を求め、足先を包む部分から空いた部分まで滑らかに接続した。自由部分は外周を保ちながら扁平にし、残り長さと断面寸法に応じて下へ曲げる。靴下自体の閉じた端の25mmだけを丸く絞る。</p>
<p>Aの54%という自然長を適用する方式は使用しない。Bの形状は足首角度・足形状・差し込み深さから決める。ロボットの初期姿勢と配置は共通設定を維持し、開口帯30mm・中心15mmの半径+5mmも維持する。</p>
<p>これは凍結した基準形状の幾何モデル。布全体の変形・重力釣合い・摩擦保持・牽引を検証した結果ではない。皮膚交差・自己交差・外周の連続性は実際に使うメッシュで検査した。形状修正を新しい把持軌道の成功とは扱わない。</p>
<h2>同じ足首角度・撮影位置での比較（No.10）</h2><p>拡大図では布を青く表示し、ロボットを非表示にしている。通常表示で両腕が見える初期姿勢は下のリンクから参照できる。</p>
<div class="compare"><figure><img src="'''+old_images+'''/pose-10-A/native-initial/sock-inspection.png"><figcaption>A</figcaption></figure><figure><img src="'''+old_images+'''/pose-10-B/native-initial/sock-inspection.png"><figcaption>前回B：つま先のくびれ</figcaption></figure><figure><img src="initial-views/pose-10-B/native-initial/sock-inspection.png"><figcaption>今回B：外周を保った連続形状</figcaption></figure></div>
<p><a href="initial-views/pose-10-B/native-initial/sock-inspection-front.png">別方向からの拡大図</a> / <a href="initial-views/pose-10-B/native-initial/fixed-camera-after.png">両腕を含む初期姿勢</a> / <a href="shapes.zip">Bの形状データ一式</a></p>
<h2>つま先と自由部分の外周</h2><img src="toe-circumference-comparison.png"><p>人体のつま先位置と、靴下自体の閉じた端を区別する。つま先付近から端の丸みが始まる手前まで、外周を前足部基準の80%以上、隣接断面間の外周比を85%以上とする検査を加えた。今回の生成値は自由部分へ90%まで滑らかに移行する設定。端面中央の頂点は外周計算から除いている。</p>
<table><tr><th>No.</th><th>前足部外周mm</th><th>前回のつま先位置外周mm</th><th>今回mm</th><th>最小外周/前足部</th><th>隣接断面の最小外周比</th><th>接触部分と皮膚の隙間中央値mm</th><th>帯中心の半径増分mm</th></tr>'''+table+'''</table><p>接触部分の隙間は材料深さ30mmから前足部基準断面までで評価。No.3と43は足首角度が重複しており、足首角度は10条件、保存形状は11件。Excelの素材・しわ等の全条件を再現した結果ではない。</p>
<p>詳細はshape-summary.jsonとvalidation/final-audit.jsonを参照。</p></html>''')
    with zipfile.ZipFile(out/'shapes.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        z.write(out/'shape-summary.json','shape-summary.json');z.write(out/'shape-summary.csv','shape-summary.csv');z.write(ROOT/'config/diverse_left_reaching_shared.yaml','shared_initial_configuration.yaml')
        for path in sorted((out/'final-shapes').glob('pose-*/*.json')):z.write(path,str(path.relative_to(out/'final-shapes')))
    print('B SHAPE REPORT',len(rows),'states, all passed',summary['all_passed'],flush=True)


if __name__=='__main__':main()
