#!/usr/bin/env python3
"""Audit completed lit data, publish comparisons, then activate the qualified dataset."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageDraw
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.reaching_collection import sha256


def smoke(out, name, expected):
    folder=out/name
    parameter=json.loads((folder/'parameter.json').read_text())
    history=json.loads((folder/'loss.json').read_text())
    if not parameter['completed'] or not parameter['test']['finite'] or parameter['episodes']!=expected:
        raise ValueError('Training smoke did not pass')
    for epoch in history:
        for split in ('train','validation'):
            if not epoch[split]['finite'] or not math.isfinite(epoch[split]['loss']):raise ValueError('Nonfinite smoke loss')
    if not (folder/'SARNN_latest.pth').is_file():raise ValueError('Missing smoke checkpoint')
    return parameter


def comparison(ep, source, index, destination):
    width,height=400,300
    columns=(('head right RGB','camera_right'),('SAM2 sock mask','camera_right_mask/sock_mask'),
             ('sock relative depth','depth_mask/sock_depth'),('SAM2 foot mask','camera_right_mask/foot_mask'),
             ('foot relative depth','depth_mask/foot_depth'))
    sheet=Image.new('RGB',(width*len(columns),height*2+70),'#202631');draw=ImageDraw.Draw(sheet)
    for row,(label,folder) in enumerate((('BEFORE',source),('AFTER',ep))):
        for col,(title,relative) in enumerate(columns):
            with Image.open(folder/relative/f'{index}.png') as image:
                sheet.paste(image.convert('RGB').resize((width,height)),(col*width,row*(height+35)+35))
            draw.text((col*width+8,row*(height+35)+10),f'{label} / {title} / frame {index}',fill='white')
    sheet.save(destination,quality=92)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/reaching-shareset-lit-v3')
    ap.add_argument('--wait',action='store_true');ap.add_argument('--no-activate',action='store_true')
    a=ap.parse_args();out=a.output.resolve()
    required=[out/'collection.json',out/'model-representative-smoke/parameter.json',out/'model-all12-smoke/parameter.json']
    while a.wait and not all(p.exists() for p in required):time.sleep(20)
    representative=smoke(out,'model-representative-smoke',dict(train=4,validation=1,test=1))
    all12=smoke(out,'model-all12-smoke',dict(train=8,validation=2,test=2))
    collection=json.loads((out/'collection.json').read_text())
    # Earlier completed/resumed renders predate the explicit scope fields.
    # Preserve their metadata before adding the same acquisition description;
    # contacts were enabled on those microsteps, and that remains recorded.
    for row in collection['rows']:
        ep=Path(row['path']);path=ep/'metadata.json';meta=json.loads(path.read_text())
        backup=ep/'confirmation/metadata-before-lighting-scope.json'
        if not backup.exists():shutil.copy2(path,backup)
        poses=[json.loads(line) for line in (ep/'camera_poses.jsonl').read_text().splitlines()]
        meta.update(rendering_is_dynamics_replay=False,
            rendering_contact_policy='Recorded pose placement with 1 microsecond articulation synchronization; no new dynamics/effort labels. Per-frame rendering_contacts_disabled records contact handling (absent means enabled).',
            rendering_contacts_disabled_frames=sum(bool(p.get('rendering_contacts_disabled')) for p in poses),
            source_physics_steps_unchanged=meta.get('source_physics_steps_unchanged',True),physics_steps_unchanged=None)
        if 'not measured during visual re-render' not in meta['effort_scope']:
            meta['effort_scope']+='; immutable source recording, not measured during visual re-render'
        for key in ('head_isolated_repaired_frames','perception_repaired_frames'):
            meta.pop(key,None)
        path.write_text(json.dumps(meta,indent=2)+'\n')
    with (out/'stored-data-audit.log').open('w') as stream:
        subprocess.run(['/usr/bin/python3',str(ROOT/'scripts/audit_lit_reaching_shareset.py'),'--output',str(out)],stdout=stream,stderr=subprocess.STDOUT,check=True)
    audit=json.loads((out/'stored-data-audit.json').read_text())
    if not audit['passed'] or audit['episodes']!=12 or audit['frames']!=6607 or audit['original_overview_copies_verified']!=6607:
        raise ValueError('Incomplete dataset audit')
    source_root=Path(collection['source_collection']).parent
    for name in ('reference-format.json','camera-migration.json'):
        shutil.copy2(source_root/name,out/name)
    comparison_dir=out/'comparisons';comparison_dir.mkdir(exist_ok=True)
    cards=[];episode_rows=[]
    for row,quality in zip(collection['rows'],audit['rows']):
        ep=Path(row['path']);meta=json.loads((ep/'metadata.json').read_text());source=Path(meta['source_episode'])
        index=min(range(row['frames']),key=lambda i:quality['before_visible_iou'][i])
        image=comparison_dir/(row['id']+'.jpg');comparison(ep,source,index,image)
        record=dict(id=row['id'],split=row['split'],frames=row['frames'],
                    before_mean_iou=sum(quality['before_visible_iou'])/row['frames'],
                    after_mean_iou=sum(quality['after_visible_iou'])/row['frames'],
                    after_minimum_iou=min(quality['after_visible_iou']),comparison_frame=index)
        episode_rows.append(record)
        cards.append(f'<article><h2>{html.escape(row["id"])} ({row["split"]}, {row["frames"]} frames)</h2>'
                     f'<p>Mean visible IoU: {record["before_mean_iou"]:.3f} → {record["after_mean_iou"]:.3f}. '
                     f'After minimum: {record["after_minimum_iou"]:.3f}. Selected frame: {index} (lowest previous IoU).</p>'
                     f'<a href="comparisons/{image.name}"><img loading="lazy" src="comparisons/{image.name}"></a></article>')
    with (out/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episode_rows[0]));writer.writeheader();writer.writerows(episode_rows)
    report=dict(collection,completed=True,perception_passed=True,stored_data_audit_passed=True,
                learning_ready=False,dynamic_grasp_validated=False,reaching_learning_ready=True,
                model_accuracy_validated=False,smoke_only=True,
                training_smoke=dict(representative=representative,all12=all12),
                before_visible_sock_iou=audit['before_visible_sock_iou'],after_visible_sock_iou=audit['after_visible_sock_iou'],
                original_overview_copies_verified=audit['original_overview_copies_verified'],
                metric_scope=audit['metric_scope'],lighting_profile='balanced',
                correction='Separate opposite-face normal accumulators; ambient .20, sun .65 at [55,180,0], shadow .25, two .25 directional fills at [40,90,0] and [40,270,0]',
                acquisition_scope='Photometric re-render of immutable measured teaching states; CSV/time/effort labels are source recordings, not new dynamics measurements',
                finalizer_source_sha256=sha256(__file__))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    before=audit['before_visible_sock_iou'];after=audit['after_visible_sock_iou']
    page=f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>ShareSet 靴下画像の照明・マスク修正</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1450px;margin:32px auto;padding:0 20px;color:#202631}}img{{max-width:100%;height:auto}}article{{border-top:1px solid #cbd2db;margin:32px 0}}h1{{font-size:28px}}h2{{font-size:20px}}code{{background:#eee}}.refs{{display:flex;gap:12px}}.refs img{{width:19%;object-fit:contain}}</style>
<h1>ShareSet 靴下画像の照明・マスク修正</h1>
<p>12経路・6,607フレームの頭部ステレオ画像、SAM2マスク、Depth Anything V2深度を更新しました。表裏の法線が相殺して黒く描画されていたため、描画用頂点の法線を分離しました（ゼロ法線6,240→0、物理粒子1,057個は同じ）。主光の方向と強度を調整し、弱い左右補助光と25%の影を設定しています。</p>
<p>可視のシミュレーション布とのIoU：平均 {before['mean']:.3f} → <strong>{after['mean']:.3f}</strong>、下位10% {before['p10']:.3f} → {after['p10']:.3f}、最小 {before['minimum']:.3f} → {after['minimum']:.3f}。全フレームでIoU≥0.90、布・足の投影支持率≥0.95を検査済みです。可視領域の独立ID描画は検査専用で、学習マスクはSAM2の出力です。</p>
<p>下の実機画像は指定tarから抽出した参考例です。実機とシミュレーションは姿勢・布形状が異なるため、上記IoUは実機との一致率ではありません。光学的な見え方と欠け・分断の改善を比較してください。</p>
<div class="refs"><img src="reference/camera_right.png" alt="実機の右頭部RGB"><img src="reference/camera_right_mask-sock_mask.png" alt="実機の靴下マスク"><img src="reference/depth_mask-sock_depth.png" alt="実機の靴下深度マスク"><img src="reference/camera_right_mask-foot_mask.png" alt="実機の足マスク"><img src="reference/depth_mask-foot_depth.png" alt="実機の足深度マスク"></div>
<p>関節角・関節力・時刻・50Hzログは元データとバイト単位で同一です。記録姿勢を固定して再描画し、画像用の新しい力・接触計測とは扱いません。旧頭部画像は各経路の <code>confirmation/previous_head_data/</code>、元の俯瞰画像は <code>confirmation/camera_overview_original/</code> で確認できます。</p>
<p>深度の較正はtrainの8経路・4,542画像のみから固定しました。validation/testは較正に使っていません。全59,463画像の復号・マスク深度整合・対象の上限飽和率≤2%を独立に確認し、ShareSetで代表経路と全12経路を各1 epoch学習しました。これは固定布への接近データの動作確認であり、着衣成功率の検証は含みません。</p>
<p><a href="stored-data-audit.json">全件検査</a> · <a href="report.json">機械可読レポート</a> · <a href="episodes.csv">経路別集計</a> · <a href="lighting-preview-fixed/native-route-initial/lighting-probe/model-comparison.jpg">照明6条件の比較</a></p>
{''.join(cards)}</html>'''
    (out/'report.html').write_text(page)
    if not a.no_activate:
        canonical=ROOT/'artifacts/reaching-shareset-v1'
        if canonical.exists() and (not canonical.is_symlink() or canonical.resolve() not in (source_root,out)):
            raise ValueError('Canonical target has changed unexpectedly')
        activation=dict(completed=True,previous_dataset=str(source_root),active_dataset=str(out),
                        source_dataset_preserved=True,original_overview_copies_verified=6607,
                        report_sha256=sha256(out/'report.json'),audit_sha256=sha256(out/'stored-data-audit.json'))
        for name in ('reaching_head_simulation.yaml','reaching_simulation.yaml'):
            config_path=ROOT.parent/'ShareSet/share_folder/configs'/name
            config=yaml.safe_load(config_path.read_text())
            config['dataset'].update(root='../../../sock_dressing_simulation/artifacts/reaching-shareset-lit-v3/data',
                manifest='../../../sock_dressing_simulation/artifacts/reaching-shareset-lit-v3/data/dataset_reaching.yaml',
                expected_lighting_profile='balanced')
            config['perception']['lighting_profile']='balanced'
            config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        temporary=canonical.with_name(canonical.name+'-lit-next')
        temporary.symlink_to(out.name,target_is_directory=True);os.replace(temporary,canonical)
        (out/'lighting-migration.json').write_text(json.dumps(activation,indent=2)+'\n')
    print('LIT DATASET FINALIZED',str(out), 'activated=',not a.no_activate,flush=True)


if __name__=='__main__':main()
