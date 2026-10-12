#!/usr/bin/env python3
"""Finish offline artifacts after the immutable V88 replay closes its videos."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
from audit_foot_skin_containment import audit
from report_foot_containment import outcome, required_regions_contained


def finalize(directory, visual_confirmation=None):
    directory = Path(directory)
    report = json.loads((directory/'foot-containment.json').read_text())
    result = outcome(report, visual_confirmation)
    rows = report['rows']
    raw = [json.loads(line) for line in (directory/'live_frame_diagnostics.jsonl').read_text().splitlines()]
    result.update(completed_commands=report['completed_commands'],
                  recording_complete=report['recording_complete'],
                  visual_review_complete=visual_confirmation is not None,
                  producer_report_sha256=hashlib.sha256((directory/'foot-containment.json').read_bytes()).hexdigest())
    result['maximum_corrected_penetration_mm'] = 1000*max(
        r['dressing_qa']['geometric_maximum_cloth_foot_penetration_m'] for r in raw)
    result['maximum_grasp_offset_drift_m'] = max(
        r['dressing_qa']['foot_collision_safety']['maximum_grasp_offset_drift_m'] for r in raw)
    result['maximum_render_position_error_m'] = max(
        r['dressing_qa']['foot_collision_safety']['maximum_render_position_error_m'] for r in raw)
    result['maximum_stretch_by_class'] = {
        name:max(r['stretch']['edge_classes'][name]['maximum_stretch'] or 0 for r in raw)
        for name in raw[0]['stretch']['edge_classes']}
    result['stretch_failed_commands'] = [r['command'] for r in rows if not r['stretch_ok']]
    result['safety_failed_commands'] = [r['command'] for r in rows if not r['safety_ok']]
    result['containment_sampling'] = report['sampling_limit']
    result['opening_passage_limit'] = report['passage_limit']
    skin_path = Path('artifacts/phase4/foot-conforming-fixed-motion/human-skin-reconstruction.json')
    geometry_path = Path('RCareUnity/Assets/SockDressing/Resources/FootSkinHullGeometry.json')
    try:
        skin = audit(directory, skin_path, geometry_path)
        (directory/'rendered-skin-containment.json').write_text(json.dumps(skin, indent=2)+'\n')
        result['supplementary_skin_audit_available'] = True
    except ValueError as error:
        skin = {'rows':[]}
        result['supplementary_skin_audit_available'] = False
        result['supplementary_skin_audit_error'] = str(error)
    videos = json.loads((directory/'video-provenance.json').read_text())
    first = next((r['command'] for r in rows if required_regions_contained(r)), None)
    selected = {}
    if first is not None:
        selected['geometry_first_frame'] = first-1
    streak = result['first_three_command_geometric_containment_completed_at']
    if streak is not None:
        selected.update({f'geometry_streak_{command}':command-1
                         for command in range(streak-2, streak+1)})
    for label, index in selected.items():
        for name, video in videos.items():
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',video['path'],
                '-vf',f'select=eq(n\\,{index})','-frames:v','1',
                str(directory/f'{label}-{name}.png')], check=True)
    os.environ.setdefault('MPLCONFIGDIR', str(directory/'.plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    commands = [r['command'] for r in rows]
    for name in ('toes','forefoot','heel','ankle','calf'):
        optional = name in ('ankle','calf')
        axes[0].plot(commands, [r['sections'].get(name,{}).get('ratio',np.nan) for r in rows],
                     linestyle='--' if optional else '-', label=name+(' (optional)' if optional else ''))
    axes[0].axhline(.95, color='gray', linestyle=':')
    axes[0].set_ylabel('Contained sample fraction'); axes[0].legend()
    counts = [sum(len(binding['particle_indices']) for binding in
                  r['dressing_qa']['foot_collision_safety']['original_grasp_anchors']) for r in raw]
    axes[1].plot(commands, [r['anchor_error_mm'] if count else np.nan for r,count in zip(rows,counts)])
    axes[1].set_ylabel('Active pin error (mm)')
    right = axes[1].twinx(); right.step(commands, counts, where='mid', color='gray', alpha=.4)
    right.set_ylabel('Active pin particles'); right.set_ylim(-.5,10.5)
    axes[2].plot([r['command'] for r in skin['rows']],
        [1000*r.get('optional_cuff_mean_height_above_ankle_m',np.nan) for r in skin['rows']])
    axes[2].set_ylabel('Cuff above ankle (mm)\n(optional)'); axes[2].set_xlabel('Recorded command')
    for axis in axes: axis.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(directory/'containment-and-grasp-reviewed.png', dpi=160); plt.close(fig)
    sources = directory/'review_sources'; sources.mkdir(exist_ok=True)
    manifest = {}
    for path in (Path(__file__), Path('scripts/report_foot_containment.py'),
                 Path('scripts/audit_foot_skin_containment.py'), skin_path, geometry_path):
        shutil.copy2(path, sources/path.name)
        manifest[path.name] = {'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (sources/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (directory/'foot-containment-outcome.json').write_text(json.dumps(result, indent=2)+'\n')
    if visual_confirmation is None:
        conclusion = '映像の目視レビューは未完了です。包含の確認は確定していません。'
    else:
        conclusion = '足の包含を確認しました。' if result['foot_enclosure_confirmed'] else '足の包含を確認する条件は満たしていません。'
    quality = '固定条件の検証を通過しました。' if result['fixed_condition_validation_passed'] else '固定条件の全項目検証は未達です。'
    (directory/'REPORT_ja.md').write_text(
        ('# V88・SAMDAMSARNN新規自律生成デモ\n\n' if report.get('mode')=='samdamsarnn_closed_loop'
         else '# V88・既存指令の再生デモ\n\n')+conclusion+' '+quality+'\n\n'+
        f'完了：{report["completed_commands"]}/{report["required_commands"]}指令。'+
        f'形状判定による3指令連続の包含：{streak}指令目。'+
        f'最終状態の部位別包含：{result["final_regional_geometric_containment"]}。\n\n'+
        'つま先・前足部・踵を必須とし、足首・下腿への引き上げは参考値です。把持誤差の修正は保留しています。\n\n'+
        f'補正後貫通の最大値：{result["maximum_corrected_penetration_mm"]:.6f} mm。'+
        f'伸び検査の未達指令：{result["stretch_failed_commands"]}。'+
        f'接触検査の未達指令：{result["safety_failed_commands"]}。\n\n'+
        '## 保存物\n\n'+
        '- [上下比較映像](top-bottom-demo.mp4)（左：真上、右：真下）\n'+
        '- [真上映像](top_demo.mp4)・[真下映像](bottom_demo.mp4)・[側面映像](side_demo.mp4)\n'+
        '- 正面映像の場所と全映像のSHA256：[video-provenance.json](video-provenance.json)\n'+
        '- [包含率・把持誤差・参考引き上げ量](containment-and-grasp-reviewed.png)\n'+
        '- [判定結果](foot-containment-outcome.json)・[元の解析](foot-containment.json)\n'+
        '- [実人体メッシュの補助照合](rendered-skin-containment.json)\n\n'+
        '診断用の開口閉鎖メッシュは物理・描画には使っていません。部位別判定は接触用包絡面の各8サンプルに基づき、面積率そのものではありません。実人体メッシュ頂点の集計にも足首の移行部が含まれるため、それだけで必須部位の判定を置き換えません。足先通過は指令端点の開口補間とNative CCD診断による支持であり、Native内部の全時刻を保存した厳密証明ではありません。\n\n'+
        '映像は各指令1フレーム・5 fpsです。1指令の物理時間は0.36秒なので、表示は物理時間の1.8倍速です。撮影では物理時間を進めません。'+
        '把持粒子が0個の区間は把持誤差を未測定として扱い、誤差0への回復とは扱いません。\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--visual-confirmation', choices=('yes','no','pending'), default='pending')
    args = parser.parse_args()
    if args.wait:
        print('Waiting for producer REPORT.md and closed videos.', flush=True)
        while not (args.directory/'REPORT.md').exists():
            time.sleep(30)
    finalize(args.directory, {'yes':True,'no':False,'pending':None}[args.visual_confirmation])
