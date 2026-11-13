#!/usr/bin/env python3
"""Audit saved Unity videos, CUDA traces and remeasure the terminal cloth mesh."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.contact_pregrasp import opening_edges
from sock_dressing_simulation.pregrasp_gap_measurement import measure


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    summary = json.loads((out / 'report.json').read_text())
    assert summary['completed'] and summary['finished'] == summary['expected']
    inventory = json.loads((out / 'checkpoint-inventory.json').read_text())
    checkpoints = {c['name']: c for c in inventory}
    manifest = json.loads((out / 'manifest.json').read_text())
    selection_path = out / 'selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else dict(split='test', ids=None)
    cases = {c['id']: c for c in manifest['rows'] if c['split'] == selection['split']
             and (selection['ids'] is None or c['id'] in selection['ids'])}
    if selection['ids'] is not None:
        assert set(cases) == set(selection['ids'])
    with (out / 'terminal-measurements.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    keys = {(r['checkpoint'], r['case']) for r in csv_rows}
    assert len(keys) == len(csv_rows) == len(cases) * len(checkpoints)
    assert keys == {(c, case) for c in checkpoints for case in cases}
    checked, files = [], []
    for row in csv_rows:
        base = out / row['checkpoint'] / row['case'] / 'policy'
        report = json.loads((base / 'report.json').read_text())
        steps = (cases[row['case']]['frames'] + 1) // 2 - 1
        assert report['completed'] and report['steps'] == steps
        assert report['actual_physics_steps'] == steps * 10 and report['physics_batch_steps'] == 1
        assert report['checkpoint_sha256'] == checkpoints[row['checkpoint']]['sha256']
        assert report['device']['precision'] == 'float32' and report['device']['device'] == 'cuda'
        assert all(d.startswith('cuda') for d in report['device']['model_devices'].values())
        assert not report['teacher_trajectory_used'] and not report['command_blending']
        assert report['cloth_maximum_drift_m'] < 2e-5
        with gzip.open(base / 'terminal-geometry.json.gz', 'rt') as stream:
            mesh = json.load(stream)
        points = np.array(mesh['particles_world'])
        faces = np.array(mesh['faces'])
        depth = np.array(mesh['material_depth_m'])
        edges = opening_edges(faces, depth)
        saved = json.loads((base / 'terminal-cloth-between-fingers.json').read_text())
        areas = {}
        for side in ('left', 'right'):
            g = saved[side]
            p, rotation = np.array(g['tcp_world_m']), np.array(g['rotation'])
            fingers = [(np.array(g['measured_finger_world_m'][suffix]) - p) @ rotation
                       for suffix in ('left_finger_link', 'right_finger_link')]
            actual = measure((points - p) @ rotation, faces, depth, fingers,
                             (points[edges] - p) @ rotation)
            for name in ('cloth_surface_area_between_fingers_mm2',
                         'material_15_30mm_band_area_between_fingers_mm2',
                         'opening_rim_length_between_fingers_mm'):
                assert np.isclose(actual[name], g[name], rtol=1e-7, atol=1e-7)
                assert np.isclose(g[name], report['geometry'][side][name], rtol=1e-7, atol=1e-7)
            area = actual['cloth_surface_area_between_fingers_mm2']
            assert np.isclose(area, float(row[side + '_cloth_area_mm2']), rtol=1e-7, atol=1e-7)
            areas[side] = area
        trace = [json.loads(line) for line in (base / 'motion-trace.jsonl').read_text().splitlines()]
        assert len(trace) == steps
        for tick in trace:
            for field in ('observed_angle', 'filtered_observed_torque', 'executed_command', 'feedback_angle', 'feedback_torque'):
                assert len(tick[field]) == 18 and np.isfinite(tick[field]).all()
            assert np.isfinite(tick['prediction']['action']).all()
            assert tick['prediction']['measured_blend'] == 0
        videos = []
        for name, frames, size in [('overview.mp4', steps + 1, (1280, 960)),
                                   ('head-input.mp4', steps, (640, 480))]:
            path = base / name
            probe = json.loads(subprocess.check_output(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                 'stream=codec_name,width,height,nb_frames,r_frame_rate', '-of', 'json', str(path)], text=True))['streams'][0]
            assert probe['codec_name'] == 'h264' and int(probe['nb_frames']) == frames
            assert (probe['width'], probe['height']) == size and probe['r_frame_rate'] == '5/1'
            assert any(c['decoded'] and c['frames'] == frames and Path(c['path']).name == name
                       for c in report['video_validation'])
            videos.append(dict(path=str(path.relative_to(out)), frames=frames, sha256=digest(path), decoded=True))
        for path in [base / 'overview-final.png'] + [base / f'{side}-{view}.png'
                     for side in ('left', 'right') for view in ('gap-side', 'gap-opposite')]:
            with Image.open(path) as picture:
                picture.verify()
        qa_failures = sum(not t['prediction'].get('sam_qa_passed', True) for t in trace)
        checked.append(dict(checkpoint=row['checkpoint'], case=row['case'], steps=steps,
                            left_cloth_area_mm2=areas['left'], right_cloth_area_mm2=areas['right'],
                            sam_qa_failed_frames=qa_failures,
                            sam_reprompt_failed_frames=sum(bool(t['prediction'].get('sam_reset_error')) for t in trace),
                            maximum_input_out_of_training_range_fraction=max(t['prediction']['input_out_of_range_fraction'] for t in trace),
                            native_penetration_events=report['native_penetration_event_count'], videos=videos))
        checked[-1]['native_contact_records_dropped']=report.get('native_contact_records_dropped',0)
        for name in ('report.json', 'device.json', 'terminal-cloth-between-fingers.json',
                     'terminal-geometry.json.gz', 'motion-trace.jsonl'):
            path = base / name
            files.append(dict(path=str(path.relative_to(out)), bytes=path.stat().st_size, sha256=digest(path)))
    assert all(digest(Path(c['path'])) == c['sha256'] for c in inventory)
    positive_control = out / 'measurement-positive-control.json'
    if positive_control.exists():
        assert json.loads(positive_control.read_text())['passed']
    audit = dict(passed=True, expected=len(keys), checked=len(checked), video_count=2 * len(checked),
                 total_policy_steps=sum(r['steps'] for r in checked),
                 total_physics_steps=10 * sum(r['steps'] for r in checked),
                 geometry_remeasured_from_saved_native_mesh=True,
                 all_checkpoints_unchanged=True, rows=checked, files=files)
    audit['measurement_positive_control_passed'] = positive_control.exists()
    (out / 'validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    summary['left_cloth_present'] = sum(r['left_cloth_present'] for r in summary['rows'])
    summary['right_cloth_present'] = sum(r['right_cloth_present'] for r in summary['rows'])
    summary['both_cloth_present'] = sum(r['left_cloth_present'] and r['right_cloth_present'] for r in summary['rows'])
    summary['artifact_validation_passed'] = True
    summary['validation_report'] = 'validation.json'
    (out / 'report.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    page_path = out / 'report.html'
    page = page_path.read_text()
    if 'href="input-quality.csv"' not in page:
        page = page.replace('<a href="report.json">全結果JSON</a>',
                            '<a href="report.json">全結果JSON</a> · <a href="input-quality.csv">入力品質CSV</a> · '
                            '<a href="validation.json">全動画・面積の検証</a> · <a href="README.txt">評価条件</a>')
        page_path.write_text(page)
    with (out / 'input-quality.csv').open('w', newline='') as stream:
        fields = [k for k in checked[0] if k != 'videos']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(checked)
    status_path = out / 'execution-status.json'
    if status_path.exists():
        status = json.loads(status_path.read_text())
        if status.get('failures'):
            original = out / 'initial-attempt-execution-status.json'
            if not original.exists():
                shutil.copy2(status_path, original)
            status['recovered_attempt_failures'] = status['failures']
            status['failures'] = []
            status['completed'] = True
            status['final_artifact_validation_passed'] = True
            status_path.write_text(json.dumps(status, indent=2) + '\n')
    shutil.copy2(Path(__file__), out / 'programs' / Path(__file__).name)
    label = {'train': '教示データ（学習に使用）', 'validation': '検証データ', 'test': 'テストデータ'}[selection['split']]
    text = f'''Flow GPU閉ループ推論

{len(checkpoints)} チェックポイント × {len(cases)} ケース（{label}） = {len(checked)} 試行完走。
全景と頭部入力の動画を各1本、合計{2 * len(checked)}本保存。
report.html: 動画・爪の接写と面積を確認する入口。
terminal-measurements.csv: 全試行・左右別の終端靴下表面積 [mm²]。
input-quality.csv: マスク品質の低下・学習範囲外の入力・接触イベント数。
validation.json: 動画フレーム数・全フレームのデコード・CUDA実行・保存メッシュからの面積再計算の検証。
各チェックポイント/<ケース>/policy/: 動画、接写、実測メッシュ、関節動作ログ。

Flow・SAM2・Depth AnythingはCUDAで実行。Flow推論はFP32、ODE 8ステップ、
各ケースで乱数シード0。8ステップチャンクの先頭を実行し、実測観測から毎0.2秒再計画。
物理刻み20ms、通信・接触記録は1物理ステップごと。教示軌道を指令に使用しない。
腕で靴下が遮蔽され再プロンプトできない場合、その時刻の実際のSAM追跡マスクを保持し、
品質低下をログに残す。正解マスクや教示画像への置き換えはしない。
駆動力の平滑化は過去サンプルのみを使う因果的な方法で、学習時の前後サンプルを使う処理とは異なる。
接触イベントの補助ログがUnity側の上限に達した場合、欠落件数をinput-quality.csvに記録する。
物理状態・実測画像・終端メッシュの取得は継続し、面積は保存した実測メッシュから再検証する。

人体と靴下は教示と同じ固定条件。面積は終端の実測布三角形を、実測された有限な爪の
凸包・対向面で切り抜いた表面積。投影面積ではない。布が爪間にあることと、爪の閉鎖、
摩擦保持、引き上げの成功は別。0 mm²は終端の爪間に布がなかったことを示す。
diagnostics/ は動作確認・再実行前の記録で、最終集計に含めない。
'''
    (out / 'README.txt').write_text(text)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('rows', 'files')}, indent=2))


if __name__ == '__main__':
    main()
