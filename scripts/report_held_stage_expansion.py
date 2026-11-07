#!/usr/bin/env python3
"""Inventory qualified shapes and map every workbook C--E row to its result."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.held_stage_shapes import DEFAULT_ROOT, STAGES, load_stage_state
from sock_dressing_simulation.pose_patterns import read_patterns


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    workbook = ROOT / 'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx'
    patterns = read_patterns(workbook)
    by_number = {p.number: p for p in patterns}
    inventory = []
    for path in sorted(root.glob('pose-*/state.json')):
        state = read(path)
        pattern = by_number[state['pattern']['number']]
        stage = state['applied_opening_stage']
        state = load_stage_state(root, pattern, stage)
        fit = read(path.parent / 'fit/report.json')
        settling = read(path.parent / 'settling-report.json')
        checks = dict(fit=fit.get('passed', False),
                      physical_hold=fit.get('assessment', {}).get('passed', False),
                      settling=settling.get('gravity_drape', {}).get('passed', False),
                      surface_fit=state['surface_fit'].get('passed', False),
                      final_skin=state['geometry_audit'].get('geometry_ok', False),
                      final_surface=state['surface_audit'].get('surface_ok', False))
        validation = root / f'preview-{pattern.number:02d}' / stage / 'native-validation.json'
        inventory.append(dict(number=pattern.number, stage=stage, pitch_deg=pattern.pitch_deg,
            roll_deg=pattern.roll_deg, passed=all(checks.values()), checks=checks,
            state=str(path), state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            native_readback=read(validation),
            cuff_perimeter_cm=100*state['surface_fit']['after_perimeter_m'],
            native_fit_scale=fit['rest_circumference_scale'],
            gravity_hold_duration_s=fit['gravity_hold_duration_s']))
    qualified = {(r['stage'], r['pitch_deg'], r['roll_deg']): r for r in inventory if r['passed']}
    rows = []
    for pattern in patterns:
        if pattern.stage not in STAGES:
            continue
        key = (pattern.stage, pattern.pitch_deg, pattern.roll_deg)
        peers = [p.number for p in patterns if (p.stage, p.pitch_deg, p.roll_deg) == key]
        canonical = min(peers)
        directory = root / f'pose-{canonical:02d}-{pattern.stage}'
        fit = read(directory / 'fit/report.json')
        diagnostic = read(directory / 'settling-diagnostic.json')
        failure = read(directory / 'failure.json')
        attempts = []
        for attempt_path in sorted(directory.glob('fit-attempts/*/report.json')):
            attempt = read(attempt_path)
            attempts.append(dict(report=str(attempt_path), passed=attempt['passed'],
                target_cuff_force_n=attempt['fit_settings']['target_cuff_force_n'],
                rest_circumference_scale=attempt['rest_circumference_scale'],
                assessment=attempt['assessment'], geometry_audit=attempt['geometry_audit'],
                surface_audit=attempt['surface_audit']))
        state = qualified.get(key)
        reason = ''
        if not state:
            if fit and not fit.get('passed'):
                checks = fit.get('assessment', {}).get('checks', {})
                reason = 'native_fit: ' + ', '.join(k for k, v in checks.items() if not v)
                if all(checks.values()):
                    reason = 'native_fit: geometry/surface audit'
            elif diagnostic:
                reason = 'settling: ' + ', '.join(k for k, v in diagnostic['result']['checks'].items() if not v)
            else:
                log = directory / 'expansion.log'
                text = log.read_text() if log.exists() else ''
                reason = ('initial_geometry_intersection' if 'Preinsert geometry intersects' in text
                          else 'preparation_failed' if failure else 'in_progress' if text else 'not_attempted')
        rows.append(dict(number=pattern.number, stage=pattern.stage, pitch_deg=pattern.pitch_deg,
            roll_deg=pattern.roll_deg, representative_number=canonical, workbook_numbers=peers,
            passed=state is not None, reason=reason,
            state=state['state'] if state else '', log=str(directory / 'expansion.log'),
            fit_report=str(directory / 'fit/report.json') if fit else '', fit_attempts=attempts,
            native_readback_passed=bool(state and state['native_readback'].get('passed')),
            fit_target_force_n=fit.get('fit_settings', {}).get('target_cuff_force_n'),
            cuff_motion_mm=(1000*fit['assessment']['cuff_point_excursion_m']
                            if fit.get('assessment', {}).get('cuff_point_excursion_m') is not None else None),
            cuff_speed_m_s=max((r['cuff_observed_speed_m_s'] for r in fit.get('telemetry', [])[-11:]), default=None),
            fit_geometry_audit=fit.get('geometry_audit', {}),
            fit_surface_audit=fit.get('surface_audit', {})))
    a_rows = [read(p) for p in (ROOT / 'artifacts/bulged-cuff').glob('pose-*/report.json')]
    b_rows = read(ROOT / 'artifacts/diverse-left-reaching/b-natural-toe/shape-summary.json')['rows']
    counts = dict(A=len({(r['pitch_deg'], r['roll_deg']) for r in a_rows if r.get('passed')}),
                  B=len({(r['pitch_deg'], r['roll_deg']) for r in b_rows if r.get('passed')}))
    counts.update({s: sum(k[0] == s for k in qualified) for s in STAGES})
    unique_requested = {(r['stage'], r['pitch_deg'], r['roll_deg']) for r in rows}
    unique_passed = {k for k in unique_requested if k in qualified}
    report = dict(workbook=str(workbook), workbook_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
        scope='Opening stage and ankle pitch/roll; Excel material, shift, diagonal rotation and local deformations are not applied',
        model='Native unpinned fit and 10 s gravity hold, supported gravity tail, supported skin fit, frozen 5 mm cuff bulge',
        adjusted_shape_unpinned_retention_validated=False,
        counts=counts, total_unique=sum(counts.values()), baseline_counts=dict(A=32, B=10, C=1, D=1, E=1),
        added_unique=sum(counts.values())-45, unique_workbook_conditions=len(unique_requested),
        passed_unique_workbook_conditions=len(unique_passed), workbook_rows=rows, inventory=inventory)
    (root / 'expansion-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    fields = ['number', 'stage', 'pitch_deg', 'roll_deg', 'representative_number', 'passed',
              'native_readback_passed', 'reason', 'fit_target_force_n', 'cuff_motion_mm',
              'cuff_speed_m_s', 'state', 'fit_report', 'log']
    with (root / 'expansion-report.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    by_stage = {s: [r for r in inventory if r['stage'] == s and r['passed']] for s in STAGES}
    canvas = Image.new('RGB', (1440, 40 + 210 * max(len(v) for v in by_stage.values())), '#f3f5f7')
    draw = ImageDraw.Draw(canvas)
    for column, stage in enumerate(STAGES):
        draw.text((column * 480 + 12, 12), f'{stage}: {len(by_stage[stage])} unique postures', fill='black')
        for index, row in enumerate(by_stage[stage]):
            x, y = column * 480, 40 + index * 210
            draw.text((x + 12, y + 8), f'No.{row["number"]:02d}  pitch {row["pitch_deg"]:g}  roll {row["roll_deg"]:g}', fill='black')
            for view_index, view in enumerate(('side', 'dorsal')):
                path = root / f'preview-{row["number"]:02d}' / stage / f'{view}.png'
                if path.exists():
                    with Image.open(path) as image:
                        image.thumbnail((240, 180))
                        canvas.paste(image, (x + view_index * 240, y + 28))
    canvas.save(root / 'expansion-overview.jpg', quality=90)
    def link(path, label):
        return f'<a href="{html.escape(os.path.relpath(path, root))}">{html.escape(label)}</a>' if path else ''
    table = []
    for row in rows:
        image = root / f'preview-{by_number[int(row["representative_number"])].number:02d}' / row['stage'] / 'side.png'
        # An alias can use a different representative saved state.
        if row['state']:
            saved_number = read(Path(row['state']))['pattern']['number']
            image = root / f'preview-{saved_number:02d}' / row['stage'] / 'side.png'
        reason = row['reason'].replace('native_fit: low_cuff_motion', '保持試験：開口部の振動が基準超過')
        reason = reason.replace('moderate_tension', '張力が許容範囲外')
        reason = reason.replace('native_fit: geometry/surface audit', '外周フィット後：皮膚・自己交差等の形状検査に不合格')
        reason = reason.replace('initial_geometry_intersection', '初期配置：交差検査に不合格')
        reason = reason.replace('not_attempted', '未試行')
        reason = reason.replace('in_progress', '試行中')
        if row['reason'].startswith('native_fit:') and row['cuff_motion_mm'] is not None:
            reason += f'（直近1秒の変動 {row["cuff_motion_mm"]:.2f}mm、速度 {row["cuff_speed_m_s"]:.3f}m/s）'
        table.append('<tr>' + ''.join(f'<td>{html.escape(str(row[k]))}</td>'
            for k in ('number', 'stage', 'pitch_deg', 'roll_deg')) +
            f'<td class="{"pass" if row["passed"] else "fail"}">{"合格" if row["passed"] else "試行中" if row["reason"] == "in_progress" else "未採用"}</td>' +
            f'<td>{html.escape(reason)}</td><td>{link(row["state"], "形状")} '
            f'{link(str(image), "Unity画像") if image.exists() else ""} '
            f'{link(row["fit_report"], "物理検査")} {link(row["log"], "ログ")}</td></tr>')
    count_text = ' / '.join(f'{s}: {n}' for s, n in counts.items())
    document = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>C〜E 初期状態の追加結果</title>
<style>body{{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border:1px solid #ccc;text-align:left}}.pass{{color:#087f4c}}.fail{{color:#a43d1b}}</style>
<h1>C〜E 初期状態の追加結果</h1><p>{count_text}、合計 {report['total_unique']} 種類（従来45種類から {report['added_unique']} 種類追加）。</p>
<p>Excelの指定開口位置 × pitch・roll: {len(unique_passed)}/{len(unique_requested)} 種類合格。DのNo.2/11、No.30/42はそれぞれ同じ姿勢。既存のC/No.10・E/No.10も合計に含む。</p>
<p>外周フィットと10秒の重力保持、支持部と下垂部の準備、皮膚断面への調整、開口帯の膨らみ、最終交差検査を記録。最終形状はリーチング用の凍結形状。調整後の支持なし保持・把持・牽引は未検証。</p>
<p>Excelの素材差・前後ずらし・斜め回転・しわ・折り返しは適用していません。</p>
<p><a href="expansion-report.csv">Excelで開けるCSV</a> / <a href="expansion-report.json">検査結果JSON</a></p>
<p><a href="expansion-overview.jpg"><img src="expansion-overview.jpg" alt="合格姿勢の側面・足の甲側のUnity比較画像" style="width:100%"></a></p>
<table><tr><th>No.</th><th>開口位置</th><th>pitch°</th><th>roll°</th><th>結果</th><th>未採用理由</th><th>保存先</th></tr>{''.join(table)}</table></html>'''
    (root / 'expansion-report.html').write_text(document)
    print(count_text, f'total={report["total_unique"]}, added={report["added_unique"]}', flush=True)
    print(f'Workbook: {len(unique_passed)}/{len(unique_requested)}; {root / "expansion-report.html"}', flush=True)


if __name__ == '__main__':
    main()
