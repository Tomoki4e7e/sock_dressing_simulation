#!/usr/bin/env python3
"""Evaluate saved checkpoints and a selected immutable split in Unity/CUDA."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def aggregate(out, inventory, cases, started):
    split = cases[0]['split']
    label = {'train': '教示データ（学習に使用）', 'validation': '検証データ', 'test': 'テストデータ'}[split]
    rows, checkpoints, cards = [], [], []
    for entry in inventory:
        name = entry['name']
        checkpoint_rows = []
        for case in cases:
            base = Path(name) / case['id'] / 'policy'
            path = out / base / 'report.json'
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            if not report.get('completed') or not report.get('video_validation'):
                continue
            assert report['checkpoint_sha256'] == entry['sha256']
            assert report['device']['device'] == 'cuda'
            assert all(d.startswith('cuda') for d in report['device']['model_devices'].values())
            assert report['steps'] == (case['frames'] + 1) // 2 - 1
            assert report['actual_physics_steps'] == 10 * report['steps']
            assert len(report['video_validation']) == 2
            row = dict(checkpoint=name, epoch=report['device']['epoch'], case=case['id'], split=split,
                       completed=True, steps=report['steps'],
                       duration_s=report['simulated_duration_s'],
                       checkpoint_sha256=entry['sha256'],
                       native_penetration_events=report['native_penetration_event_count'],
                       overview_video=str(base / 'overview.mp4'),
                       head_video=str(base / 'head-input.mp4'),
                       measurement_json=str(base / 'terminal-cloth-between-fingers.json'))
            for side in ('left', 'right'):
                geometry = report['geometry'][side]
                value = geometry['cloth_surface_area_between_fingers_mm2']
                assert math.isfinite(value) and value >= 0
                row[side + '_cloth_area_mm2'] = value
                row[side + '_material_15_30mm_band_area_mm2'] = geometry['material_15_30mm_band_area_between_fingers_mm2']
                row[side + '_opening_rim_mm'] = geometry['opening_rim_length_between_fingers_mm']
                row[side + '_cloth_present'] = geometry['cloth_present']
                row[side + '_goal_geometry_passed'] = geometry['contact_geometry_checks']['passed']
            rows.append(row)
            checkpoint_rows.append(row)
        checkpoints.append(dict(**entry, finished=len(checkpoint_rows), expected=len(cases),
                                epoch=checkpoint_rows[0]['epoch'] if checkpoint_rows else None,
                                both_cloth_present=sum(r['left_cloth_present'] and r['right_cloth_present'] for r in checkpoint_rows),
                                both_goal_geometry_passed=sum(r['left_goal_geometry_passed'] and r['right_goal_geometry_passed'] for r in checkpoint_rows)))
        table = ''.join(
            '<tr><td>{case}</td><td>{left_cloth_area_mm2:.3f}</td><td>{right_cloth_area_mm2:.3f}</td>'
            '<td><a href="{overview_video}">全景動画</a> · <a href="{head_video}">入力動画</a> · '
            '<a href="{measurement_json}">測定JSON</a></td></tr>'.format(**r)
            for r in checkpoint_rows)
        cards.append(f'<section><h2>{html.escape(name)} — epoch {checkpoints[-1]["epoch"]}</h2>'
                     f'<p>{len(checkpoint_rows)} / {len(cases)} ケース完了 · '
                     f'<a href="{name}/report.html">動画・爪の接写を見る</a></p>'
                     '<table><thead><tr><th>ケース</th><th>左の面積 mm²</th><th>右の面積 mm²</th><th>記録</th></tr></thead>'
                     f'<tbody>{table}</tbody></table></section>')
    expected = len(cases) * len(inventory)
    summary = dict(completed=len(rows) == expected, expected=expected, finished=len(rows),
                   checkpoint_count=len(inventory), dataset_split=split, case_ids=[c['id'] for c in cases],
                   video_count=len(rows) * 2, elapsed_s=time.time() - started,
                   device='cuda', policy_precision='float32', flow_seed_per_case=0, ode_steps=8,
                   action_selection='Execute first slot, replan every 0.2 simulated seconds',
                   measurement='Terminal actual cloth triangle surface clipped to finite measured opposing jaws, mm^2',
                   scope='Frozen cloth/human; geometric jaw occupancy, not closure/retention/pulling success',
                   checkpoints=checkpoints, rows=rows)
    write(out / 'report.json', summary)
    write(out / 'progress.json', {k: v for k, v in summary.items() if k != 'rows'})
    if rows:
        with (out / 'terminal-measurements.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    page = ('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Flow GPU推論</title><style>'
            'body{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:30px auto;padding:0 20px;background:#edf2f7;color:#172435}'
            'section{background:white;padding:20px;margin:24px 0;border-radius:12px}table{border-collapse:collapse;width:100%}'
            'th,td{border-bottom:1px solid #d9e1ec;padding:8px;text-align:right}td:first-child{text-align:left}a{color:#086399}'
            '</style><h1>Flow：GPU推論と爪間の靴下面積</h1>'
            f'<p>{len(inventory)} チェックポイント × {len(cases)} ケース（{label}）。{len(rows)} / {expected} 試行完了。</p>'
            '<p>Unityの実測RGB・関節角・駆動力からFlow・SAM2・Depth AnythingをCUDAで実行。'
            '8ステップのチャンクの先頭を実行し、0.2秒ごとに観測から再計画します。全チェックポイントで同じ乱数シード0を使用。</p>'
            '<p>人体と靴下は教示と同じ固定条件です。表は動作終端の実測メッシュ・有限な爪形状から測定した布表面積です。'
            '爪間に布が存在することと、閉鎖・摩擦保持・引き上げの成功は別です。'
            '駆動力の入力は因果的な平滑化で、学習時の前後サンプルによる平滑化との差があります。</p>'
            '<p><a href="terminal-measurements.csv">全試行の測定CSV</a> · <a href="report.json">全結果JSON</a></p>'
            + ''.join(cards) + '</html>')
    (out / 'report.html').write_text(page)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--port-base', type=int, default=46700)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--ids', nargs='+')
    parser.add_argument('--checkpoints', nargs='+', help='Checkpoint stems, e.g. SAMDAMSARNNFlow_best')
    args = parser.parse_args()
    run, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cases = [r for r in json.loads((run / 'manifest.json').read_text())['rows']
             if r['split'] == args.split and (not args.ids or r['id'] in args.ids)]
    if args.ids:
        assert len(set(args.ids)) == len(args.ids) and {r['id'] for r in cases} == set(args.ids)
    if args.split == 'train':
        # Each rollout has its own RNG/state. Start longer episodes first.
        cases.sort(key=lambda r: (-r['frames'], r['id']))
    paths = sorted((run / 'model').glob('*.pth'))
    if args.checkpoints:
        paths = [p for p in paths if p.stem in args.checkpoints]
        assert len(set(args.checkpoints)) == len(args.checkpoints) and {p.stem for p in paths} == set(args.checkpoints)
    assert cases and paths
    selection = dict(split=args.split, ids=[r['id'] for r in cases])
    selection_path = out / 'selection.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == selection, 'Selected cases changed'
    else:
        write(selection_path, selection)
    inventory = [dict(name=p.stem, path=str(p), bytes=p.stat().st_size, sha256=digest(p)) for p in paths]
    previous = out / 'checkpoint-inventory.json'
    if previous.exists():
        assert json.loads(previous.read_text()) == inventory, 'Checkpoint inventory changed'
    else:
        write(previous, inventory)
    programs = out / 'programs'
    programs.mkdir(exist_ok=True)
    for name in ('evaluate_all_pregrasp_checkpoints.py', 'contact_pregrasp_policy_worker.py',
                 'contact_pregrasp_policy_scene.py', 'run_contact_pregrasp_policy_tests.py',
                 'report_contact_pregrasp_policy_tests.py', 'photograph_grasp_candidates.py',
                 'capture_fixed_torso_initial.py', 'native_pregrasp_geometry.py'):
        source = ROOT / 'scripts' / name
        target = programs / name
        if target.exists():
            assert digest(target) == digest(source), 'Evaluation program changed during a resumed run'
        else:
            shutil.copy2(source, target)
    for name in ('configuration.json', 'normalization.json', 'manifest.json'):
        target = out / name
        if target.exists():
            assert digest(target) == digest(run / name)
        else:
            shutil.copy2(run / name, target)
    for case in cases:
        for name, expected in case['source_sha256'].items():
            assert digest(Path(case['path']) / name) == expected, 'Test recording changed'
    model_root = ROOT.parent / 'ShareSet/share_folder/src/models'
    for name in ('SAMDAMSARNN.py', 'SAMDAMSARNN_Flow.py', 'HierarchicalRNNCell.py', 'pretrained_action_flow.py'):
        assert digest(model_root / name) == digest(run / 'sources' / name), 'Model differs from training snapshot: ' + name
    write(out / 'evaluation-settings.json', dict(run=str(run), output=str(out), workers=args.workers,
          flow_seed=0, ode_steps=8, policy_precision='float32', checkpoint_count=len(paths), case_count=len(cases),
          dataset_split=args.split, selected_case_ids=selection['ids'],
          command_period_s=0.2, physics_step_s=0.02, physics_steps_per_exchange=1,
          checkpoint_inventory_sha256=digest(previous), started_at=time.time()))
    aggregate(out, inventory, cases, started)
    from run_contact_pregrasp_policy_tests import run_case
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {}
        for index, (entry, case) in enumerate((e, c) for e in inventory for c in cases):
            job = pool.submit(run_case, case, run, out / entry['name'], index,
                              Path(entry['path']), entry['sha256'], args.port_base)
            jobs[job] = (entry, case)
        for job in as_completed(jobs):
            entry, case = jobs[job]
            try:
                result = job.result()
                if not result.get('completed'):
                    failures.append(dict(checkpoint=entry['name'], case=case['id'], failure=result.get('failure')))
            except Exception as error:
                failures.append(dict(checkpoint=entry['name'], case=case['id'], failure=repr(error)))
            current = aggregate(out, inventory, cases, started)
            print('CASE_FINISH', entry['name'], case['id'], current['finished'], '/', current['expected'], flush=True)
    for entry in inventory:
        dest = out / entry['name']
        command = [sys.executable, '-u', str(ROOT / 'scripts/run_contact_pregrasp_policy_tests.py'),
                   '--run', str(run), '--output', str(dest), '--checkpoint', entry['path'],
                   '--workers', str(args.workers), '--port-base', str(args.port_base),
                   '--split', args.split, '--ids', *selection['ids']]
        # Successful cases are reused without launching Unity again. Avoid
        # silently retrying failed cases while finalizing the batch.
        if not any(f['checkpoint'] == entry['name'] for f in failures):
            with (dest / 'suite.log').open('a') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, str(ROOT / 'scripts/report_contact_pregrasp_policy_tests.py'),
                            '--run', str(run), '--output', str(dest)], cwd=ROOT, check=True)
    assert all(digest(Path(entry['path'])) == entry['sha256'] for entry in inventory), 'Weights changed during inference'
    summary = aggregate(out, inventory, cases, started)
    write(out / 'execution-status.json', dict(completed=summary['completed'], failures=failures))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('rows', 'checkpoints')}, indent=2), flush=True)
    if not summary['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
