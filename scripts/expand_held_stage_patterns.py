#!/usr/bin/env python3
"""Try every unique workbook C--E posture, retaining failures and passed shapes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sock_dressing_simulation.held_stage_shapes import DEFAULT_ROOT, STAGES, load_stage_state
from sock_dressing_simulation.pose_patterns import read_patterns


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / DEFAULT_ROOT)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--port', type=int, default=18600)
    parser.add_argument('--numbers', type=int, nargs='+')
    parser.add_argument('--collect-only', action='store_true',
                        help='Reconcile saved results without starting Players or replacing trial logs')
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error('--workers must be between 1 and 12')
    workbook = ROOT / 'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx'
    patterns = read_patterns(workbook)
    if args.numbers and set(args.numbers) - {p.number for p in patterns}:
        parser.error('Unknown workbook number')
    groups = {}
    for pattern in patterns:
        if pattern.stage in STAGES:
            key = (pattern.stage, pattern.pitch_deg, pattern.roll_deg)
            groups.setdefault(key, []).append(pattern)
    trials = [rows[0] for rows in groups.values()
              if args.numbers is None or any(p.number in args.numbers for p in rows)]
    if not trials:
        parser.error('No workbook C--E conditions selected')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    initial = sorted(str(p.relative_to(args.output)) for p in args.output.glob('pose-*/state.json'))
    rows = []
    report = dict(workbook=str(workbook), workbook_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),
                  scope='Workbook opening stage and pitch/roll only; material and local deformations are not applied',
                  preexisting_states=initial, requested=[asdict(p) for p in trials], rows=rows,
                  complete=False, all_passed=False)
    report_path = args.output / 'workbook-expansion.json'
    if report_path.exists():
        initial = json.loads(report_path.read_text()).get('preexisting_states', initial)
        report['preexisting_states'] = initial
    report['collection_only'] = args.collect_only
    write(report_path, report)

    def run(index, pattern):
        directory = args.output / f'pose-{pattern.number:02d}-{pattern.stage}'
        directory.mkdir(exist_ok=True)
        log = directory / 'expansion.log'
        command = [sys.executable, '-u', str(ROOT / 'scripts/prepare_held_stage_shapes.py'),
                   '--numbers', str(pattern.number), '--stages', pattern.stage,
                   '--output', str(args.output), '--port', str(args.port + 20 * index)]
        print(f'START {pattern.number:02d}-{pattern.stage} pitch={pattern.pitch_deg:g} roll={pattern.roll_deg:g}', flush=True)
        code = 0
        if not args.collect_only:
            with log.open('w') as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                code = result.returncode
        row = dict(number=pattern.number, stage=pattern.stage, pitch_deg=pattern.pitch_deg,
                   roll_deg=pattern.roll_deg, workbook_numbers=[p.number for p in groups[
                       (pattern.stage, pattern.pitch_deg, pattern.roll_deg)]],
                   exit_code=None if args.collect_only else code, log=str(log), passed=False)
        try:
            state = load_stage_state(args.output, pattern, pattern.stage)
            row['passed'] = bool(code == 0 and state['geometry_audit']['geometry_ok']
                                 and state['surface_audit']['surface_ok'])
            row['state'] = str(directory / 'state.json')
        except (OSError, ValueError, KeyError) as error:
            row['error'] = str(error)
        if not row['passed']:
            row['log_tail'] = log.read_text()[-6000:] if log.exists() else ''
        print(f'{"PASS" if row["passed"] else "FAIL"} {pattern.number:02d}-{pattern.stage}', flush=True)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        pending = [executor.submit(run, i, p) for i, p in enumerate(trials)]
        for future in as_completed(pending):
            rows.append(future.result())
            rows.sort(key=lambda r: (r['stage'], r['number']))
            write(report_path, report)
    report['complete'] = True
    report['all_passed'] = all(r['passed'] for r in rows)
    write(report_path, report)
    print(f'{sum(r["passed"] for r in rows)}/{len(rows)} unique workbook conditions passed: {report_path}', flush=True)


if __name__ == '__main__':
    main()
