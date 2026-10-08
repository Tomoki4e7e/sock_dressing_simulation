#!/usr/bin/env python3
"""Finish existing controls while recording independent fixed-motion materials."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONTROL_NAMES = ('control_legacy_94d8c74dba66', 'local_surface_0db51a32378e',
                 'fixed_grasp_f727dd886d49', 'local_transport_26ba670fdd43')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-root', type=Path, required=True)
    parser.add_argument('--materials-root', type=Path, required=True)
    parser.add_argument('--player', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--control-port', type=int, default=6400)
    parser.add_argument('--material-port', type=int, default=6440)
    parser.add_argument('--resume-existing-materials', action='store_true',
                        help='Keep active recordings; the replay locks and full identity checks handle their reuse')
    args = parser.parse_args()
    materials = args.materials_root
    materials.mkdir(parents=True, exist_ok=True)
    # Reuse exactly the prepared state and compiled kernel identity. Candidates
    # remain independent until all completed outcomes are compared.
    for name in ('common_preparation.json', 'common_preparation_kernel.json'):
        source = args.controls_root / name
        target = materials / name
        if target.exists():
            if source.read_bytes() != target.read_bytes():
                raise RuntimeError('Preparation mismatch; do not combine comparisons')
        else:
            shutil.copy2(source, target)
    progress = {'controls_root': str(args.controls_root), 'materials_root': str(materials),
                'player': str(args.player), 'best_published': False}

    def report(**values):
        progress.update(values, observed_at_utc=datetime.now(timezone.utc).isoformat())
        args.status.write_text(json.dumps(progress, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(values, ensure_ascii=False), flush=True)

    common = [sys.executable, '-u', str(ROOT / 'scripts/tune_foot_conforming.py'),
              '--workers', '4', '--foot-shape', 'skin-hull', '--player-executable', str(args.player)]
    log_path = materials / ('resumed-scheduler.log' if args.resume_existing_materials else 'scheduler.log')
    if log_path.exists():
        raise RuntimeError('Material scheduler log already exists; inspect its state before restarting')
    with log_path.open('w') as material_log:
        process = subprocess.Popen(common + ['--phase', 'materials-final', '--output-root', str(materials),
                                   '--port', str(args.material_port)], cwd=ROOT,
                                   stdout=material_log, stderr=subprocess.STDOUT)
        report(stage='recording_controls_and_materials', material_scheduler_pid=process.pid)
        try:
            while not all((args.controls_root / name / 'result.json').exists() for name in CONTROL_NAMES):
                complete = [name for name in CONTROL_NAMES if (args.controls_root / name / 'result.json').exists()]
                report(completed_existing_controls=complete, material_scheduler_returncode=process.poll())
                time.sleep(20)
            # The cached replay emits a new stdout log; preserve the complete
            # recording's original stdout before asking the control evaluator
            # to read cached outcomes and run the remaining box comparison.
            for name in CONTROL_NAMES:
                source = args.controls_root / name / 'replay.log'
                target = source.with_name('recording_replay_stdout.log')
                if target.exists():
                    raise RuntimeError('Control stdout archive already exists; inspect before rerunning')
                shutil.copy2(source, target)
                target.with_suffix('.log.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest() + '\n')
            report(stage='completing_control_assessments_and_box_comparison')
            with (args.controls_root / 'resumed-controls-scheduler.log').open('w') as log:
                control = subprocess.run(common + ['--phase', 'controls', '--output-root', str(args.controls_root),
                                         '--port', str(args.control_port)], cwd=ROOT,
                                         stdout=log, stderr=subprocess.STDOUT)
            report(control_scheduler_returncode=control.returncode,
                   stage='waiting_for_material_selection_and_final_grid')
            while process.poll() is None:
                time.sleep(20)
            report(material_scheduler_returncode=process.returncode,
                   stage='recording_finished_requires_assessment_and_visual_review')
            if control.returncode or process.returncode:
                report(stage='comparison_requires_followup', best_published=False)
                raise RuntimeError('A comparison failed or yielded no qualifying material; inspect recorded diagnostics')
        except Exception as exception:
            report(supervisor_error=str(exception), best_published=False)
            # A supervision error does not cancel an independent recording.
            raise


if __name__ == '__main__':
    main()
