#!/usr/bin/env python3
"""Replay unchanged commands after pin repair; never resume material searches."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTER = ROOT / 'artifacts/phase4/foot-conforming-fixed-motion'


def validate_player(player: Path) -> str:
    """Reject incomplete or modified builds before starting a recorder."""
    if not player.is_file() or not os.access(player, os.X_OK):
        raise RuntimeError(f'Player is missing or not executable: {player}')
    archive = player.parent / 'runtime_sources'
    manifest_path = archive / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError(f'Player has no completed build manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    for name in ('RCareWorld.dll', 'Obi.dll'):
        compiled = player.parent / 'Player_Data/Managed' / name
        if not compiled.is_file() or hashlib.sha256(compiled.read_bytes()).hexdigest() != manifest.get('compiled_' + name):
            raise RuntimeError(f'Player compiled assembly differs from its manifest: {name}')
    for path, expected in manifest.items():
        if path.startswith('compiled_'):
            continue
        source = archive / Path(path).name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Player archived source differs from its manifest: {source}')
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def evaluate(metadata: Path, steps: int) -> dict:
    assess = runpy.run_path(str(ROOT / 'scripts/tune_foot_conforming.py'))['assess']
    result = assess(metadata, steps)
    data = json.loads(metadata.read_text())
    violations = []
    for frame, quality in enumerate(data['cloth_quality_by_frame']):
        for name, edge in quality['stretch']['edge_classes'].items():
            stretch = edge.get('maximum_stretch')
            if stretch is None and edge.get('edge_count') == 0:
                continue
            # Pin membership does not define a material region. A pin/body
            # edge can also be an opening/body edge, checked separately below.
            limit = 2.0 if name == 'opening_rim' else 1.1 if name == 'opening_body' else 1.5
            if stretch is None or not math.isfinite(stretch) or stretch > limit * 1.01:
                violations.append({'frame': frame, 'edge_class': name, 'stretch': stretch, 'limit': limit})
    safety = [row['foot_collision_safety'] for row in data['dressing_quality_by_frame']]
    result['all_frame_stretch_violations'] = violations
    result['checks']['all_frames_within_stretch_limits'] = not violations
    result['checks']['fixed_original_offsets'] = all(s['maximum_grasp_offset_drift_m'] == 0 for s in safety)
    result['checks']['coupled_zero_compliance_pins'] = all(s.get('coupled_zero_compliance_pins') is True for s in safety)
    result['checks']['coupled_pin_projection_order'] = all(
        s.get('coupled_pin_projection_order') == 'pins_then_material_then_face_contact' for s in safety)
    result['checks']['coupled_cuff_face_projection'] = all(
        s.get('coupled_cuff_face_projection') is True for s in safety)
    result['repair_validation_passed'] = all(result['checks'].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--player', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTER / 'controlled-v32')
    parser.add_argument('--port', type=int, default=7704)
    parser.add_argument('--prepared-state', type=Path, default=OUTER / 'controlled-v31/common_preparation.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output / 'validation-status.json'
    def status(stage: str, **details) -> None:
        value = {'stage': stage, 'observed_at_utc': datetime.now(timezone.utc).isoformat(),
                 'material_search_resumed': False, **details}
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(status_path)
    try:
        player_identity = validate_player(args.player)
        for steps in (40, 100):
            if validate_player(args.player) != player_identity:
                raise RuntimeError('Player changed between the fixed replays')
            name = f'coupled_pins_fixed{steps}'
            directory = args.output / name
            if directory.exists():
                raise RuntimeError(f'Recording already exists; use a new output directory: {directory}')
            directory.mkdir()
            command = [sys.executable, '-u', str(ROOT / 'scripts/replay_foot_contact.py'),
                       '--output-root', str(args.output), '--candidate-name', name, '--steps', str(steps),
                       '--substeps', '8', '--iterations', '20', '--surface-mode', 'local',
                       '--foot-shape', 'skin-hull', '--transport', 'local', '--fixed-grasp-offsets',
                       '--post-preparation-material', '--prepared-state', str(args.prepared_state.resolve()),
                       '--player-executable', str(args.player.resolve()), '--port', str(args.port)]
            started = time.monotonic()
            with (directory / 'controller.log').open('w') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                previous_frames = -1
                while child.poll() is None:
                    diagnostic = directory / 'live_frame_diagnostics.jsonl'
                    frames = sum(1 for _ in diagnostic.open()) if diagnostic.exists() else 0
                    status('replaying', required_steps=steps, recorded_frames=frames, pid=child.pid,
                           elapsed_s=time.monotonic() - started)
                    if frames != previous_frames:
                        print(f'{name}: {frames}/{steps} frames', flush=True); previous_frames = frames
                    time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'Replay failed ({child.returncode}): {directory / "controller.log"}')
            recording = json.loads((directory / 'result.json').read_text())
            metadata = Path(recording['metadata'])
            assessment = evaluate(metadata, steps)
            assessment['wall_time_s'] = time.monotonic() - started
            assessment['prepared_state_source'] = str(args.prepared_state.resolve())
            assessment['prepared_state_sha256'] = hashlib.sha256(args.prepared_state.read_bytes()).hexdigest()
            material = json.loads((directory / 'applied_material.json').read_text())
            baseline = json.loads((OUTER / 'controlled-v31/local_transport_26ba670fdd43/applied_material.json').read_text())
            assessment['checks']['unchanged_requested_material'] = material['requested'] == baseline['requested']
            assessment['repair_validation_passed'] = all(assessment['checks'].values())
            (directory / 'repair-validation.json').write_text(json.dumps(assessment, indent=2) + '\n')
            print(f'{name}: passed={assessment["repair_validation_passed"]}', flush=True)
            if not assessment['repair_validation_passed']:
                status('validation_failed', steps=steps, assessment=str(directory / 'repair-validation.json'))
                raise SystemExit(1)
        status('validated_40_and_100_commands', visual_review_required=True)
    except Exception as error:
        status('error', error=str(error)); raise


if __name__ == '__main__':
    main()
