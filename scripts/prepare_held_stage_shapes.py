#!/usr/bin/env python3
"""Prepare C--E with A's physical circumference fit, toe gravity, and cuff bulge."""
from __future__ import annotations

import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import copy
import hashlib
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.deps/stage-python'))
sys.path.insert(0, str(ROOT / '.deps/RCareWorld/pyrcareworld'))

from sock_dressing_simulation.cli import _prepare
from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.environment import SockDressingEnv
from sock_dressing_simulation.scenario import scenario_from_config
from sock_dressing_simulation.pose_patterns import read_patterns
from sock_dressing_simulation.circumference_fit import run_fit
from sock_dressing_simulation.held_ankle_poses import initialize_held_ankle_pattern
from sock_dressing_simulation.held_stage_shapes import (
    DEFAULT_ROOT, BASELINE_ROOT, PROFILE, STAGES, state_directory, build_bulged_stage_state, load_stage_state,
    tighten_supported_surface,
    qualify_native_stage_fit,
    audit_actual_skin,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(root, trials, failure=None):
    rows = []
    for path in sorted(root.glob('pose-*-*/state.json')):
        state = json.loads(path.read_text())
        if state.get('shape_profile') != PROFILE:
            continue
        rows.append(dict(number=state['pattern']['number'], stage=state['applied_opening_stage'],
                         passed=True, state=str(path), preparation=state['preparation']))
    completed = {(r['number'], r['stage']) for r in rows}
    requested = {(p.number, s) for p, s in trials}
    write(root / 'summary.json', dict(shape_profile=PROFILE, rows=rows,
          all_passed=failure is None and requested <= completed,
          requested=[dict(number=n,stage=s) for n,s in sorted(requested)], failure=failure))


def local_runtime_locks(directory):
    """Keep the dependency's process locks in the output, not ~/.rfuniverse."""
    path = ROOT / '.deps/RCareWorld/pyrcareworld/pyrcareworld/utils/locker.py'
    spec = importlib.util.spec_from_file_location('pyrcareworld.utils.locker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory.mkdir(parents=True, exist_ok=True)
    def initialize(self, name):
        self.lck_path = str(directory / (name + '.lck'))
        Path(self.lck_path).touch(exist_ok=True)
        self.fp = None
    module.Locker.__init__ = initialize
    sys.modules['pyrcareworld.utils.locker'] = module


def verify_stage_player():
    manifest_path = ROOT/'Build/SockDressingPlayer/HeldStagesV1/runtime_sources/manifest.json'
    source_path = ROOT/'unity/RuntimeExtensions/SockHeldStageGravity.cs'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    contact_source = manifest_path.parent/'SockCircumferenceFit.cs'
    if (manifest.get(str(source_path.relative_to(ROOT))) != hashlib.sha256(source_path.read_bytes()).hexdigest()
            or not contact_source.exists() or 'HeldStageEffectiveGravity(p)/inverseMass' not in contact_source.read_text()):
        raise RuntimeError('Build the current stage Player first: bash scripts/build_held_stage_player.sh')


def initialize_stage_preinsert(env, stage, directory):
    """Check the actual skin if a conservative hull rejects the initial surface.

    As in the final native-fit gate, hull diagnostics remain recorded. Initial
    particle/edge containment, actual skin crossings, and cloth self-crossings
    must all pass before any physical fitting starts.
    """
    from sock_dressing_simulation.preinsert import generate_preinsert, audit_preinsert, audit_preinsert_surface
    try:
        return env.initialize_preinsert(stage)
    except RuntimeError as error:
        if not str(error).startswith('Preinsert geometry intersects'):
            raise
        attr = env.sock_cloth
        attr.request_preinsert_source()
        attr._send_data('GetSockPatternSkin')
        env._env.step(simulate=False)
        source = attr.data['sock_preinsert_source']
        edges = attr.data['particle_edges']
        faces = attr.data['particle_surface_triangles']
        write(directory / 'initial-placement-source.json', dict(source=source,
            skin_surface=attr.data['sock_pattern_skin'], edges=edges, triangles=faces,
            rest_edge_lengths=attr.data['particle_rest_edge_lengths']))
        trials = []
        for clearance in (.015, .0175, .020, .025, .030):
            # Only the loose starting envelope changes. Native fitting still
            # contracts the same material and must pass the unchanged gates.
            points, report = generate_preinsert(source, stage, clearance_m=clearance)
            conservative = audit_preinsert(points, source, edges, attr.data['particle_rest_edge_lengths'])
            conservative_surface = audit_preinsert_surface(points, source, edges, faces)
            body, surface = audit_actual_skin(points, attr.data['sock_pattern_skin'], edges, faces)
            report.update(conservative_convex_geometry_audit=conservative,
                          conservative_convex_surface_audit=conservative_surface,
                          geometry_audit=body, surface_audit=surface)
            report['passed'] = bool(conservative['geometry_ok'] and body['geometry_ok'] and surface['surface_ok'])
            trials.append(copy.deepcopy(report))
            if report['passed']:
                break
        report['initial_clearance_trials'] = trials
        write(directory / 'initial-placement-audit.json', report)
        if not report['passed']:
            raise RuntimeError(f'Initial stage surface fails containment, actual skin or self-intersection gates: {directory}') from error
        attr.release('left')
        attr.release('right')
        attr.initialize_preinsert(points, report['opening_inward_normal_world'])
        attr.request_particles()
        env._env.step(simulate=False)
        native = attr.data.get('sock_preinsert_report', {})
        observed = attr.particles()
        if not native.get('ok') or observed.shape != points.shape or not np.allclose(observed, points, atol=5e-6, rtol=0):
            raise RuntimeError('Native stage preinsert initialization readback differs')
        report['native'] = native
        report['maximum_readback_error_m'] = float(np.abs(observed-points).max())
        env._preinsert_observation_pending = True
        return report


def prepare_phase(args, pattern, stage, directory):
    verify_stage_player()
    config = load_config(ROOT / 'config/pose_patterns.yaml')
    settings = config.pop('pose_pattern')
    settings['toe_contact'] = None
    settings['gravity_drape'] = dict(enabled=True, settle_steps=args.settle_steps, gravity_scale=args.drape_gravity_scale)
    os.environ['SOCK_STAGE_GRAVITY'] = '1'
    config['rcareworld'].update(graphics=False, port=args.port, log_level=0,
                              executable=str(ROOT / 'scripts/run_stage_preparation_player.sh'))
    os.environ['SOCK_STAGE_PLAYER_LOG'] = str(directory / (args.phase + '-player.log'))
    config['assets']['output_dir'] = str(directory / (args.phase + '-assets'))
    if args.phase == 'fit':
        config['preinsert'] = dict(stage=stage, above_heel_m=.05, clearance_m=.015)
    write(directory / (args.phase+'-runtime-config.json'), config)
    local_runtime_locks(directory / 'runtime-locks')
    scenario = scenario_from_config(config, seed=0)
    prepared = _prepare(config, scenario)
    reference = directory / 'fit'
    with SockDressingEnv(config) as env:
        env.load(Path(prepared['runtime_urdf']), Path(config['assets']['output_dir']) / 'sock.obj',
                 initial_joints=scenario.initial_joints)
        env.apply_scenario(scenario)
        attr = env.sock_cloth
        if args.phase == 'fit':
            # Fit at this actual ankle angle. Rotating a saved A or C--E shape
            # as a rigid whole would also rotate its shin and cut through skin.
            attr.set_pattern_foot_pose(pattern.pitch_deg, pattern.roll_deg)
            env._env.step(simulate=False)
            placement = initialize_stage_preinsert(env, stage, directory)
            attr.request_preinsert_source()
            attr._send_data('GetSockPatternSkin')
            env._env.step(simulate=False)
            source = copy.deepcopy(attr.data['sock_preinsert_source'])
            source['skin_surface'] = copy.deepcopy(attr.data['sock_pattern_skin'])
            attr.request_prepared_state()
            env._env.step(simulate=False)
            reference.mkdir(parents=True, exist_ok=True)
            (reference/'initial-prepared-state.json').write_text(attr.data['prepared_sock_state']+'\n')
            hold_samples = 0
            def progress(row):
                nonlocal hold_samples
                if row['phase'] == 'contract':
                    print(f'{pattern.number:02d}-{stage} circumference={row["rest_circumference_scale"]:.2f}', flush=True)
                elif row['phase'] == 'gravity_hold':
                    hold_samples += 1
                    if hold_samples % 10 == 0:
                        print(f'{pattern.number:02d}-{stage} gravity hold: {hold_samples*5} steps', flush=True)
            fit_settings = load_config(ROOT / 'config/preinsert_fitted.yaml')['circumference_fit']
            fit_settings['target_cuff_force_n'] = args.target_cuff_force if args.target_cuff_force is not None else .08
            fit_settings['stage_contact'] = True
            fit_settings['settle_steps'] = args.fit_settle_steps or 5
            report = run_fit(env, **fit_settings, on_sample=progress)
            report.update(pattern=asdict(pattern), initial_placement=placement, fit_settings=fit_settings,
                          contact_neighbour_model='A_material_coordinates_v1')
            attr.request_prepared_state()
            attr.request_particles()
            env._env.step(simulate=False)
            report = qualify_native_stage_fit(report, source, attr.particles(),
                attr.data['particle_edges'], attr.data['particle_surface_triangles'])
            write(reference / 'report.json', report)
            write(reference / 'foot-collision-hull.json', report['foot_hull'])
            write(reference / 'fitted-geometry.json', dict(source=source,
                  particles_world=attr.particles().tolist(), triangles=attr.data['particle_surface_triangles'],
                  edges=attr.data['particle_edges'], rest_edge_lengths=attr.data['particle_rest_edge_lengths']))
            (reference / 'fitted-prepared-state.json').write_text(attr.data['prepared_sock_state'] + '\n')
            if not report['passed']:
                raise RuntimeError(f'{stage} native circumference fit failed; see {reference / "report.json"}')
        else:
            try:
                report = initialize_held_ankle_pattern(env, pattern, str(reference),
                    signs=settings['signs'], gravity_drape=settings['gravity_drape'], reference_stage=stage)
            except Exception:
                diagnostic = getattr(env, '_gravity_drape_debug', None)
                if diagnostic:
                    write(directory / 'settling-diagnostic.json', diagnostic)
                raise
            state = dict(pattern=asdict(pattern),
                         settings=dict(generator='prepare_held_stage_shapes', held_reference=str(reference),
                                       reference_stage=stage, gravity_drape=settings['gravity_drape']),
                         source=env._pose_pattern_source,
                         particles_world=attr.particles().tolist(),
                         opening_normal=report['opening_inward_normal_world'],
                         triangles=attr.data['particle_surface_triangles'], edges=attr.data['particle_edges'])
            write(directory / 'settling-report.json', report)
            write(directory / 'settled-state.json', state)
            state = tighten_supported_surface(state, strength=args.fit_strength, clearance_m=args.skin_clearance_m)
            write(directory / 'surface-fit-report.json', state['surface_fit'])
            fit = json.loads((reference / 'report.json').read_text())
            state, before, profile = build_bulged_stage_state(state, fit, report, stage)
            manifest = ROOT/'Build/SockDressingPlayer/HeldStagesV1/runtime_sources/manifest.json'
            state['preparation']['runtime_sources_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            patterns = read_patterns(ROOT / 'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
            state['scene_number'] = next(p.number for p in patterns
                if (p.pitch_deg, p.roll_deg) == (pattern.pitch_deg, pattern.roll_deg))
            write(directory / 'before-bulge.json', dict(particles_world=before.tolist()))
            write(directory / 'bulge-profile.json', profile)
            write(directory / 'state.json', state)
            print(f'{pattern.number:02d}-{stage}: A-method shape PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--numbers', type=int, nargs='+')
    parser.add_argument('--stages', choices=STAGES, nargs='+', help='Default: each row\'s workbook stage')
    parser.add_argument('--output', type=Path, default=ROOT / DEFAULT_ROOT)
    parser.add_argument('--port', type=int, default=18300)
    parser.add_argument('--fit-strength', type=float, default=.75)
    parser.add_argument('--skin-clearance-m', type=float, default=.0025)
    parser.add_argument('--drape-gravity-scale', type=float, default=1.)
    parser.add_argument('--settle-steps', type=int, default=150,
                        help='A gravity section-chain preparation steps')
    parser.add_argument('--target-cuff-force', type=float,
                        help='Fix the contraction stop target; default tries 0.08, 0.04, 0.02 N with unchanged hold gates')
    parser.add_argument('--fit-settle-steps', type=int,
                        help='Relaxation per 1 percent contraction; default:5')
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--phase', choices=['fit', 'settle'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 0 < args.fit_strength <= 1 or not 0 < args.skin_clearance_m < .02 or not 0 < args.drape_gravity_scale <= 1:
        parser.error('Invalid skin fit or drape gravity parameters')
    if not 100 <= args.settle_steps <= 2000:
        parser.error('--settle-steps must be between 100 and 2000')
    if args.target_cuff_force is not None and not 0 < args.target_cuff_force <= .5:
        parser.error('--target-cuff-force must be between zero and 0.5 N')
    if args.fit_settle_steps is not None and args.fit_settle_steps < 1:
        parser.error('--fit-settle-steps must be positive')
    patterns = read_patterns(ROOT / 'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
    if args.numbers:
        if set(args.numbers) - {p.number for p in patterns}:
            parser.error('Unknown workbook number')
        patterns = [p for p in patterns if p.number in args.numbers]
    trials = [(p, s) for p in patterns for s in (args.stages or [p.stage]) if s in STAGES]
    if args.phase and len(trials) != 1:
        parser.error('A native worker requires exactly one number and stage')
    for pattern, stage in trials:
        directory = state_directory(args.output.resolve(), pattern.number, stage)
        directory.mkdir(parents=True, exist_ok=True)
        if args.phase:
            prepare_phase(args, pattern, stage, directory)
            return
        if not args.retry and (directory / 'state.json').exists():
            load_stage_state(args.output, pattern, stage)
        else:
            # Each physical phase gets a fresh collision world, as in A's
            # validated preparation. Native fitting has no in-place stop API.
            try:
                baseline = state_directory(ROOT/BASELINE_ROOT, pattern.number, stage)/'fit'
                if not (directory/'fit/report.json').exists() and (baseline/'report.json').exists():
                    saved_fit = json.loads((baseline/'report.json').read_text())
                    if saved_fit.get('passed') and saved_fit.get('pattern') == asdict(pattern):
                        shutil.copytree(baseline, directory/'fit', dirs_exist_ok=True)
                        write(directory/'fit/provenance.json', dict(source=str(baseline),
                            report_sha256=hashlib.sha256((baseline/'report.json').read_bytes()).hexdigest(),
                            scope='Native reference retention before supported surface fitting'))
                for phase in ('fit', 'settle'):
                    fit_path = directory / 'fit/report.json'
                    if phase == 'fit' and fit_path.exists() and not args.retry:
                        previous = json.loads(fit_path.read_text())
                        if previous.get('passed') and previous.get('pattern') == asdict(pattern):
                            continue
                    command = [sys.executable, '-u', str(Path(__file__).resolve()),
                        '--numbers', str(pattern.number), '--stages', stage,
                        '--output', str(args.output.resolve()), '--port', str(args.port),
                        '--settle-steps', str(args.settle_steps), '--phase', phase]
                    command += ['--fit-strength', str(args.fit_strength), '--skin-clearance-m', str(args.skin_clearance_m),
                                '--drape-gravity-scale', str(args.drape_gravity_scale)]
                    if args.fit_settle_steps is not None:
                        command += ['--fit-settle-steps', str(args.fit_settle_steps)]
                    if phase == 'fit':
                        targets = [args.target_cuff_force] if args.target_cuff_force is not None else [.08, .04, .02]
                        for target in targets:
                            cached = json.loads(fit_path.read_text()) if fit_path.exists() else {}
                            cached_target = cached.get('fit_settings', {}).get('target_cuff_force_n', .08)
                            if (not args.retry and cached.get('pattern') == asdict(pattern)
                                    and cached_target == target and not cached.get('passed')
                                    and cached.get('fit_settings', {}).get('settle_steps') == (args.fit_settle_steps or 5)
                                    and cached.get('contact_neighbour_model') == 'A_material_coordinates_v1'):
                                code = 2
                            else:
                                code = subprocess.run(command + ['--target-cuff-force', str(target)], cwd=ROOT).returncode
                            if fit_path.exists():
                                archive = directory/'fit-attempts'/f'force-{target:g}-material-contact'
                                shutil.copytree(fit_path.parent, archive, dirs_exist_ok=True)
                            if code == 0:
                                break
                        if code != 0:
                            raise subprocess.CalledProcessError(code, command)
                    else:
                        subprocess.run(command, cwd=ROOT, check=True)
            except subprocess.CalledProcessError as error:
                failure = dict(number=pattern.number, stage=stage, passed=False, error=str(error))
                write(directory / 'failure.json', failure)
                summarize(args.output, trials, failure=failure)
                raise
        (directory / 'failure.json').unlink(missing_ok=True)
        summarize(args.output, trials)


if __name__ == '__main__':
    main()
