#!/usr/bin/env python3
"""Search one pair of starts, then validated opening routes for all 70 states."""
from __future__ import annotations

import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
from collections import Counter
import copy
import itertools
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from search_centered_reaching import write, REGISTERED_MANIFEST
from search_fixed_torso_starts import context
from sock_dressing_simulation.common_opening_approach import (
    inventory, load_state, reference, union_scene, own_arm_issues, sha256, orient,
    approach_targets, edge_samples, guided_approach_targets, linear_quintic_profile,
)
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics, ControlledCollisionScene


def inputs(args):
    initial = yaml.safe_load(args.reference_initial.read_text())
    rows = inventory(ROOT)
    write(args.output/'states.json', dict(rows=rows, count=len(rows),
        reference_initial=str(args.reference_initial.relative_to(ROOT)), reference_initial_sha256=sha256(args.reference_initial),
        translation_m=initial['translation_m'], geometry_manifest_sha256=sha256(REGISTERED_MANIFEST)))
    return initial, rows


def starts(args):
    initial, rows = inputs(args)
    began = time.monotonic()
    k, base, state = union_scene(ROOT, rows, initial)
    base.park_left(initial['left_q_rad'], .04)
    models = {s: ControlledArmKinematics(k, s, initial['right_q_rad'], .04) for s in ('left', 'right')}
    ref = reference(ROOT, models['left'], state, np.asarray(initial['translation_m']))
    write(args.output/'reference.json', ref)
    print('REFERENCE', ref['template_knots'], 'left', ref['left']['position_m'], 'right', ref['right']['position_m'], flush=True)
    pools = {}
    for side in ('left', 'right'):
        cache = args.output/f'{side}-start-candidates.json'
        if args.reuse_candidates and cache.exists():
            pools[side] = json.loads(cache.read_text())['candidates']
            print('REUSE POOL', side, len(pools[side]), flush=True)
            continue
        model = copy.copy(models[side])
        model.lower = model.lower+np.deg2rad(5)
        model.upper = model.upper-np.deg2rad(5)
        checker = ControlledCollisionScene(base, side, initial['right_q_rad'], .04)
        p0, R0 = np.array(ref[side]['position_m']), np.array(ref[side]['rotation'])
        q0 = np.array(ref['left']['q_rad'] if side == 'left' else initial['right_q_rad'])
        alternate = q0.copy()
        alternate[4] = -2.4
        alternate[5] = .7
        alternate[6] = .3
        offsets = sorted((np.array(v)*.02 for v in itertools.product(range(-7, 8), repeat=3)
                          if np.linalg.norm(v)*.02 <= args.radius+1e-9), key=lambda x: (np.linalg.norm(x), -x[1], x[2]))
        counts, candidates = Counter(), []
        last = q0.copy()
        original_R = model.pose(initial[side+'_q_rad'])[1]
        rotations = [(R0, 'reflected_or_reference', 0), (original_R, 'original_arm_orientation', 0)]
        rotations += [(base_R@Rotation.from_rotvec([np.deg2rad(roll), 0., 0.]).as_matrix(), label, roll)
                      for roll in (30, -30, 60, -60, 90, -90)
                      for base_R, label in ((R0, 'reflected_or_reference'), (original_R, 'original_arm_orientation'))]
        for offset in offsets:
            if len(candidates) >= args.candidates:
                break
            for rotation, rotation_kind, roll in rotations:
                for seed in (last, q0, alternate, np.array(initial[side+'_q_rad'])):
                    counts['ik_tested'] += 1
                    result = model.solve(p0+offset, rotation, seed)
                    if result['position_error_m'] > 1e-4 or result['orientation_error_deg'] > .5:
                        continue
                    q = np.array(result['q'])
                    last = q
                    check = checker.slacks(q, .04, initial['translation_m'], details=True)
                    issues = own_arm_issues(check, side)
                    if issues:
                        counts['own_arm_collision'] += 1
                        if counts['own_arm_collision'] <= 3:
                            print('START REJECT', side, offset.tolist(), issues[:2], flush=True)
                        continue
                    if any(np.linalg.norm(q-np.array(c['q_rad'])) < .1 for c in candidates):
                        continue
                    candidates.append(dict(q_rad=q.tolist(), position_m=model.pose(q)[0].tolist(),
                        rotation=model.pose(q)[1].tolist(), offset_m=offset.tolist(),
                        center_distance_m=float(np.linalg.norm(offset)), tool_roll_offset_deg=roll,
                        rotation_kind=rotation_kind,
                        position_error_m=result['position_error_m'], orientation_error_deg=result['orientation_error_deg']))
                    print('START CANDIDATE', side, len(candidates), offset.tolist(), 'roll', roll, flush=True)
                    break
                if candidates and np.array_equal(np.array(candidates[-1]['offset_m']), offset):
                    break
        pools[side] = candidates
        write(args.output/f'{side}-start-candidates.json', dict(center=ref[side], counts=dict(counts), candidates=candidates))
        print('POOL', side, len(candidates), dict(counts), flush=True)
    pairs = sorted(itertools.product(pools['left'], pools['right']),
                   key=lambda pair: pair[0]['center_distance_m']+pair[1]['center_distance_m'])
    if args.right_candidate_index is not None:
        selected_right = pools['right'][args.right_candidate_index]
        pairs = [(left, right) for left, right in pairs if right is selected_right]
    accepted, failures = [], []
    for left, right in pairs:
        base.park_left(left['q_rad'], .04)
        checker = ControlledCollisionScene(base, 'right', right['q_rad'], .04)
        check = checker.slacks(right['q_rad'], .04, initial['translation_m'], details=True)
        if not check['passed']:
            failures.append(dict(left=left['position_m'], right=right['position_m'], issues=check['issues']))
            continue
        accepted.append(dict(left=left, right=right, check=check))
        if len(accepted) >= args.pairs:
            break
    write(args.output/'start-search.json', dict(reference=ref, candidate_pairs=accepted,
        rejected_pairs=failures, elapsed_s=time.monotonic()-began, shared_state_count=len(rows),
        collision_screen='Union of all 70 cloth surfaces and all 32 native human/leg postures; full registered robot geometry'))
    if not accepted:
        raise RuntimeError('No common collision-free pair found within the requested center radius')
    chosen = accepted[0]
    config = dict(initial)
    for side in ('left', 'right'):
        config[side+'_q_rad'] = chosen[side]['q_rad']
        config[side+'_q_deg'] = np.rad2deg(chosen[side]['q_rad']).tolist()
        config[side+'_gap_m'] = .04
    config['initialization'] = 'Common opening approach starts selected around the A penultimate waypoint and its foot reflection; identical for every sock state'
    config['minimum_joint_margin_deg'] = min(float(np.rad2deg(np.minimum(
        np.asarray(config[s+'_q_rad'])-models[s].lower,
        models[s].upper-np.asarray(config[s+'_q_rad']))).min()) for s in ('left', 'right'))
    for key in ('facing_cuff_required', 'forearms_down_required', 'common_start_reference',
                'common_left_joint_delta_deg', 'common_right_joint_delta_deg'):
        config.pop(key, None)
    config['joint_limit_margin_required_deg'] = 5.
    config['right_start_candidate_index'] = args.right_candidate_index
    config['source_reference'] = str((args.output/'reference.json').relative_to(ROOT))
    config['reachability_verified'] = False
    (args.output/'initial.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    print('COMMON START PAIR FOUND', chosen['left']['position_m'], chosen['right']['position_m'], flush=True)


def validate_route(model, checker, raw, gap, delta, target):
    from validate_compound_trajectory import smooth_profile
    raw = np.asarray(raw)
    profile = smooth_profile(raw, model.velocity, model.acceleration)
    prior_failures = []
    interpolation = 'PCHIP with global quintic time map'
    for attempt in range(2):
        grid = np.unique(np.r_[profile['u'], (profile['u'][:-1]+profile['u'][1:])/2])
        qs = profile['curve'](grid)
        margin = float(np.rad2deg(np.minimum(qs-model.lower, model.upper-qs)).min())
        failures = []
        for i, q in enumerate(qs):
            if not checker.is_clear(q, gap, delta):
                failures.append(dict(index=i, check=checker.slacks(q, gap, delta, details=True)))
                break
        if not failures or attempt == 1:
            break
        prior_failures.append(dict(interpolation=interpolation, failures=failures))
        profile = linear_quintic_profile(raw, model.velocity, model.acceleration)
        interpolation = 'Exact joint polyline with piecewise quintic timing and zero velocity/acceleration at turns'
    actual, R = model.pose(qs[-1])
    desired_R = np.asarray(target['rotation'])
    pe = float(np.linalg.norm(actual-target['tcp_target_m']))
    ae = float(np.rad2deg(Rotation.from_matrix(R@desired_R.T).magnitude()))
    tip_error = float(np.linalg.norm(actual+.010*R[:, 0]-target['tip_midplane_target_m']))
    vf = float(np.max(profile['velocity_bound']/model.velocity))
    af = float(np.max(profile['acceleration_bound']/model.acceleration))
    passed = not failures and margin >= 5-1e-6 and pe <= 1e-4 and ae <= .5 and tip_error <= .0002 and vf <= .1+1e-9 and af <= .1+1e-9
    validation = dict(passed=bool(passed), checked_samples=len(qs), failures=failures,
        minimum_joint_margin_deg=margin, endpoint_position_error_m=pe,
        endpoint_orientation_error_deg=ae, physical_tip_midplane_error_m=tip_error,
        maximum_velocity_fraction=vf, maximum_acceleration_fraction=af,
        interpolation=interpolation+'; independent curve-grid and midpoint collision checks',
        rejected_interpolations=prior_failures,
        continuous_collision_certified=False)
    data = dict(raw_q_rad=raw.tolist(), q_rad=profile['q'].tolist(), times_s=profile['t'].tolist(),
        velocity_rad_s=profile['velocity'].tolist(), acceleration_rad_s2=profile['acceleration'].tolist(),
        tcp_world_m=[model.pose(q)[0].tolist() for q in profile['q']], duration_s=float(profile['duration']),
        joint_names=model.arm_names, opening_gap_m=gap, target=target, validation=validation)
    if 'knots_q_rad' in profile:
        data.update(knots_q_rad=profile['knots_q_rad'], segment_durations_s=profile['segment_durations_s'])
    return data


def side_connection(model, checker, start, pre, gap, delta, side):
    """Try a short outward detour before a general joint-space search."""
    p, R = model.pose(start)
    sign = 1. if side == 'left' else -1.
    for outward, lift in ((.04, .02), (.07, .02), (.07, .05), (.10, .03)):
        target = p+np.array([sign*outward, lift, 0.])
        solution = model.solve(target, R, start)
        if solution['position_error_m'] > 1e-4 or solution['orientation_error_deg'] > .5:
            continue
        waypoint = np.asarray(solution['q'])
        if (all(checker.is_clear(q, gap, delta) for q in edge_samples(start, waypoint, .75))
                and all(checker.is_clear(q, gap, delta) for q in edge_samples(waypoint, pre, .75))):
            return dict(passed=True, raw_q=[start.tolist(), waypoint.tolist(), pre.tolist()],
                        stage='outward_side_connection')
    return dict(passed=False)


def plan_arm(args, row, state, k, base, initial, side):
    from sock_dressing_simulation.grasp_geometry import SkinBody
    from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
    from plan_fixed_torso_route import rrt
    began = time.monotonic()
    print('PLANNING', row['id'], side, flush=True)
    start = np.asarray(initial[side+'_q_rad'])
    delta = np.asarray(initial['translation_m'])
    base.park_left(initial['left_q_rad'], initial['left_gap_m'])
    actual_model = ControlledArmKinematics(k, side, initial['right_q_rad'], initial['right_gap_m'])
    model = copy.copy(actual_model)
    model.lower, model.upper = model.lower+np.deg2rad(5), model.upper-np.deg2rad(5)
    checker = ControlledCollisionScene(base, side, initial['right_q_rad'], initial['right_gap_m'])
    initial_check = checker.slacks(start, initial[side+'_gap_m'], delta, details=True)
    counts = Counter()
    result = dict(condition=row, side=side, initial_sha256=sha256(args.output/'initial.yaml'),
        start_q_rad=start.tolist(), other_arm_q_rad=initial[('left' if side == 'right' else 'right')+'_q_rad'],
        translation_m=delta.tolist(), initial_check=initial_check, passed=False, counts={},
        task='Open-gripper approach: physical finger-tip midplane 2 mm outside a material point in the opening 0--15 mm band; no closure or capture requirement',
        geometry_manifest_sha256=sha256(REGISTERED_MANIFEST), executed=False)
    result['search_parameters'] = dict(seconds=args.seconds, rrt_seconds=args.rrt_seconds,
        iterations=args.iterations, arm_guided=args.guided and side == 'right',
        spatial_aabb=True, final_fixed_attitude_segment_required=False)
    if not initial_check['passed']:
        result['reason'] = 'shared_initial_collision'
        return result
    skin = SkinBody(state['source']['skin_surface'])
    cloth = ClothScreen(state['particles_world'], state['triangles'])
    reference_R = actual_model.pose(start)[1]
    rng = np.random.default_rng(20261108+row['number']+100*ord(row['stage'])+(side == 'right'))
    alternate = start.copy()
    alternate[4] = -2.4
    alternate[5] = .7
    alternate[6] = .3
    recent = start.copy()
    rejected = []
    bank_cache = {}
    seed_path = args.output/'seed-routes'/side/f'{row["id"]}.json'
    seed_route = json.loads(seed_path.read_text()) if seed_path.exists() else None
    targets = approach_targets(state, side, reference_R, delta)
    if args.guided and side == 'right':
        targets = itertools.chain(guided_approach_targets(state, side, model, start, delta), targets)
    if seed_route and seed_route['condition']['state_sha256'] == row['state_sha256']:
        targets = itertools.chain([seed_route['route']['target']], targets)
    for target in targets:
        if time.monotonic()-began > args.seconds:
            break
        R = np.asarray(target['rotation'])
        p = np.asarray(target['tcp_target_m'])
        for gap in (.04, .02, .05):
            counts['geometry_tested'] += 1
            if gap not in bank_cache:
                bank_cache[gap] = k.hand.bank(gap)
            bank = bank_cache[gap]
            if not skin.inspect(bank, R, p-delta, .001)[0]:
                counts['hand_skin'] += 1
                continue
            if cloth.intersections(bank, R, p-delta):
                counts['hand_cloth'] += 1
                continue
            counts['endpoint_geometry_passed'] += 1
            seeds = [recent, start, alternate]
            if 'seed_q_rad' in target:
                seeds.insert(0, np.asarray(target['seed_q_rad']))
            if seed_route and target == seed_route['route']['target']:
                seeds.insert(0, np.asarray(seed_route['route']['q_rad'][-1]))
            if counts['endpoint_geometry_passed'] % 12 == 0:
                seeds += [np.clip(start+rng.normal(0, .7, 7), model.lower+1e-6, model.upper-1e-6)]
            for seed in seeds:
                if time.monotonic()-began > args.seconds:
                    break
                counts['ik_tested'] += 1
                ik = model.solve(p, R, seed)
                if ik['position_error_m'] > 1e-4 or ik['orientation_error_deg'] > .5:
                    counts['ik_failed'] += 1
                    continue
                goal = np.asarray(ik['q'])
                recent = goal
                if not checker.is_clear(goal, gap, delta):
                    counts['endpoint_arm_collision'] += 1
                    if len(rejected) < 12:
                        rejected.append(dict(stage='endpoint_arm_collision', target=target,
                            check=checker.slacks(goal, gap, delta, details=True)))
                    continue
                counts['full_endpoint_passed'] += 1
                # Prefer a straight final segment when one IK branch permits
                # it; the user's opening approach does not require 40 mm of
                # fixed-attitude travel. Every connection is checked in full.
                reverse = [goal]
                for distance in (.002, .005, .01, .02, .03, .04):
                    pre = model.solve(p-R[:, 0]*distance, R, reverse[-1])
                    q = np.asarray(pre['q'])
                    if pre['position_error_m'] > 1e-4 or pre['orientation_error_deg'] > .5:
                        break
                    if np.max(abs(q-reverse[-1])) > np.deg2rad(12):
                        break
                    if not all(checker.is_clear(v, gap, delta) for v in edge_samples(reverse[-1], q, .75)):
                        break
                    reverse.append(q)
                if len(reverse) == 1:
                    counts['no_straight_final_segment'] += 1
                opening_ok = all(checker.is_clear(start, float(g), delta)
                                 for g in np.linspace(initial[side+'_gap_m'], gap, 5))
                if not opening_ok:
                    counts['initial_jaw_adjustment_failed'] += 1
                    continue
                pre = reverse[-1]
                if all(checker.is_clear(q, gap, delta) for q in edge_samples(start, pre, 1.)):
                    connection = dict(passed=True, raw_q=[start.tolist(), pre.tolist()], stage='straight_joint_connection')
                else:
                    connection = side_connection(model, checker, start, pre, gap, delta, side)
                    if not connection['passed']:
                        counts['rrt_attempted'] += 1
                        # rrt applies the 5 degree margin itself. Pass hard model
                        # bounds so the margin is not applied twice.
                        connection = rrt(actual_model, checker, start, pre, gap, delta, iterations=args.iterations,
                                         timeout_s=min(args.rrt_seconds, max(.1, args.seconds-(time.monotonic()-began))))
                if not connection['passed']:
                    counts['connection_failed'] += 1
                    continue
                raw = connection['raw_q']+[q.tolist() for q in reverse[-2::-1]]
                write(args.output/'candidate-routes'/side/f'{row["id"]}.json', dict(
                    condition=row, side=side, initial_sha256=result['initial_sha256'],
                    raw_q_rad=raw, gap_m=gap, target=target, passed=False,
                    note='Planned candidate only; requires the independent interpolated collision check'))
                print('VALIDATING', row['id'], side, len(raw), 'elapsed', round(time.monotonic()-began, 1), flush=True)
                route = validate_route(actual_model, checker, raw, gap, delta, target)
                if not route['validation']['passed']:
                    counts['smooth_validation_failed'] += 1
                    if len(rejected) < 12:
                        rejected.append(dict(stage='smooth_validation_failed', validation=route['validation']))
                    continue
                actual, observed_R = actual_model.pose(np.asarray(route['q_rad'][-1]))
                if not skin.inspect(k.hand.bank(gap), observed_R, actual-delta, .001)[0] or cloth.intersections(k.hand.bank(gap), observed_R, actual-delta):
                    counts['actual_endpoint_geometry_failed'] += 1
                    continue
                result.update(passed=True, route=route, counts=dict(counts),
                    elapsed_s=time.monotonic()-began, rejected=rejected,
                    connection_stage=connection['stage'], initial_jaw_adjustment_passed=True,
                    actual_endpoint_skin_and_cloth_passed=True,
                    final_cartesian_approach_distance_m=float(np.linalg.norm(model.pose(reverse[-1])[0]-p)))
                print('ROUTE PASS', row['id'], side, round(result['elapsed_s'], 1), dict(counts), flush=True)
                return result
    result.update(reason='bounded_search_exhausted', counts=dict(counts), rejected=rejected,
        elapsed_s=time.monotonic()-began)
    print('ROUTE FAIL', row['id'], side, dict(counts), flush=True)
    return result


def run_condition(payload):
    args, row = payload
    initial = yaml.safe_load((args.output/'initial.yaml').read_text())
    state = load_state(ROOT, row)
    print('CONTEXT', row['id'], flush=True)
    k, base, *_ = context(row['scene_number'], state, REGISTERED_MANIFEST,
        left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
    base.use_spatial_aabb = True
    results = []
    for side in args.sides:
        destination = args.output/'routes'/side/f'{row["id"]}.json'
        if destination.exists() and not args.retry:
            old = json.loads(destination.read_text())
            if old['initial_sha256'] != sha256(args.output/'initial.yaml') or old['condition']['state_sha256'] != row['state_sha256']:
                raise ValueError('Cached route uses a different initial pose or shape')
            if old['passed']:
                results.append(old)
                continue
            archive = args.output/'search-attempts'/side/row['id']
            archive.mkdir(parents=True, exist_ok=True)
            write(archive/f'attempt-{len(list(archive.glob("attempt-*.json"))):03d}.json', old)
        result = plan_arm(args, row, state, k, base, initial, side)
        write(destination, result)
        results.append(result)
    return [dict(id=r['condition']['id'], side=r['side'], passed=r['passed'],
                 file=str((args.output/'routes'/r['side']/f'{r["condition"]["id"]}.json').relative_to(ROOT)),
                 counts=r['counts']) for r in results]


def routes(args):
    initial, rows = inputs(args)
    if not (args.output/'initial.yaml').exists():
        raise RuntimeError('Run starts first')
    if args.numbers:
        rows = [r for r in rows if r['number'] in args.numbers]
    if args.exclude_numbers:
        rows = [r for r in rows if r['number'] not in args.exclude_numbers]
    if args.stages:
        rows = [r for r in rows if r['stage'] in args.stages]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_condition, (args, row)) for row in rows]
        for future in as_completed(futures):
            results.extend(future.result())
            write(args.output/'route-search-progress.json', dict(rows=results,
                passed=sum(r['passed'] for r in results), completed=len(results), requested=len(rows)*len(args.sides)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['starts', 'routes'])
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/common-opening-approach')
    parser.add_argument('--reference-initial', type=Path, default=ROOT/'config/diverse_left_reaching_shared.yaml')
    parser.add_argument('--radius', type=float, default=.14)
    parser.add_argument('--candidates', type=int, default=24)
    parser.add_argument('--pairs', type=int, default=12)
    parser.add_argument('--right-candidate-index', type=int)
    parser.add_argument('--reuse-candidates', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--sides', nargs='+', choices=['left', 'right'], default=['right', 'left'])
    parser.add_argument('--numbers', nargs='+', type=int)
    parser.add_argument('--exclude-numbers', nargs='+', type=int)
    parser.add_argument('--stages', nargs='+', choices=list('ABCDE'))
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--iterations', type=int, default=800)
    parser.add_argument('--rrt-seconds', type=float, default=20.)
    parser.add_argument('--retry', action='store_true')
    parser.add_argument('--guided', action='store_true', help='Prefer arm-guided exact physical-tip targets')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.reference_initial = args.reference_initial.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    {'starts': starts, 'routes': routes}[args.phase](args)


if __name__ == '__main__':
    main()
