#!/usr/bin/env python3
"""Search the user's open two-finger cloth/foot contact task, without closing.

Stages: geometry -> endpoints -> audit -> starts -> routes -> report. Outputs are separate from
the old 1 mm skin-clearance / 4 mm2 closure task and never overwrite its starts.
"""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
import html
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from search_centered_reaching import write, URDF, REGISTERED_MANIFEST
from search_fixed_torso_starts import context
from sock_dressing_simulation.common_opening_approach import inventory, load_state, sha256, edge_samples, linear_quintic_profile, piecewise_knots
from sock_dressing_simulation.contact_pregrasp import ContactGoal, ContactCriteria
from sock_dressing_simulation.grasp_geometry import RightHand, candidate_samples, orientation
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics, ControlledCollisionScene


class SearchDeadline(Exception):
    pass


class RepairDeadline(Exception):
    pass


def wide_starts(args,rows):
    """Search both hands outside/above the complete foot-pose inventory."""
    from itertools import product
    from sock_dressing_simulation.common_opening_approach import union_scene,own_arm_issues
    out=args.output.resolve();initial=yaml.safe_load(args.initial.read_text())
    k,scene,state=union_scene(ROOT,inventory(ROOT),initial)
    scene.use_spatial_aabb=True
    delta=np.asarray(initial['translation_m'])
    foot=np.asarray(state['source']['surfaces'][0]['vertices'])+delta
    center=(foot.min(0)+foot.max(0))/2
    pools={};counts=Counter()
    for side in ('left','right'):
        scene.park_left(initial['left_q_rad'],.04)
        model=ControlledArmKinematics(k,side,initial['right_q_rad'],.04)
        comfortable=copy.copy(model)
        comfortable.lower=model.lower+np.deg2rad(args.joint_margin_deg)
        comfortable.upper=model.upper-np.deg2rad(args.joint_margin_deg)
        checker=ControlledCollisionScene(scene,side,initial['right_q_rad'],.04)
        seeds=[np.asarray(initial[side+'_q_rad'])]
        representative=[]
        for identifier in ('E-50','E-07','C-10','D-34','A-10','B-08'):
            path=out/'endpoints'/f'{identifier}.json'
            if not path.exists():continue
            d=json.loads(path.read_text())
            accepted=sorted(d['sides'].get(side,{}).get('accepted',[]),key=endpoint_priority)
            if accepted:
                seed=np.asarray(accepted[0]['q_rad']);seeds.append(seed);representative.append(model.pose(seed)[1])
        rng=np.random.default_rng(20261115+(side=='right'))
        seeds.extend(rng.uniform(comfortable.lower,comfortable.upper,(3,7)))
        base=np.diag([-1.,1.,-1.]) if side=='left' else np.eye(3)
        rotations=[base,base@Rotation.from_euler('z',-20.,degrees=True).as_matrix()]+representative[:3]
        offsets=(.16,.22,.28) if side=='left' else (-.10,-.16,-.22)
        preferred=center+np.array([.22 if side=='left' else -.16,.22,-.05])
        positions=[center+np.array([x,y,z])for x,y,z in product(offsets,(.12,.22,.32),(-.15,-.05,.05))]
        positions.sort(key=lambda p:np.linalg.norm(p-preferred))
        candidates=[];began=time.monotonic()
        for desired in positions:
            if len(candidates)>=args.maximum or time.monotonic()-began>args.seconds:break
            found=None
            for R in rotations:
                if time.monotonic()-began>args.seconds:break
                for seed in seeds:
                    if time.monotonic()-began>args.seconds:break
                    sol=comfortable.solve(desired,R,seed);counts[side+'_ik']+=1
                    if sol['position_error_m']>1e-4 or sol['orientation_error_deg']>.5:continue
                    q=np.asarray(sol['q'])
                    check=checker.slacks(q,.04,delta,details=True)
                    if own_arm_issues(check,side):counts[side+'_own_collision']+=1;continue
                    p,actual_R=model.pose(q)
                    margin=float(np.rad2deg(np.minimum(q-model.lower,model.upper-q)).min())
                    record=dict(q_rad=q.tolist(),position_m=p.tolist(),rotation=actual_R.tolist(),
                        requested_position_m=desired.tolist(),minimum_joint_margin_deg=margin,
                        center_distance_m=float(np.linalg.norm(p-preferred)),source='wide_radial_waiting_pose')
                    if found is None or margin>found['minimum_joint_margin_deg']:found=record
                    if margin>=8:break
                if found is not None and found['minimum_joint_margin_deg']>=8:break
            if found:
                candidates.append(found)
                print('WIDE START',side,len(candidates),found['position_m'],found['minimum_joint_margin_deg'],flush=True)
        pools[side]=candidates
        write(out/f'wide-{side}-start-candidates.json',dict(candidates=candidates,counts=dict(counts),elapsed_s=time.monotonic()-began))
    pair_indices=[]
    for i in range(max(len(pools['left']),len(pools['right']))):
        if not pools['left'] or not pools['right']:break
        li=i%len(pools['left'])
        for ri in (i%len(pools['right']),(i+len(pools['right'])//2)%len(pools['right']),0):
            if (li,ri)not in pair_indices:pair_indices.append((li,ri))
    accepted=[]
    for li,ri in pair_indices:
        left,right=pools['left'][li],pools['right'][ri]
        scene.park_left(left['q_rad'],.04)
        check=scene.slacks(right['q_rad'],.04,delta,details=True)
        if not check['passed']:continue
        accepted.append(dict(left=left,right=right,check=check,source='wide_radial'))
        if len(accepted)>=args.pairs:break
    write(out/'additional-start-pairs.json',dict(candidate_pairs=accepted,pools=pools,shared_state_count=70,
        reference_initial_sha256=sha256(args.initial),geometry_manifest_sha256=sha256(REGISTERED_MANIFEST),counts=dict(counts)))
    print('WIDE INITIAL PAIRS',len(accepted),flush=True)


def protocol():
    return dict(task='Both open hands on opposite foot sides; one finger near cloth material depth 15--30 mm, the other near actual foot skin; positive-length local rim segment between finite jaws. No closure.',
                criteria=asdict(ContactCriteria()), contact_definition='Surface distance in [0, 0.5 mm], no penetration; not force contact',
                rim_definition='A local actual boundary segment at each hand; not the whole circumference in each hand',
                skin_contact_region='Entire actual saved target-leg skin, foot through calf; user-confirmed, no ankle cutoff',
                geometry_coordinates='Contact points and hand-only poses use the saved-state frame; add translation_m for the robot world frame',
                initial_coordinates='Unity world metres, +Y up; rotations map canonical hand axes into this world',
                side_definition='TCPs lie on opposite sides of the foot sagittal plane; left is oriented toward world +X. Contact witness sides are reported, not separately constrained.',
                saved_shapes_unchanged=True, base_torso_unchanged=True, closure_performed=False,
                native_dynamics_verified=False, continuous_collision_certified=False)


def audit_geometry(out, rows, initial_path=None):
    hand = RightHand(URDF)
    for row in rows:
        path = out/'geometry'/f'{row["id"]}.json'
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        if d['condition'] != row or d['hand_urdf_sha256'] != sha256(URDF):
            raise ValueError('Stale geometry source: '+str(path))
        goal = ContactGoal(load_state(ROOT, row), hand)
        endpoint_path = out/'endpoints'/f'{row["id"]}.json'
        endpoints = json.loads(endpoint_path.read_text()) if endpoint_path.exists() else None
        if endpoints and initial_path:
            initial = yaml.safe_load(initial_path.read_text())
            k = base = None
            for side, result in endpoints['sides'].items():
                best = result.get('best')
                if not best:
                    continue
                c = best['contact_check']
                check = goal.inspect(np.asarray(c['position_m']), np.asarray(c['rotation']), c['opening_gap_m'], side, c['cloth_finger'])
                best['contact_check'] = check
                best['residual_norm_mm'] = float(np.linalg.norm(goal.residual(np.asarray(c['position_m']), np.asarray(c['rotation']), c['opening_gap_m'], side, c['cloth_finger'])))
                if not check['passed'] or any(np.linalg.norm(np.asarray(g['q_rad'])-best['q_rad']) < .1 for g in result['accepted']):
                    continue
                if base is None:
                    k, base, *_ = context(row['number'], load_state(ROOT, row), REGISTERED_MANIFEST,
                                         left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
                    original_pairs = list(base.pairs)
                base.pairs = list(original_pairs)
                contact_scene(base, {side:check['foot_finger']})
                base.park_left(initial['left_q_rad'], initial['left_gap_m'])
                model = ControlledArmKinematics(k, side, initial['right_q_rad'], initial['right_gap_m'])
                p, R = model.pose(best['q_rad'])
                check = goal.inspect(p-np.asarray(initial['translation_m']), R, c['opening_gap_m'], side, c['cloth_finger'])
                checker = ControlledCollisionScene(base, side, initial['right_q_rad'], initial['right_gap_m'])
                whole = checker.slacks(best['q_rad'], c['opening_gap_m'], initial['translation_m'], details=True)
                if check['passed'] and whole['passed']:
                    result['accepted'].append(dict(q_rad=best['q_rad'], opening_gap_m=c['opening_gap_m'], contact_check=check,
                        whole_scene_check=whole, tcp_world_m=p.tolist(), rotation=R.tolist(), source='Independent audit against clarified contact region'))
        for side, record in d['sides'].items():
            good, rejected = [], []
            for old in record['accepted']:
                check = goal.inspect(np.asarray(old['position_m']), np.asarray(old['rotation']), old['opening_gap_m'], side, old['cloth_finger'])
                (good if check['passed'] else rejected).append(check)
            record['accepted'] = good
            record['independent_audit_rejected'] = rejected
            record['independent_audit_passed'] = True
            if endpoints and side in endpoints['sides']:
                endpoint_good = []
                for endpoint in endpoints['sides'][side]['accepted']:
                    old = endpoint['contact_check']
                    check = goal.inspect(np.asarray(old['position_m']), np.asarray(old['rotation']), endpoint['opening_gap_m'], side, old['cloth_finger'])
                    if check['passed']:
                        endpoint['contact_check'] = check
                        endpoint_good.append(endpoint)
                        if not any(np.linalg.norm(np.asarray(c['position_m'])-check['position_m']) < .0001 and
                                   Rotation.from_matrix(np.asarray(c['rotation'])@np.asarray(check['rotation']).T).magnitude() < .01 for c in good):
                            good.append(dict(check, source='Independently verified actual arm FK endpoint'))
                endpoints['sides'][side]['accepted'] = endpoint_good
        d['protocol'] = protocol()
        write(path, d)
        if endpoints:
            endpoints['protocol'] = protocol()
            endpoints['independent_contact_audit_completed'] = True
            write(endpoint_path, endpoints)
    report(out, rows)


def seed_poses(goal, row, side, rng):
    seeds = []
    # Include prior open-hand approach and hand-only successes as proposals,
    # never as evidence of contact under the new task.
    for name in [f'artifacts/common-opening-approach/negative-wrist-start/routes/{side}/{row["id"]}.json',
                 f'artifacts/common-opening-approach/negative-wrist-start/{side}/{row["id"]}.json']:
        path = ROOT/name
        if path.exists():
            record = json.loads(path.read_text())
            target = record.get('route', {}).get('target', {})
            if target:
                R = np.asarray(target['rotation'])
                initial = yaml.safe_load((ROOT/'config/common_opening_initial_pose.yaml').read_text())
                p = np.asarray(target['tcp_target_m'])-initial['translation_m']
                for gap in (.008, .025, .045):
                    for finger in ('left_finger_link', 'right_finger_link'):
                        seeds.append((p, R, gap, finger))
    for directory in ('', 'transport', 'transfer-all', 'symmetry', 'foot-reflection'):
        path = ROOT/'artifacts/common-grasp-start-search'/directory/'geometry'/f'{row["id"]}.json'
        if path.exists():
            for record in json.loads(path.read_text()).get('accepted', [])[:5]:
                for gap in (.008, .025, .045):
                    for finger in ('left_finger_link', 'right_finger_link'):
                        seeds.append((np.asarray(record['position']), np.asarray(record['rotation']), gap, finger))
    samples = candidate_samples(goal.state, 32, (.015, .0225, .030))
    sign = 1 if side == 'left' else -1
    samples = [s for s in samples if sign*np.dot(np.asarray(s['position'])-goal.center, goal.lateral) > 0]
    generated,axial = [],[]
    for sample in samples:
        # The new task also allows one tip on bare skin above the rim and the
        # other on cloth below it. Seed closing along the garment axis, with
        # the palm OUTSIDE the foot, instead of assuming a radial cloth pinch.
        normal=np.asarray(sample['normal']);along=np.asarray(sample['along'])
        for closing_sign in (1,-1):
            finger='right_finger_link' if closing_sign==1 else 'left_finger_link'
            label='right_gripper/'+finger
            v=goal.vertices[label];tip=v[v[:,0]>=v[:,0].max()-1e-7].mean(0)
            base=np.column_stack([-normal,closing_sign*along,np.cross(-normal,closing_sign*along)])
            for tilt in (-20.,0.,20.):
                R=base@Rotation.from_euler('z',tilt,degrees=True).as_matrix()
                for gap in (.025,.035,.045):
                    local=tip+np.array([0.,goal.coefficient(label)*(gap-goal.reference_gap),0.])
                    p=np.asarray(sample['position'])-R@local-R[:,0]*goal.criteria.contact_tolerance_m/2
                    axial.append((p,R,gap,finger))
        for yaw in (0., 180.):
            for tilt in (-45., 0., 45.):
                for gap in (.003, .012, .028, .048):
                    for closing_sign in (1, -1):
                        R = orientation(sample, yaw, tilt, closing_sign)
                        finger = 'right_finger_link' if closing_sign == 1 else 'left_finger_link'
                        # Place the cloth-facing inner surface at the target;
                        # vary contact along the 105 mm finite finger length.
                        x = rng.choice([-.005, -.015, -.030])
                        local_y = closing_sign*(gap/2+goal.criteria.contact_tolerance_m/2)
                        p = np.asarray(sample['position'])-R@np.array([x, local_y, 0.])
                        generated.append((p, R, gap, finger))
    rng.shuffle(generated)
    rng.shuffle(axial)
    return axial+seeds+generated


def geometry_one(payload):
    row, out, seconds, maximum = payload
    state = load_state(ROOT, row)
    goal = ContactGoal(state, RightHand(URDF))
    result = dict(condition=row, protocol=protocol(), hand_urdf_sha256=sha256(URDF), sides={})
    for side in ('left', 'right'):
        rng = np.random.default_rng(20261112+row['number']+ord(row['stage'])*100+(side == 'right'))
        began = time.monotonic()
        deadline = began+seconds
        seeds = seed_poses(goal, row, side, rng)
        counts = Counter()
        ranked, accepted, best = [], [], None
        def residual(p, R, gap, finger):
            if time.monotonic() > deadline:
                raise SearchDeadline
            counts['geometry_evaluations'] += 1
            return goal.residual(p, R, gap, side, finger)
        def save():
            result['sides'][side] = dict(accepted=accepted, best=best, counts=dict(counts), elapsed_s=time.monotonic()-began,
                                         budget_s=seconds, budget_exhausted=time.monotonic() >= deadline)
            write(out/'geometry'/f'{row["id"]}.json', result)
        try:
            # Alternate small batches of new proposals and bounded local fits.
            for batch in range(10000):
                for index in range(batch*32, min(len(seeds), (batch+1)*32)):
                    p, R, gap, finger = seeds[index]
                    score = float(np.linalg.norm(residual(p, R, gap, finger)))
                    ranked.append((score, p, R, gap, finger))
                    counts['seed_poses'] += 1
                ranked.sort(key=lambda r: r[0])
                if not ranked:
                    break
                score, p0, R0, gap0, finger = ranked.pop(0)
                def fun(x):
                    return residual(p0+x[:3], R0@Rotation.from_rotvec(x[3:6]).as_matrix(), x[6], finger)
                fitted = least_squares(fun, np.r_[np.zeros(6), gap0],
                    bounds=(np.r_[[-.035]*3, [-1.3]*3, .0001], np.r_[[.035]*3, [1.3]*3, .05]),
                    diff_step=2e-4, x_scale=np.r_[[.01]*3, [.3]*3, .02], max_nfev=90,
                    ftol=1e-7, xtol=1e-7, gtol=1e-6)
                counts['local_fits'] += 1
                x = fitted.x
                p, R, gap = p0+x[:3], R0@Rotation.from_rotvec(x[3:6]).as_matrix(), float(x[6])
                score = float(np.linalg.norm(fitted.fun))
                if best is None or score < best['residual_norm_mm']:
                    # Full independent test, including exact rim edge clipping.
                    check = goal.inspect(p, R, gap, side, finger)
                    best = dict(residual_norm_mm=score, **check)
                    save()
                else:
                    check = goal.inspect(p, R, gap, side, finger) if score < 1. else None
                if check and check['passed']:
                    duplicate = any(np.linalg.norm(p-np.asarray(c['position_m'])) < .003 and Rotation.from_matrix(R@np.asarray(c['rotation']).T).magnitude() < .2 for c in accepted)
                    if not duplicate:
                        accepted.append(check)
                        print('CONTACT GEOMETRY', row['id'], side, len(accepted), score, flush=True)
                        save()
                        if len(accepted) >= maximum:
                            break
                if (batch+1)*32 >= len(seeds) and not ranked:
                    break
        except SearchDeadline:
            counts['deadline'] += 1
        save()
        print('GEOMETRY DONE', row['id'], side, len(accepted), None if best is None else round(best['residual_norm_mm'], 4), flush=True)
    return dict(id=row['id'], left=len(result['sides']['left']['accepted']), right=len(result['sides']['right']['accepted']))


def contact_scene(base, assignments):
    """Only the designated foot finger loses its old 1 mm skin margin.

    Robot/body and inter-arm margins remain intact. Use the original hand
    convex meshes for exact surface checks; do not suppress cloth collisions.
    """
    pairs = []
    for a, b, margin in base.pairs:
        for side, foot_finger in assignments.items():
            if a == side+'_gripper/'+foot_finger and b in ('right_leg_actual_skin_rigid', 'human_full_skin'):
                margin = 0.
        pairs.append((a, b, margin))
    base.pairs = pairs
    base.fixed_pair_cache = {}
    base.fixed_pair_valid_cache = {}


def retain_blocking_pairs(base, full_pairs, full_shoulders, blocking):
    """Keep the actual shoulder triangle guard when reindexing a pair subset."""
    base.pairs = [pair for pair in full_pairs if tuple(pair[:2]) in blocking]
    indices = {tuple(pair[:2]):index for index,pair in enumerate(base.pairs)}
    base.shoulder_pairs = {side:indices[tuple(full_pairs[old_index][:2])] for side,old_index in full_shoulders.items()
                           if tuple(full_pairs[old_index][:2]) in indices}


def opening_retreat_hint(rim, cloth_point):
    """Point from the cloth contact back toward the nearest actual rim edge."""
    a,b=np.asarray(rim)[:,0],np.asarray(rim)[:,1]
    edge=b-a
    t=np.clip(np.einsum('ij,ij->i',np.asarray(cloth_point)-a,edge)/np.maximum(np.sum(edge*edge,axis=1),1e-20),0.,1.)
    closest=a+t[:,None]*edge
    v=closest[np.argmin(np.linalg.norm(closest-cloth_point,axis=1))]-cloth_point
    return None if np.linalg.norm(v)<1e-8 else v/np.linalg.norm(v)


def endpoint_priority(endpoint):
    """Prefer external approaches with the skin tip beyond the cuff rim."""
    c=endpoint['contact_check'];w=np.asarray(c['contacts']['foot']['witnesses_m']);n=w[0]-w[1]
    alignment=abs(float(np.asarray(c['rotation'])[:,0]@n/max(np.linalg.norm(n),1e-12)))
    return (0 if alignment>=.5 else 1,c.get('skin_contact_opening_side_m',0.),-alignment)


def plan_contact_route(model, checker, start, target, gap, delta, row, side, lateral, seconds, joint_margin_deg=5.,retreat_hint=None,route_root=None):
    """Reuse verified-shaped approaches and explicitly leave narrow contacts."""
    from plan_fixed_torso_route import rrt
    began = time.monotonic()
    checked = {}
    def trace(stage,**values):
        if os.environ.get('CONTACT_ROUTE_TRACE'):
            print('CONTACT TRACE',row['id'],side,stage,round(time.monotonic()-began,2),len(checked),values,flush=True)
    def clear(q):
        key = tuple(np.round(q, 10))
        if key not in checked:
            checked[key] = checker.is_clear(q, gap, delta)
        return checked[key]
    def edge(a, b, deadline=None):
        for q in edge_samples(a, b, .25):
            if time.monotonic()-began>=seconds or (deadline is not None and time.monotonic()>=deadline) or not clear(q):
                return False
        return True
    if not clear(start):
        return dict(passed=False, stage='invalid_contact_route_endpoint')
    p, R = model.pose(target)
    comfortable = copy.copy(model)
    comfortable.lower = model.lower+np.deg2rad(joint_margin_deg)
    comfortable.upper = model.upper-np.deg2rad(joint_margin_deg)
    # A contact pose can have several IK solutions. Follow it from the free
    # start as well as retreating from the saved joint solution: a different
    # elbow configuration is acceptable only after independent FK/contact and
    # simultaneous-scene checks in the caller.
    forward_deadline = began+min(8., seconds*.10)
    for distance in (.08,.04,.12):
        if time.monotonic()>=forward_deadline:
            break
        solved=comfortable.solve(p-distance*R[:,0],R,start)
        q=np.asarray(solved['q'])
        if solved['position_error_m']>.00002 or solved['orientation_error_deg']>.05 or not edge(start,q,forward_deadline):
            continue
        forward=[start,q]
        for remaining in np.linspace(distance,0.,int(np.ceil(distance/.002))+1)[1:]:
            if time.monotonic()>=forward_deadline:
                break
            fitted=comfortable.solve(p-remaining*R[:,0],R,forward[-1])
            q=np.asarray(fitted['q'])
            if fitted['position_error_m']>.000002 or fitted['orientation_error_deg']>.02 or not edge(forward[-1],q,forward_deadline):
                break
            forward.append(q)
        else:
            return dict(passed=True,stage='forward_cartesian_contact_approach',raw_q=[v.tolist() for v in forward])
    if not clear(target):
        return dict(passed=False, stage='invalid_contact_route_endpoint')
    trace('forward_complete')
    old_path = ROOT/'artifacts/common-opening-approach/negative-wrist-start/routes'/side/f'{row["id"]}.json'
    if edge(start,target,time.monotonic()+min(3.,seconds*.05)):
        return dict(passed=True, stage='straight_contact_route', raw_q=[start.tolist(), target.tolist()])
    prefix = [start]
    # Rechecking a long old approach must leave time for the new contact
    # retreat and free-space search, even if that old route remains clear.
    warm_deadline=began+seconds*.25
    proposals=[]
    if route_root is not None:
        neighbors=[]
        directories=[]
        search_path=route_root/'initial-search.json'
        if search_path.exists():
            for candidate in json.loads(search_path.read_text())['candidates']:
                pair=candidate['pair']
                if np.allclose(pair[side]['q_rad'],start,atol=1e-10,rtol=0):
                    directories.append(route_root/'pair-routes'/f'pair-{pair["index"]:02d}')
        if not directories:directories=[route_root/'routes']
        for path in sorted({p for directory in directories for p in directory.glob('*.json')}):
            if path.stem==row['id']:continue
            previous=json.loads(path.read_text())
            if not previous.get('passed')or not np.allclose(previous['initial_'+side+'_q_rad'],start,atol=1e-10,rtol=0):continue
            sequence=[start];continuous=True
            for segment in previous.get('initial_transition_segments',[])+previous['segments']:
                if segment['side']!=side:continue
                knots=np.asarray(segment['knots_q_rad'])
                if not np.allclose(sequence[-1],knots[0],atol=1e-10,rtol=0):continuous=False;break
                sequence.extend(knots[1:])
            if continuous:
                condition=previous['condition']
                distance=(condition['pitch_deg']-row['pitch_deg'])**2+(condition['roll_deg']-row['roll_deg'])**2
                terminal=np.asarray(previous['terminal'][side]['q_rad'])
                # A validated wrist turn can be useful across opening stages.
                # Every transferred edge is rechecked in the target scene.
                score=(int(terminal[4]*target[4]<0),float(np.linalg.norm(terminal-target)),distance)
                neighbors.append((score,sequence))
        proposals.extend(sequence for _,sequence in sorted(neighbors,key=lambda v:v[0])[:3])
    if old_path.exists():
        old = json.loads(old_path.read_text())
        old_q = np.asarray(old['route']['raw_q_rad'])
        old_q = old_q[piecewise_knots(old_q)]
        proposals.append(old_q)
    for proposal in proposals:
        candidate=[start]
        for q in proposal:
            if np.max(abs(candidate[-1]-q))<1e-10:continue
            if time.monotonic()>=warm_deadline or not edge(candidate[-1],q,warm_deadline):break
            candidate.append(np.asarray(q))
        if edge(candidate[-1],target,warm_deadline):
            return dict(passed=True,stage='prior_approach_contact_extension',raw_q=[q.tolist()for q in candidate]+[target.tolist()])
        if np.linalg.norm(candidate[-1]-target)<np.linalg.norm(prefix[-1]-target):prefix=candidate
    trace('warm_prefix_complete',knots=len(prefix))
    sign = 1 if side == 'left' else -1
    directions = [-R[:, 0],R[:, 0],sign*lateral,-R[:, 0]+sign*lateral,R[:, 0]+sign*lateral,-R[:, 0]+np.array([0.,1.,0.])]
    if retreat_hint is not None:
        directions.insert(0,np.asarray(retreat_hint))
    best_retreat = [target]
    for direction in directions:
        if np.linalg.norm(direction)<1e-8:
            continue
        direction = direction/np.linalg.norm(direction)
        retreat = [target]
        for distance in np.arange(.001, .121, .001):
            if time.monotonic()-began >= seconds*.50:
                break
            solved = comfortable.solve(p+distance*direction, R, retreat[-1])
            q = np.asarray(solved['q'])
            if solved['position_error_m'] > .0001 or solved['orientation_error_deg'] > .5 or not edge(retreat[-1], q,began+seconds*.50):
                break
            retreat.append(q)
            # Connect from several previously clear approach points, not just
            # a point that might sit in the narrow rim pocket itself.
            if round(distance*1000) in (10,20,40,60,80,120):
                indices=sorted({0,len(prefix)//2,len(prefix)-1},reverse=True)
                if distance<.020:
                    indices=[i for i in indices if np.max(abs(prefix[i]-q))<np.deg2rad(15)]
                for index in indices:
                    if edge(prefix[index],q,began+seconds*.50):
                        raw=prefix[:index+1]+retreat[::-1]
                        return dict(passed=True,stage='contact_retreat_connection',raw_q=[v.tolist() for v in raw])
        if np.linalg.norm(model.pose(retreat[-1])[0]-p)>np.linalg.norm(model.pose(best_retreat[-1])[0]-p):
            best_retreat = retreat
    trace('fixed_retreat_complete',distance_m=float(np.linalg.norm(model.pose(best_retreat[-1])[0]-p)))
    if np.linalg.norm(model.pose(best_retreat[-1])[0]-p)>=.010:
        remaining=min(seconds*.30,seconds-(time.monotonic()-began))
        if remaining>0:
            connected=rrt(model,checker,start,best_retreat[-1],gap,delta,iterations=1500,
                          timeout_s=remaining,joint_margin_deg=joint_margin_deg,
                          shortcut_iterations=12,adaptive_extension=True)
            if connected['passed']:
                raw=list(np.asarray(connected['raw_q']))+best_retreat[-2::-1]
                return dict(passed=True,stage='separated_retreat_rrt_contact_route',raw_q=[q.tolist()for q in raw])
    trace('early_connection_complete')
    # A fixed tool attitude can trap the wrist against the shoulder or rim.
    # Free the attitude while preserving each Cartesian retreat position and
    # enforcing every registered collision inequality, including shoulders.
    for direction in directions[:3]:
        if np.linalg.norm(direction)<1e-8:
            continue
        direction = direction/np.linalg.norm(direction)
        retreat = [target]
        for distance in np.arange(.002, .081, .002):
            if time.monotonic()-began >= seconds*.90:
                break
            desired = p+distance*direction
            previous = retreat[-1]
            cached_constraints = {}
            def constraints(q):
                if time.monotonic()-began >= seconds*.90:
                    raise SearchDeadline
                if 'q' not in cached_constraints or not np.array_equal(cached_constraints['q'],q):
                    values, jacobian = checker.slacks(q,gap,delta,with_jacobian=True)
                    # Exact zero-distance cloth/skin contact is rejected by
                    # the path gate. Keep a 10 um buffer for zero-margin pairs;
                    # static obstacle-floor constraints must remain at zero.
                    buffer=np.r_[[.00001 if pair[2]<=0 else 0. for pair in checker.base.pairs],
                                 np.zeros(len(checker.base.floor_minima))]
                    cached_constraints.update(q=q.copy(),values=values-buffer,jacobian=jacobian[:,:7])
                return cached_constraints['values'],cached_constraints['jacobian']
            try:
                fitted = minimize(lambda q:np.sum((q-previous)**2)+.0001*np.sum((q-start)**2),previous,
                    jac=lambda q:2*(q-previous)+.0002*(q-start),method='SLSQP',
                    bounds=list(zip(comfortable.lower,comfortable.upper)),
                    constraints=[dict(type='eq',fun=lambda q:model.pose(q)[0]-desired,jac=lambda q:model.fast_pose(q)[2]),
                                 dict(type='ineq',fun=lambda q:constraints(q)[0],jac=lambda q:constraints(q)[1])],
                    options=dict(maxiter=30,ftol=1e-10))
            except SearchDeadline:
                break
            q = fitted.x
            if np.linalg.norm(model.pose(q)[0]-desired) > .0001 or not edge(previous,q,began+seconds*.90):
                break
            retreat.append(q)
            if round(distance*1000) in (10,20,40,60,80):
                indices=sorted({len(prefix)//2,len(prefix)-1},reverse=True)
                if distance<.020:
                    indices=[i for i in indices if np.max(abs(prefix[i]-q))<np.deg2rad(15)]
                for index in indices:
                    if edge(prefix[index],q,began+seconds*.90):
                        raw = prefix[:index+1]+retreat[::-1]
                        return dict(passed=True,stage='constrained_orientation_retreat',raw_q=[v.tolist() for v in raw])
        if np.linalg.norm(model.pose(retreat[-1])[0]-p)>np.linalg.norm(model.pose(best_retreat[-1])[0]-p):
            best_retreat=retreat
    remaining = seconds-(time.monotonic()-began)
    trace('constrained_retreat_complete',distance_m=float(np.linalg.norm(model.pose(best_retreat[-1])[0]-p)))
    if remaining <= 0:
        return dict(passed=False, stage='contact_connection_budget')
    # Grow the RRT toward a separated retreat point whenever one was found.
    reached = best_retreat[-1]
    connected = rrt(model,checker,start,reached,gap,delta,iterations=1500,timeout_s=remaining,joint_margin_deg=joint_margin_deg,shortcut_iterations=12,adaptive_extension=True)
    if connected['passed']:
        raw = list(np.asarray(connected['raw_q']))+best_retreat[-2::-1]
        return dict(passed=True, stage='retreat_rrt_contact_route', raw_q=[q.tolist() for q in raw])
    return connected


def route_one(payload):
    row, out, initial_path, pair, seconds, joint_margin_deg = payload
    print('PAIRED PLANNING', pair['index'], row['id'], flush=True)
    initial = yaml.safe_load(initial_path.read_text())
    for side in ('left', 'right'):
        initial[side+'_q_rad'] = pair[side]['q_rad']
    cached_path = out/'pair-routes'/f'pair-{pair["index"]:02d}'/f'{row["id"]}.json'
    if cached_path.exists():
        cached = json.loads(cached_path.read_text())
        if (cached.get('passed') and cached['condition'] == row and cached['protocol']['criteria'] == protocol()['criteria']
            and cached.get('joint_limit_margin_deg',5.) >= joint_margin_deg
            and cached['geometry_manifest_sha256'] == sha256(REGISTERED_MANIFEST)
            and np.array_equal(cached['translation_m'], initial['translation_m'])
            and all(np.array_equal(cached['initial_'+s+'_q_rad'], initial[s+'_q_rad']) for s in ('left', 'right'))):
            # The clarified foot-to-calf domain includes the earlier foot-only
            # contacts. Recheck the new goal predicate before retaining the
            # unchanged, already checked collision path.
            goal = ContactGoal(load_state(ROOT, row), RightHand(URDF))
            checks = {}
            for side, endpoint in cached['terminal'].items():
                c = endpoint['contact_check']
                checks[side] = goal.inspect(np.asarray(c['position_m']), np.asarray(c['rotation']), endpoint['opening_gap_m'], side, c['cloth_finger'])
            if all(c['passed'] for c in checks.values()):
                cached.update(protocol=protocol(), final_contact_checks=checks, clarified_contact_region_independently_rechecked=True)
                write(cached_path, cached)
                if not cached_path.with_suffix('.csv').exists():
                    export_route_csv(cached_path.with_suffix('.csv'), initial, cached)
                return dict(id=row['id'], pair_index=pair['index'], passed=True, reused=True)
    if pair.get('initial_bridge'):
        anchor = out/'pair-routes'/f'pair-{pair["initial_bridge"]["anchor_pair_index"]:02d}'/f'{row["id"]}.json'
        if anchor.exists():
            previous_route = json.loads(anchor.read_text())
            if (previous_route.get('passed') and previous_route['condition'] == row and previous_route['geometry_manifest_sha256'] == sha256(REGISTERED_MANIFEST)
                and previous_route.get('joint_limit_margin_deg',5.) >= joint_margin_deg
                and np.array_equal(previous_route['translation_m'], initial['translation_m'])
                and all(np.array_equal(previous_route['initial_'+s+'_q_rad'], pair['initial_bridge']['anchor_'+s+'_q_rad']) for s in ('left','right'))):
                goal = ContactGoal(load_state(ROOT, row), RightHand(URDF))
                checks = {}
                for side, endpoint in previous_route['terminal'].items():
                    c = endpoint['contact_check']
                    checks[side] = goal.inspect(np.asarray(c['position_m']), np.asarray(c['rotation']), endpoint['opening_gap_m'], side, c['cloth_finger'])
                if all(c['passed'] for c in checks.values()):
                    inherited = copy.deepcopy(previous_route)
                    inherited.update(pair_index=pair['index'], protocol=protocol(),
                        joint_limit_margin_deg=min(previous_route.get('joint_limit_margin_deg',5.), joint_margin_deg),
                        initial_left_q_rad=initial['left_q_rad'], initial_right_q_rad=initial['right_q_rad'],
                        initial_transition_segments=pair['initial_bridge']['segments']+previous_route.get('initial_transition_segments', []),
                        initial_transition_all_70_union_checked=True, final_contact_checks=checks,
                        reused_approach_source=str(anchor.relative_to(out)),
                        jaw_adjustment_location='Separated anchor common start, after the checked open-hand initial transition')
                    write(cached_path, inherited)
                    export_route_csv(cached_path.with_suffix('.csv'), initial, inherited)
                    print('BRIDGED PAIRED ROUTE', pair['index'], row['id'], True, flush=True)
                    return dict(id=row['id'], pair_index=pair['index'], passed=True, initial_transition=True)
    state = load_state(ROOT, row)
    k, base, *_ = context(row['number'], state, REGISTERED_MANIFEST,
                         left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
    base.use_spatial_aabb = True
    delta = np.asarray(initial['translation_m'])
    goal = ContactGoal(state, k.hand)
    record = json.loads((out/'endpoints'/f'{row["id"]}.json').read_text())
    accepted = {}
    for side in ('left','right'):
        start=np.asarray(initial[side+'_q_rad'])
        def priority(endpoint):
            q=np.asarray(endpoint['q_rad'])
            # The current shoulder guard separates the right joint-1 regions
            # at 110 degrees (see shoulder-component-barrier.json). This is a
            # search ordering only: no geometric contact candidate is removed.
            crossing=side=='right'and (q[0]-np.deg2rad(110.))*(start[0]-np.deg2rad(110.))<0
            return (int(crossing),float(np.linalg.norm(q-start)),endpoint_priority(endpoint))
        accepted[side]=sorted(record['sides'][side]['accepted'],key=priority)
    result = dict(condition=row, pair_index=pair['index'], passed=False, protocol=protocol(),joint_limit_margin_deg=joint_margin_deg,
                  initial_right_q_rad=initial['right_q_rad'], initial_left_q_rad=initial['left_q_rad'],
                  translation_m=delta.tolist(), attempts=[], geometry_manifest_sha256=sha256(REGISTERED_MANIFEST))
    original_pairs = list(base.pairs)
    began = time.monotonic()
    for left in accepted['left']:
        for right in accepted['right']:
            if time.monotonic()-began > seconds*4:
                break
            terminal = copy.deepcopy(dict(left=left, right=right))
            base.pairs = list(original_pairs)
            contact_scene(base, {side: terminal[side]['contact_check']['foot_finger'] for side in terminal})
            final_checks = {}
            for side in terminal:
                c = terminal[side]['contact_check']
                model = ControlledArmKinematics(k, side, right['q_rad'], right['opening_gap_m'])
                p, R = model.pose(terminal[side]['q_rad'])
                final_checks[side] = goal.inspect(p-delta, R, terminal[side]['opening_gap_m'], side, c['cloth_finger'])
            if not all(c['passed'] for c in final_checks.values()):
                result['attempts'].append(dict(stage='independent_contact_recheck', checks=final_checks))
                continue
            base.park_left(left['q_rad'], left['opening_gap_m'])
            final_scene = base.slacks(right['q_rad'], right['opening_gap_m'], delta, details=True)
            if not final_scene['passed']:
                result['attempts'].append(dict(stage='simultaneous_endpoint_collision', check=final_scene))
            # Set the open widths while still at the separated shared start.
            # This preparation is never a final closing / grasp test.
            jaw_checks = []
            for side in ('left', 'right'):
                for width in np.linspace(initial[side+'_gap_m'], terminal[side]['opening_gap_m'], 21):
                    lg = float(width) if side == 'left' else left['opening_gap_m']
                    rg = initial['right_gap_m'] if side == 'left' else float(width)
                    base.park_left(initial['left_q_rad'], lg)
                    if not base.is_clear(initial['right_q_rad'], rg, delta):
                        jaw_checks.append(dict(side=side, width_m=float(width), check=base.slacks(initial['right_q_rad'], rg, delta, details=True)))
                        break
                if jaw_checks:
                    break
            if jaw_checks:
                result['attempts'].append(dict(stage='initial_open_width_adjustment', failures=jaw_checks))
                continue
            for order in (('right', 'left'), ('left', 'right')):
                terminal = copy.deepcopy(dict(left=left, right=right))
                current = {s: np.asarray(initial[s+'_q_rad']) for s in terminal}
                segments, failures = [], []
                for side in order:
                    other = 'right' if side == 'left' else 'left'
                    base.park_left(current['left'], left['opening_gap_m'])
                    model = ControlledArmKinematics(k, side, current['right'], right['opening_gap_m'])
                    checker = ControlledCollisionScene(base, side, current['right'], right['opening_gap_m'])
                    target = np.asarray(terminal[side]['q_rad'])
                    # --seconds is a per-arm search budget. Verification of a
                    # found left path must not consume the right arm's budget.
                    budget = seconds
                    cloth_contact=np.mean(terminal[side]['contact_check']['contacts']['cloth']['witnesses_m'],axis=0)
                    hint=opening_retreat_hint(goal.rim,cloth_contact)
                    planned = plan_contact_route(model, checker, current[side], target, terminal[side]['opening_gap_m'], delta,
                                                 row, side, goal.lateral, budget, joint_margin_deg,hint,route_root=out)
                    if not planned['passed']:
                        failures.append(dict(side=side, stage=planned['stage']))
                        break
                    actual=np.asarray(planned['raw_q'][-1])
                    p,R=model.pose(actual)
                    actual_contact=goal.inspect(p-delta,R,terminal[side]['opening_gap_m'],side,
                                                terminal[side]['contact_check']['cloth_finger'])
                    if not actual_contact['passed']:
                        failures.append(dict(side=side,stage='actual_route_terminal_contact',check=actual_contact))
                        break
                    terminal[side].update(q_rad=actual.tolist(),tcp_world_m=p.tolist(),rotation=R.tolist(),contact_check=actual_contact)
                    profile = linear_quintic_profile(planned['raw_q'], model.velocity, model.acceleration)
                    # Exact polylines avoid smoothing into cloth or skin. Check
                    # independent time samples, plus a 0.25 degree joint grid.
                    grid = np.unique(np.r_[profile['u'], (profile['u'][:-1]+profile['u'][1:])/2])
                    # The generated profile already subdivides every knot
                    # interval at <=0.25 degrees. Midpoints halve that bound;
                    # duplicating the dense planner grid adds no coverage.
                    dense = profile['curve'](grid)
                    valid = all(checker.is_clear(q, terminal[side]['opening_gap_m'], delta) for q in dense)
                    if not valid:
                        failures.append(dict(side=side, stage='independent_trajectory_collision'))
                        break
                    segments.append(dict(side=side, fixed_other_q_rad=current[other].tolist(),
                                         planning_method=planned['stage'],
                                         raw_q_rad=planned['raw_q'], knots_q_rad=profile['knots_q_rad'],
                                         segment_durations_s=profile['segment_durations_s'],
                                         q_rad=profile['q'].tolist(), times_s=profile['t'].tolist(),
                                         velocity_rad_s=profile['velocity'].tolist(), acceleration_rad_s2=profile['acceleration'].tolist(),
                                         duration_s=profile['duration'], opening_gap_m=terminal[side]['opening_gap_m'],
                                         independent_collision_samples=len(dense), continuous_collision_certified=False))
                    current[side] = actual
                if failures:
                    result['attempts'].append(dict(order=list(order), failures=failures))
                    continue
                final_checks={s:terminal[s]['contact_check'] for s in terminal}
                base.park_left(current['left'],left['opening_gap_m'])
                final_scene=base.slacks(current['right'],right['opening_gap_m'],delta,details=True)
                if not final_scene['passed']:
                    result['attempts'].append(dict(order=list(order),stage='actual_simultaneous_endpoint_collision',check=final_scene))
                    continue
                result.update(passed=True, order=list(order), segments=segments, terminal=terminal,
                              final_contact_checks=final_checks, simultaneous_endpoint_check=final_scene,
                              initial_open_width_adjustment_passed=True,
                              jaw_adjustment_location='Separated shared start, before either arm approaches',
                              grasp_closure_performed=False, elapsed_s=time.monotonic()-began)
                export_route_csv(out/'pair-routes'/f'pair-{pair["index"]:02d}'/f'{row["id"]}.csv', initial, result)
                write(out/'pair-routes'/f'pair-{pair["index"]:02d}'/f'{row["id"]}.json', result)
                print('PAIRED ROUTE', pair['index'], row['id'], True, flush=True)
                return dict(id=row['id'], pair_index=pair['index'], passed=True)
    result['elapsed_s'] = time.monotonic()-began
    write(out/'pair-routes'/f'pair-{pair["index"]:02d}'/f'{row["id"]}.json', result)
    return dict(id=row['id'], pair_index=pair['index'], passed=False)


def export_route_csv(path, initial, result):
    """Both arms and both widths on one clock; include preparation at start."""
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {side: np.asarray(initial[side+'_q_rad']) for side in ('left', 'right')}
    widths = {side: float(initial[side+'_gap_m']) for side in current}
    names = ['time_s', 'stage']+[f'{s}_joint_{i}_rad' for s in ('right', 'left') for i in range(1, 8)]+['right_gap_m', 'left_gap_m']
    with path.open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        def emit(t, stage):
            writer.writerow([t, stage]+current['right'].tolist()+current['left'].tolist()+[widths['right'], widths['left']])
        emit(0., 'initial')
        offset = 0.
        for segment in result.get('initial_transition_segments', []):
            side = segment['side']
            for t, q in zip(segment['times_s'][1:], segment['q_rad'][1:]):
                current[side] = np.asarray(q)
                emit(offset+t, 'initial_transition_'+side)
            offset += segment['duration_s']
        for side in ('left', 'right'):
            before = widths[side]
            after = result['terminal'][side]['opening_gap_m']
            duration = max(.2, 1.875*abs(after-before)/.01)
            for tau in np.linspace(0, 1, max(2, int(np.ceil(duration/.02))+1))[1:]:
                widths[side] = before+(after-before)*(10*tau**3-15*tau**4+6*tau**5)
                emit(offset+tau*duration, 'prepare_'+side+'_open_width')
            offset += duration
        for segment in result['segments']:
            side = segment['side']
            for t, q in zip(segment['times_s'][1:], segment['q_rad'][1:]):
                current[side] = np.asarray(q)
                emit(offset+t, 'approach_'+side)
            offset += segment['duration_s']


def routes(args, rows):
    out = args.output.resolve()
    initial_path = args.initial.resolve()
    reference = yaml.safe_load(initial_path.read_text())
    for row in rows:
        path=out/'endpoints'/f'{row["id"]}.json'
        d=json.loads(path.read_text())
        for values in d['sides'].values():
            values['accepted'].sort(key=endpoint_priority)
        write(path,d)
    source = ROOT/'artifacts/common-grasp-start-search/start-candidates/start-search.json'
    candidates = json.loads(source.read_text())['candidate_pairs']
    pairs = [dict(index=i, left=p['left'], right=p['right']) for i, p in enumerate(candidates)]
    additional=out/'additional-start-pairs.json'
    # The old accepted list stopped after 24 right-hand variants of one left
    # start. Explicitly explore different LEFT starts as well as right starts.
    pool_paths = {side: source.parent/f'{side}-start-candidates.json' for side in ('left', 'right')}
    pools = {side: json.loads(path.read_text())['candidates'] for side, path in pool_paths.items()}
    for li in range(0, len(pools['left']), max(1, len(pools['left'])//8)):
        for ri in range(0, len(pools['right']), max(1, len(pools['right'])//4)):
            left, right = pools['left'][li], pools['right'][ri]
            if any(np.linalg.norm(np.asarray(left['q_rad'])-p['left']['q_rad']) < 1e-8 and
                   np.linalg.norm(np.asarray(right['q_rad'])-p['right']['q_rad']) < 1e-8 for p in pairs):
                continue
            pairs.append(dict(index=len(pairs), left=left, right=right,
                              source_left_candidate_index=li, source_right_candidate_index=ri))
    if additional.exists():
        for i,pair in enumerate(json.loads(additional.read_text())['candidate_pairs']):
            pairs.append(dict(index=1000+i,left=pair['left'],right=pair['right'],source='wide_radial'))
    from sock_dressing_simulation.common_opening_approach import union_scene
    considered_pair_count = len(pairs)
    _, union, _ = union_scene(ROOT, inventory(ROOT), reference)
    valid_pairs = []
    for pair in pairs:
        union.park_left(pair['left']['q_rad'], reference['left_gap_m'])
        check = union.slacks(pair['right']['q_rad'], reference['right_gap_m'], reference['translation_m'], details=True)
        if check['passed']:
            pair['all_70_initial_union_check'] = check
            valid_pairs.append(pair)
    pairs = valid_pairs
    if not pairs:
        raise ValueError('No common start candidate passes the fresh all-70 union screen')
    # Rank starts by aggregate distance to accepted contact endpoints on both
    # arms. Every candidate is still checked with the actual paired scenes.
    for pair in pairs:
        score = 0.
        for row in rows:
            path = out/'endpoints'/f'{row["id"]}.json'
            if not path.exists():
                raise ValueError('Missing contact endpoint search: '+str(path))
            d = json.loads(path.read_text())
            for side in ('left', 'right'):
                goals = d['sides'].get(side, {}).get('accepted', [])
                if goals:
                    score += min(float(np.linalg.norm(np.asarray(g['q_rad'])-pair[side]['q_rad'])) for g in goals)
        pair['distance_score'] = score
        pair['previous_passed_states'] = 0
        for path in (out/'pair-routes'/f'pair-{pair["index"]:02d}').glob('*.json'):
            old = json.loads(path.read_text())
            if (old.get('passed') and old['condition'] in rows and
                np.array_equal(old['translation_m'], reference['translation_m']) and
                all(np.array_equal(old['initial_'+s+'_q_rad'], pair[s]['q_rad']) for s in ('left','right'))):
                pair['previous_passed_states'] += 1
    pairs.sort(key=lambda p: (-p['previous_passed_states'], p['distance_score']))
    # Evaluate distinct left starts first; score alone tends to pick several
    # neighbouring right starts while keeping the left hand unchanged.
    selected_pairs = []
    if additional.exists() and any(pair.get('source')=='wide_radial' for pair in pairs):
        selected_pairs.append(pairs[0])
        wide=sorted((p for p in pairs if p.get('source')=='wide_radial'),
            key=lambda p:(min(p[s].get('minimum_joint_margin_deg',0.)for s in ('left','right'))<8.,p['distance_score']))
        for pair in wide:
            if len(selected_pairs)>=args.pairs:break
            if not any(np.linalg.norm(np.asarray(pair['left']['q_rad'])-p['left']['q_rad'])<.1 for p in selected_pairs):
                selected_pairs.append(pair)
            if len(selected_pairs)>=args.pairs:break
    for pair in pairs:
        if len(selected_pairs)>=args.pairs:break
        if not any(np.linalg.norm(np.asarray(pair['left']['q_rad'])-p['left']['q_rad']) < .1 for p in selected_pairs):
            selected_pairs.append(pair)
        if len(selected_pairs) >= args.pairs:
            break
    for pair in pairs:
        if len(selected_pairs) >= args.pairs:
            break
        if pair['index'] not in {p['index'] for p in selected_pairs}:
            selected_pairs.append(pair)
    write(out/'initial-union-screen.json', dict(candidates=valid_pairs, considered_pairs=considered_pair_count,
          shared_state_count=70, side_pools={side:dict(source=str(path.relative_to(ROOT)), sha256=sha256(path), count=len(pools[side])) for side,path in pool_paths.items()}))
    results = []
    for pair in selected_pairs:
        if results:
            anchor = max(results, key=lambda r:r['passed_states'])['pair']
            # Prove a universal open-hand transition between the two common
            # starts once against the full 70-state union. This permits exact
            # reuse of an anchor approach without 70 duplicate path checks.
            for order in (('left', 'right'), ('right', 'left')):
                current = {side:np.asarray(pair[side]['q_rad']) for side in ('left', 'right')}
                bridge, valid = [], True
                for side in order:
                    target = np.asarray(anchor[side]['q_rad'])
                    bridge_model = ControlledArmKinematics(union.k, side, current['right'], reference['right_gap_m'])
                    profile = linear_quintic_profile([current[side], target], bridge_model.velocity, bridge_model.acceleration) if np.linalg.norm(current[side]-target) > 1e-8 else None
                    if profile is None:
                        continue
                    for q in edge_samples(current[side], target, .125):
                        left = q if side == 'left' else current['left']
                        right = q if side == 'right' else current['right']
                        union.park_left(left, reference['left_gap_m'])
                        if not union.is_clear(right, reference['right_gap_m'], reference['translation_m']):
                            valid = False
                            break
                    if not valid:
                        break
                    bridge.append(dict(side=side, q_rad=profile['q'].tolist(), times_s=profile['t'].tolist(), duration_s=profile['duration'],
                                       knots_q_rad=profile['knots_q_rad'], segment_durations_s=profile['segment_durations_s']))
                    current[side] = target
                if valid:
                    pair['initial_bridge'] = dict(anchor_pair_index=anchor['index'], segments=bridge, checked_states=70,
                                                  anchor_left_q_rad=anchor['left']['q_rad'], anchor_right_q_rad=anchor['right']['q_rad'],
                                                  both_gaps_m=[reference['right_gap_m'], reference['left_gap_m']], maximum_joint_step_deg=.125)
                    break
        jobs = []
        for row in rows:
            endpoints = json.loads((out/'endpoints'/f'{row["id"]}.json').read_text())
            if not all(endpoints['sides'].get(s, {}).get('accepted', []) for s in ('left', 'right')):
                continue
            jobs.append((row, out, initial_path, pair, args.seconds,args.joint_margin_deg))
        pair_results = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(route_one, job) for job in jobs]):
                pair_results.append(future.result())
        results.append(dict(pair=pair, passed_states=sum(r['passed'] for r in pair_results), evaluated_states=len(jobs), rows=pair_results))
        write(out/'initial-search.json', dict(candidates=results, reference_initial_sha256=sha256(initial_path),
                                             source_start_candidates_sha256=sha256(source), required_states=70))
        if len(rows) == 70 and len(pair_results) == 70 and all(r['passed'] for r in pair_results):
            break
    select_routes(out,rows,reference,results,union.k,args.joint_margin_deg)


def best_route_candidate(results,kinematics):
    for result in results:
        pair=result['pair']
        for side in ('left','right'):
            model=ControlledArmKinematics(kinematics,side,pair['right']['q_rad'],.04)
            q=np.asarray(pair[side]['q_rad'])
            pair[side]['minimum_joint_margin_deg']=float(np.rad2deg(np.minimum(q-model.lower,model.upper-q)).min())
    return max(results,key=lambda r:(r['passed_states'],min(r['pair'][s]['minimum_joint_margin_deg']for s in ('left','right'))))


def select_routes(out,rows,reference,results,kinematics,joint_margin_deg):
    """Publish one shared configuration and only that configuration's paths."""
    chosen = best_route_candidate(results,kinematics)
    pair = chosen['pair']
    config = dict(reference)
    for side in ('left', 'right'):
        config[side+'_q_rad'] = pair[side]['q_rad']
        config[side+'_q_deg'] = np.rad2deg(pair[side]['q_rad']).tolist()
        model=ControlledArmKinematics(kinematics,side,config['right_q_rad'],config['right_gap_m'])
        p,R=model.pose(pair[side]['q_rad'])
        config[side+'_tcp_world_m'] = p.tolist()
        config[side+'_rotation'] = R.tolist()
    config.update(initialization='Open cloth/foot contact task; candidate ranked by paired validated route coverage',
                  reachability_verified=bool(chosen['passed_states'] == 70 and len(rows) == 70),
                  verified_paired_state_count=chosen['passed_states'], required_paired_state_count=70,
                  contact_tolerance_m=.0005, closure_performed=False,
                  source_candidate_pair_index=pair['index'],
                  source_reference=str((out/'initial-search.json').relative_to(ROOT)),
                  verification_protocol=str((out/'protocol.json').relative_to(ROOT)),
                  native_dynamics_verified=False,continuous_collision_certified=False)
    config.pop('right_start_candidate_index',None)
    config['joint_limit_margin_required_deg'] = joint_margin_deg
    margins = []
    for side in ('left', 'right'):
        model = ControlledArmKinematics(kinematics, side, config['right_q_rad'], config['right_gap_m'])
        q = np.asarray(config[side+'_q_rad'])
        margins.append(float(np.rad2deg(np.minimum(q-model.lower, model.upper-q)).min()))
    config['minimum_joint_margin_deg'] = min(margins)
    (out/'candidate-initial.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    import shutil
    (out/'routes').mkdir(exist_ok=True)
    for row in rows:
        path = out/'pair-routes'/f'pair-{pair["index"]:02d}'/f'{row["id"]}.json'
        if path.exists():
            shutil.copyfile(path, out/'routes'/path.name)
            csv_path = path.with_suffix('.csv')
            if csv_path.exists():
                shutil.copyfile(csv_path, out/'routes'/csv_path.name)
        else:
            write(out/'routes'/f'{row["id"]}.json', dict(condition=row, pair_index=pair['index'], passed=False,
                  reason='No paired contact endpoints; route not evaluated', protocol=protocol()))
            csv_path = out/'routes'/f'{row["id"]}.csv'
            if csv_path.exists():
                csv_path.unlink()
    report(out, rows)


def collect_routes(args,rows):
    """Recount completed parallel searches without repeating failed searches."""
    from sock_dressing_simulation.common_opening_approach import union_scene
    out=args.output.resolve();reference=yaml.safe_load(args.initial.read_text())
    path=out/'initial-search.json';search=json.loads(path.read_text())
    if search['reference_initial_sha256']!=sha256(args.initial):
        raise ValueError('Changed initial reference')
    for candidate in search['candidates']:
        pair=candidate['pair'];records=[]
        for row in rows:
            route_path=out/'pair-routes'/f'pair-{pair["index"]:02d}'/f'{row["id"]}.json'
            route=json.loads(route_path.read_text())if route_path.exists()else{}
            passed=bool(route.get('passed'))
            if passed:
                valid=(route['condition']==row and route['protocol']==protocol() and
                       route['geometry_manifest_sha256']==sha256(REGISTERED_MANIFEST) and
                       route.get('joint_limit_margin_deg',5.)>=args.joint_margin_deg-1e-9 and
                       np.array_equal(route['translation_m'],reference['translation_m']) and
                       all(np.array_equal(route['initial_'+s+'_q_rad'],pair[s]['q_rad'])for s in ('left','right')))
                if not valid:raise ValueError('Stale successful route: '+str(route_path))
            records.append(dict(id=row['id'],pair_index=pair['index'],passed=passed))
        candidate.update(rows=records,passed_states=sum(r['passed']for r in records),evaluated_states=len(records))
    k,union,_=union_scene(ROOT,inventory(ROOT),reference)
    chosen=best_route_candidate(search['candidates'],k)
    pair=chosen['pair'];union.park_left(pair['left']['q_rad'],reference['left_gap_m'])
    check=union.slacks(pair['right']['q_rad'],reference['right_gap_m'],reference['translation_m'],details=True)
    if not check['passed']:raise ValueError('Selected start fails the full state union')
    pair['all_70_initial_union_check']=check
    write(path,search)
    select_routes(out,rows,reference,search['candidates'],k,args.joint_margin_deg)


def endpoint_one(payload):
    row, out, initial_path, seconds, maximum, joint_margin_deg = payload[:6]
    right_joint_1_min_deg=payload[6]if len(payload)>6 else None
    right_elbow_min_deg=payload[7]if len(payload)>7 else None
    right_wrist_max_deg=payload[8]if len(payload)>8 else None
    initial = yaml.safe_load(initial_path.read_text())
    state = load_state(ROOT, row)
    k, base, *_ = context(row['number'], state, REGISTERED_MANIFEST, left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
    base.use_spatial_aabb = True
    delta = np.asarray(initial['translation_m'])
    base.park_left(initial['left_q_rad'], initial['left_gap_m'])
    goal = ContactGoal(state, k.hand)
    source = json.loads((out/'geometry'/f'{row["id"]}.json').read_text())
    result = dict(condition=row, initial_sha256=sha256(initial_path), geometry_manifest_sha256=sha256(REGISTERED_MANIFEST), protocol=protocol(),joint_limit_margin_deg=joint_margin_deg, sides={})
    prior_path = out/'endpoints'/f'{row["id"]}.json'
    previous = json.loads(prior_path.read_text()) if prior_path.exists() else None
    if previous and (previous['condition'] != row or previous['initial_sha256'] != sha256(initial_path) or previous['geometry_manifest_sha256'] != sha256(REGISTERED_MANIFEST)):
        raise ValueError('Stale endpoint cache: '+str(prior_path))
    if previous:
        result['sides'] = copy.deepcopy(previous['sides'])
    result['search_round'] = (previous or {}).get('search_round', 0)+1
    original_pairs = list(base.pairs)
    for side in ('left', 'right'):
        began = time.monotonic()
        model = ControlledArmKinematics(k, side, initial['right_q_rad'], initial['right_gap_m'])
        comfortable = copy.copy(model)
        comfortable.lower = model.lower+np.deg2rad(joint_margin_deg)
        comfortable.upper = model.upper-np.deg2rad(joint_margin_deg)
        if side=='right'and right_joint_1_min_deg is not None:
            comfortable.lower[0]=max(comfortable.lower[0],np.deg2rad(right_joint_1_min_deg))
            if comfortable.lower[0]>=comfortable.upper[0]:raise ValueError('Right joint 1 search lower bound exceeds its upper limit')
            result['right_joint_1_search_lower_bound_deg']=right_joint_1_min_deg
        if side=='right'and right_elbow_min_deg is not None:
            comfortable.lower[3]=max(comfortable.lower[3],np.deg2rad(right_elbow_min_deg))
            if comfortable.lower[3]>=comfortable.upper[3]:raise ValueError('Right elbow search lower bound exceeds its upper limit')
            result['right_elbow_search_lower_bound_deg']=right_elbow_min_deg
        if side=='right'and right_wrist_max_deg is not None:
            comfortable.upper[4]=min(comfortable.upper[4],np.deg2rad(right_wrist_max_deg))
            if comfortable.lower[4]>=comfortable.upper[4]:raise ValueError('Right wrist search upper bound is below its lower limit')
            result['right_wrist_search_upper_bound_deg']=right_wrist_max_deg
        rng = np.random.default_rng(20261113+row['number']+10000*(result['search_round']-1))
        seeds = [np.asarray(initial[side+'_q_rad'])]
        # Existing routes supply reachable joint branches, not contact passes.
        path = ROOT/'artifacts/common-opening-approach/negative-wrist-start/routes'/side/f'{row["id"]}.json'
        if path.exists():
            old = json.loads(path.read_text())
            if old.get('passed'):
                seeds.insert(0, np.asarray(old['route']['raw_q_rad'][-1]))
        seeds.extend(rng.uniform(comfortable.lower, comfortable.upper, (10, 7)))
        neighbor_direct=[];neighbor_seeds=[];neighbors=[]
        route_neighbors=[]
        for path in (out/'routes').glob('*.json'):
            d=json.loads(path.read_text())
            if not d.get('passed')or side not in d.get('terminal',{}):continue
            other=d['condition'];distance=(other['pitch_deg']-row['pitch_deg'])**2+(other['roll_deg']-row['roll_deg'])**2
            route_neighbors.append(((int(other['stage']!=row['stage']),distance),d))
        for _,d in sorted(route_neighbors,key=lambda value:value[0])[:4]:
            endpoint=d['terminal'][side];seed=np.clip(endpoint['q_rad'],comfortable.lower+1e-9,comfortable.upper-1e-9)
            neighbor_seeds.append(seed);position,attitude=model.pose(seed)
            neighbor_direct.append(dict(position_m=(position-delta).tolist(),rotation=attitude.tolist(),
                opening_gap_m=endpoint['opening_gap_m'],cloth_finger=d['final_contact_checks'][side]['cloth_finger'],seed_q_rad=seed.tolist()))
        for path in (out/'endpoints').glob(row['stage']+'-*.json'):
            if path.stem==row['id']:continue
            d=json.loads(path.read_text());other=d['condition']
            distance=(other['pitch_deg']-row['pitch_deg'])**2+(other['roll_deg']-row['roll_deg'])**2
            neighbors.append((distance,d))
        for _,neighbor in sorted(neighbors,key=lambda item:item[0])[:4]:
            candidates=sorted(neighbor['sides'].get(side,{}).get('accepted',[]),key=endpoint_priority)
            for accepted_neighbor in candidates[:2]:
                seed=np.clip(accepted_neighbor['q_rad'],comfortable.lower+1e-9,comfortable.upper-1e-9)
                neighbor_seeds.append(seed)
                position,attitude=model.pose(seed)
                neighbor_direct.append(dict(position_m=(position-delta).tolist(),rotation=attitude.tolist(),
                    opening_gap_m=accepted_neighbor['opening_gap_m'],cloth_finger=accepted_neighbor['contact_check']['cloth_finger'],seed_q_rad=seed.tolist()))
        seeds=neighbor_seeds+seeds
        proposals = source['sides'][side]['accepted']
        if source['sides'][side]['best']:
            proposals = proposals+[source['sides'][side]['best']]
        alternate_attitudes = []
        for proposal in proposals:
            for flip, swap in ((np.diag([1., -1., -1.]), True), (np.diag([-1., 1., -1.]), False)):
                changed = dict(proposal)
                changed['rotation'] = (np.asarray(proposal['rotation'])@flip).tolist()
                if swap:
                    changed['cloth_finger'] = 'right_finger_link' if proposal['cloth_finger'] == 'left_finger_link' else 'left_finger_link'
                changed['limited_ik_seeds'] = True
                alternate_attitudes.append(changed)
        direct = []
        direct_seeds = [seeds[0]]
        for joint, increment in ((6, -.6), (6, .6), (4, -.6), (4, .6)):
            varied = seeds[0].copy()
            varied[joint] = np.clip(varied[joint]+increment, comfortable.lower[joint]+1e-8, comfortable.upper[joint]-1e-8)
            direct_seeds.append(varied)
        for index, seed in enumerate(direct_seeds):
            position, attitude = model.pose(seed)
            for finger in ('left_finger_link', 'right_finger_link'):
                for gap in ((.012, .030, .048) if index == 0 else (.030,)):
                    direct.append(dict(position_m=(position-delta).tolist(), rotation=attitude.tolist(),
                                       opening_gap_m=gap, cloth_finger=finger, seed_q_rad=seed.tolist()))
        warm = []
        prior_best = previous.get('sides', {}).get(side, {}).get('best') if previous else None
        if prior_best:
            c = prior_best['contact_check']
            for spread in (0., .03, .10, .30):
                seed = np.clip(np.asarray(prior_best['q_rad'])+rng.normal(0, spread, 7), comfortable.lower+1e-9, comfortable.upper-1e-9)
                warm.append(dict(position_m=c['position_m'], rotation=c['rotation'],
                                 opening_gap_m=c['opening_gap_m'], cloth_finger=c['cloth_finger'], seed_q_rad=seed.tolist()))
        # Change wrist branches on subsequent rounds instead of replaying the
        # same deterministic failures with a larger wall-clock budget.
        varied = []
        if previous:
            for index in range(24):
                seed = seeds[0].copy()
                seed[3:] += rng.normal(0, .65, 4)
                seed = np.clip(seed, comfortable.lower+1e-9, comfortable.upper-1e-9)
                position, attitude = model.pose(seed)
                varied.append(dict(position_m=(position-delta).tolist(), rotation=attitude.tolist(),
                                   opening_gap_m=float(rng.uniform(.005, .05)),
                                   cloth_finger=('left_finger_link' if index%2 else 'right_finger_link'), seed_q_rad=seed.tolist()))
        # Visit different contact poses before spending every IK seed on the
        # first wrist configuration. This matters when one pose is folded into
        # the forearm but a different rim location has a clear wrist branch.
        round_robin = []
        for seed in seeds[:4]:
            for proposal in alternate_attitudes+proposals:
                round_robin.append(dict(proposal, ik_seed_q_rad=np.asarray(seed).tolist()))
        proposals = neighbor_direct+round_robin+warm+direct+varied
        cached = previous.get('sides', {}).get(side, {}) if previous else {}
        accepted = list(cached.get('accepted', []))
        accepted = [c for c in accepted if np.all(np.asarray(c['q_rad'])>=comfortable.lower-1e-9) and np.all(np.asarray(c['q_rad'])<=comfortable.upper+1e-9)]
        best = cached.get('best')
        if best:
            best = copy.deepcopy(best)
            c = best['contact_check']
            best['residual_norm_mm'] = float(np.linalg.norm(goal.residual(np.asarray(c['position_m']), np.asarray(c['rotation']), c['opening_gap_m'], side, c['cloth_finger'])))
        counts = Counter()
        counts['previously_verified_endpoints'] = len(accepted)
        try:
            for proposal in proposals:
                if len(accepted) >= maximum:
                    break
                p, R = np.asarray(proposal['position_m'])+delta, np.asarray(proposal['rotation'])
                gap, finger = proposal['opening_gap_m'], proposal['cloth_finger']
                for seed in ([np.asarray(proposal['seed_q_rad'])] if 'seed_q_rad' in proposal else [np.asarray(proposal['ik_seed_q_rad'])] if 'ik_seed_q_rad' in proposal else seeds[:4] if proposal.get('limited_ik_seeds') else seeds):
                    if time.monotonic()-began >= seconds:
                        raise SearchDeadline
                    if 'seed_q_rad' in proposal:
                        q = np.clip(seed, comfortable.lower+1e-9, comfortable.upper-1e-9)
                        counts['direct_fk_seeds'] += 1
                    else:
                        sol = comfortable.solve(p, R, seed)
                        counts['ik_attempts'] += 1
                        q = np.asarray(sol['q'])
                    # Optimize contact in actual reachable FK, instead of
                    # accepting an IK pose merely because it is nearby.
                    def fun(x):
                        if time.monotonic()-began >= seconds:
                            raise SearchDeadline
                        actual, actual_R = model.pose(x[:7])
                        counts['contact_evaluations'] += 1
                        return goal.residual(actual-delta, actual_R, x[7], side, finger)
                    fit = least_squares(fun, np.r_[q, gap], bounds=(np.r_[comfortable.lower, .0001], np.r_[comfortable.upper, .05]),
                                        max_nfev=100, diff_step=2e-5, ftol=1e-7, xtol=1e-7, gtol=1e-6)
                    q, gap_fit = fit.x[:7], float(fit.x[7])
                    actual, actual_R = model.pose(q)
                    check = goal.inspect(actual-delta, actual_R, gap_fit, side, finger)
                    score = float(np.linalg.norm(fit.fun))
                    if best is None or score < best['residual_norm_mm']:
                        best = dict(q_rad=q.tolist(), residual_norm_mm=score, contact_check=check)
                    if not check['passed']:
                        counts['contact_rejected'] += 1
                        continue
                    base.pairs = list(original_pairs)
                    contact_scene(base, {side: check['foot_finger']})
                    base.park_left(initial['left_q_rad'], initial['left_gap_m'])
                    checker = ControlledCollisionScene(base, side, initial['right_q_rad'], initial['right_gap_m'])
                    whole = checker.slacks(q, gap_fit, delta, details=True)
                    if not whole['passed'] and time.monotonic()-began < seconds:
                        # A contact fit can fold the wrist into its own palm.
                        # Repair the recorded blocking pairs jointly with the
                        # contact constraints, then recheck the ENTIRE scene.
                        full_pairs = list(base.pairs)
                        full_shoulders = dict(base.shoulder_pairs)
                        blocking = {tuple(issue['pair']) for issue in whole['issues']}
                        for repair in range(2):
                            retain_blocking_pairs(base, full_pairs, full_shoulders, blocking)
                            repair_deadline = min(began+seconds, time.monotonic()+5.)
                            try:
                                def repair_fun(x):
                                    if time.monotonic()-began >= seconds:
                                        raise SearchDeadline
                                    if time.monotonic() >= repair_deadline:
                                        raise RepairDeadline
                                    contact = fun(x)
                                    slacks = checker.slacks(x[:7], float(x[7]), delta)
                                    return np.r_[contact, 10000*np.minimum(slacks-.00005, 0.)]
                                repaired = least_squares(repair_fun, np.r_[q, gap_fit],
                                    bounds=(np.r_[comfortable.lower, .0001], np.r_[comfortable.upper, .05]),
                                    max_nfev=70, diff_step=2e-5, ftol=1e-7, xtol=1e-7, gtol=1e-6)
                                q, gap_fit = repaired.x[:7], float(repaired.x[7])
                            except RepairDeadline:
                                counts['repair_budget_exhausted'] += 1
                                break
                            finally:
                                base.pairs = full_pairs
                                base.shoulder_pairs = full_shoulders
                                base.fixed_pair_cache = {}
                                base.fixed_pair_valid_cache = {}
                            actual, actual_R = model.pose(q)
                            check = goal.inspect(actual-delta, actual_R, gap_fit, side, finger)
                            whole = checker.slacks(q, gap_fit, delta, details=True)
                            counts['collision_repairs'] += 1
                            if check['passed'] and whole['passed']:
                                break
                            blocking.update(tuple(issue['pair']) for issue in whole['issues'])
                        if not check['passed']:
                            counts['repair_contact_rejected'] += 1
                            continue
                    if not whole['passed']:
                        counts['whole_scene_rejected'] += 1
                        if best and score <= best['residual_norm_mm']:
                            best['whole_scene_check'] = whole
                        continue
                    if any(np.linalg.norm(q-np.asarray(old['q_rad'])) < .1 and abs(gap_fit-old['opening_gap_m']) < .002 for old in accepted):
                        counts['duplicate_endpoint'] += 1
                        continue
                    accepted.append(dict(q_rad=q.tolist(), opening_gap_m=gap_fit, contact_check=check, whole_scene_check=whole,
                                         tcp_world_m=actual.tolist(), rotation=actual_R.tolist()))
                    accepted.sort(key=endpoint_priority)
                    print('CONTACT ENDPOINT', row['id'], side, len(accepted), flush=True)
                    if len(accepted) >= maximum:
                        break
                if len(accepted) >= maximum:
                    break
        except SearchDeadline:
            counts['deadline'] += 1
        result['sides'][side] = dict(accepted=accepted, best=best, counts=dict(counts), elapsed_s=time.monotonic()-began, budget_s=seconds)
        base.pairs = list(original_pairs)
        write(out/'endpoints'/f'{row["id"]}.json', result)
        print('ENDPOINT DONE', row['id'], side, len(accepted), dict(counts), flush=True)
    return dict(id=row['id'], left=len(result['sides']['left']['accepted']), right=len(result['sides']['right']['accepted']))


def report(out, rows):
    data = []
    initial_path = out/'candidate-initial.yaml'
    selected = yaml.safe_load(initial_path.read_text()) if initial_path.exists() else None
    for row in rows:
        record = dict(id=row['id'], stage=row['stage'])
        for stage in ('geometry', 'endpoints'):
            path = out/stage/f'{row["id"]}.json'
            source = json.loads(path.read_text()) if path.exists() else {}
            for side in ('left', 'right'):
                s = source.get('sides', {}).get(side, {})
                record[f'{stage}_{side}'] = len(s.get('accepted', []))
                best = s.get('best')
                record[f'{stage}_{side}_best_mm'] = None if best is None else best['residual_norm_mm']
        path = out/'routes'/f'{row["id"]}.json'
        route = json.loads(path.read_text()) if path.exists() else {}
        same_initial = selected is not None and route.get('pair_index') == selected['source_candidate_pair_index']
        if same_initial and route.get('passed'):
            same_initial = all(np.array_equal(route['initial_'+side+'_q_rad'], selected[side+'_q_rad']) for side in ('left', 'right'))
        record['paired_route_passed'] = bool(same_initial and route.get('passed', False))
        record['route_evaluated'] = bool(route and route.get('attempts') is not None)
        data.append(record)
    complete = len(rows) == 70 and all(r['paired_route_passed'] for r in data)
    summary = dict(protocol=protocol(), requested_states=len(rows), full_inventory_count=70,
                   universal_initial_verified=complete, geometry_both=sum(r['geometry_left'] > 0 and r['geometry_right'] > 0 for r in data),
                   endpoints_both=sum(r['endpoints_left'] > 0 and r['endpoints_right'] > 0 for r in data),
                   paired_routes=sum(r['paired_route_passed'] for r in data), rows=data,
                   initial_candidate_file='candidate-initial.yaml' if selected is not None else None,
                   initial_candidate_pair_index=None if selected is None else selected['source_candidate_pair_index'],
                   scope='Finite search. Missing goals do not prove impossibility. Sampled collision checks are not continuous or native dynamics certification.')
    summary['groups'] = [dict(stage=s, total=sum(r['stage']==s for r in data),
        geometry_both=sum(r['stage']==s and r['geometry_left'] > 0 and r['geometry_right'] > 0 for r in data),
        endpoints_both=sum(r['stage']==s and r['endpoints_left'] > 0 and r['endpoints_right'] > 0 for r in data),
        paired_routes=sum(r['stage']==s and r['paired_route_passed'] for r in data)) for s in 'ABCDE']
    summary['missing_paired_endpoint_ids'] = [r['id'] for r in data if not (r['endpoints_left'] and r['endpoints_right'])]
    summary['missing_paired_route_ids'] = [r['id'] for r in data if not r['paired_route_passed']]
    write(out/'summary.json', summary)
    import csv
    with (out/'cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    table = ''.join('<tr>'+f'<td><a href="endpoints/{r["id"]}.json">{r["id"]}</a></td>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ('geometry_left', 'geometry_right', 'endpoints_left', 'endpoints_right', 'paired_route_passed'))+
        ('<td><a href="routes/'+r['id']+'.json">両腕経路</a> / <a href="routes/'+r['id']+'.csv">CSV</a></td>' if r['paired_route_passed'] else '<td>未成立</td>')+'</tr>' for r in data)
    initial_text = '<p>初期姿勢候補はまだ選定していません。</p>'
    if selected:
        initial_text = '<p><a href="candidate-initial.yaml">初期姿勢候補の位置・回転・関節角</a>。全70状態での採用条件成立：'+str(complete)+'。</p><table><tr><th>グリッパー</th><th>初期TCPの世界座標 [m]</th></tr>'
        initial_text += ''.join('<tr><td>'+s+'</td><td>'+str(selected[s+'_tcp_world_m'])+'</td></tr>' for s in ('left', 'right'))+'</table>'
        initial_text+='<p>基準シーンの人体平行移動 '+str(selected['translation_m'])+' m と、立脚側左足の床侵入許容 '+str(selected['left_foot_floor_allowance_m'])+' m を継承しています。基準姿勢で記録された立脚側左足の床侵入量は '+str(selected.get('grounded_left_foot_penetration_m'))+' m です。</p>'
    joint_margin = selected.get('joint_limit_margin_required_deg',5.) if selected else json.loads((out/'protocol.json').read_text()).get('joint_limit_margin_deg',5.)
    missing_text='<p>接近軌道の未取得状態：'+(', '.join(summary['missing_paired_route_ids'])or'なし')+'。</p>'
    if (out/'local-route-diagnostics.json').exists():
        missing_text+='<p><a href="local-route-diagnostics.json">局所移動の干渉診断</a>は、接触終点から1mm移動した場合の検査です。経路の不可能性を示すものではありません。</p>'
    group_table='<table><tr><th>開口位置</th><th>保存状態数</th><th>各腕の接触終点が両側で成立</th><th>同一初期姿勢から両腕軌道が成立</th></tr>'+''.join('<tr>'+''.join('<td>'+str(g[k])+'</td>'for k in ('stage','total','endpoints_both','paired_routes'))+'</tr>'for g in summary['groups'])+'</table>'
    gallery = ''.join('<a href="figures/'+r['id']+'.png"><img style="width:48%" src="figures/'+r['id']+'.png" alt="'+r['id']+'"></a>' for r in data if r['paired_route_passed'] and (out/'figures'/f'{r["id"]}.png').exists())
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>接触条件による両腕初期姿勢探索</title><style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:30px auto}td,th{border:1px solid #bbb;padding:5px}table{border-collapse:collapse}</style>'
        f'<h1>閉鎖しない両グリッパーの接触条件探索</h1><p>保存70状態での共通初期姿勢・サンプル軌道検査の成立：{complete}。単体幾何が両側で成立：{summary["geometry_both"]}/{len(rows)}。両腕IK終点：{summary["endpoints_both"]}/{len(rows)}。両腕配置と接近軌道：{summary["paired_routes"]}/{len(rows)}。</p>'
        f'<p>接触距離0〜0.5mm、貫通なし。靴下の材料座標15〜30mm帯、足部〜下腿の実皮膚面全体、各爪間の局所開口境界線を検査。左右は足の矢状面に対する手先中心で判定。接触対象以外の皮膚離隔1mm、関節余裕{joint_margin}度。保存形状・基部・胴体は維持。閉鎖動作なし。動力学未検証。未取得は不可能性の証明ではありません。</p>'+initial_text+
        group_table+missing_text+'<table><tr><th>状態</th><th>幾何 左</th><th>幾何 右</th><th>IK 左</th><th>IK 右</th><th>両腕軌道</th><th>記録</th></tr>'+table+'</table>'+gallery+'<p><a href="summary.json">JSON</a> / <a href="cases.csv">CSV</a> / <a href="initial-union-screen.json">初期干渉検査</a> / <a href="initial-search.json">共通初期位置の比較</a></p>')
    print('REPORT', {k:v for k,v in summary.items() if k not in ('rows', 'protocol')}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['starts','geometry', 'endpoints', 'routes', 'collect', 'audit', 'report'])
    ap.add_argument('--output', type=Path, default=ROOT/'artifacts/contact-pregrasp-start-search')
    ap.add_argument('--initial', type=Path, default=ROOT/'artifacts/common-grasp-start-search/start-candidates/initial.yaml')
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--seconds', type=float, default=30., help='Search budget per side and state')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--maximum', type=int, default=3)
    ap.add_argument('--pairs', type=int, default=4, help='Number of common start pairs to evaluate')
    ap.add_argument('--joint-margin-deg',type=float,default=5.,help='Margin inside the actual URDF joint limits; never permits limit violations')
    ap.add_argument('--right-joint-1-min-deg',type=float,help='Optional endpoint-search prior within physical limits; use a separate output to retain endpoints in other components')
    ap.add_argument('--right-elbow-min-deg',type=float,help='Optional endpoint-search lower bound for right joint 4, within physical limits')
    ap.add_argument('--right-wrist-max-deg',type=float,help='Optional endpoint-search upper bound for right joint 5, within physical limits')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.seconds <= 0 or args.workers < 1 or args.maximum < 1 or args.pairs < 1:
        ap.error('Positive budgets, workers and maximum required')
    if not np.isfinite(args.joint_margin_deg) or not 0 <= args.joint_margin_deg <= 10:
        ap.error('Joint margin must be finite and within 0--10 degrees')
    if args.right_joint_1_min_deg is not None and not np.isfinite(args.right_joint_1_min_deg):
        ap.error('Finite right joint 1 search bound required')
    if args.right_elbow_min_deg is not None and not np.isfinite(args.right_elbow_min_deg):
        ap.error('Finite right elbow search bound required')
    if args.right_wrist_max_deg is not None and not np.isfinite(args.right_wrist_max_deg):
        ap.error('Finite right wrist search bound required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in inventory(ROOT) if not args.ids or r['id'] in args.ids]
    if not rows:
        ap.error('No matching states')
    if args.ids and set(args.ids)-{r['id'] for r in rows}:
        ap.error('Unknown state IDs')
    write(out/'protocol.json', dict(protocol(), inputs=inventory(ROOT), hand_urdf_sha256=sha256(URDF),
                                   joint_limit_margin_deg=args.joint_margin_deg,
                                   geometry_manifest_sha256=sha256(REGISTERED_MANIFEST),
                                   reference_initial_sha256=sha256(args.initial)))
    if args.stage == 'report':
        report(out, rows)
        return
    if args.stage == 'audit':
        audit_geometry(out, rows, args.initial.resolve())
        return
    if args.stage=='starts':
        wide_starts(args,rows)
        return
    if args.stage == 'routes':
        routes(args, rows)
        return
    if args.stage=='collect':
        collect_routes(args,rows)
        return
    jobs = []
    for row in rows:
        path = out/args.stage/f'{row["id"]}.json'
        if args.resume and path.exists():
            existing = json.loads(path.read_text())
            if existing['condition'] != row or existing['protocol'] != protocol():
                raise ValueError('Stale result: '+str(path))
            if args.stage=='endpoints'and (existing.get('right_joint_1_search_lower_bound_deg')!=args.right_joint_1_min_deg or existing.get('right_elbow_search_lower_bound_deg')!=args.right_elbow_min_deg or existing.get('right_wrist_search_upper_bound_deg')!=args.right_wrist_max_deg):
                raise ValueError('Stale endpoint search prior: '+str(path))
            if all(side in existing['sides'] for side in ('left', 'right')):
                continue
        jobs.append((row, out, args.seconds, args.maximum) if args.stage == 'geometry' else (row, out, args.initial.resolve(), args.seconds, args.maximum,args.joint_margin_deg,args.right_joint_1_min_deg,args.right_elbow_min_deg,args.right_wrist_max_deg))
    function = geometry_one if args.stage == 'geometry' else endpoint_one
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(function, payload) for payload in jobs]
        for future in as_completed(futures):
            print('COMPLETED', future.result(), flush=True)
    report(out, rows)


if __name__ == '__main__':
    main()
