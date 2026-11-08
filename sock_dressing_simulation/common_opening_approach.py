"""Shared arm starts and opening approaches against the 70 saved sock states."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    """Use the accepted saved meshes; collapse only stage/pitch/roll duplicates."""
    root = Path(root)
    from .pose_patterns import read_patterns
    patterns = read_patterns(root/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
    first = {}
    for p in patterns:
        first.setdefault((p.pitch_deg, p.roll_deg), p.number)
    b = json.loads((root/'artifacts/diverse-left-reaching/b-natural-toe/shape-summary.json').read_text())
    qualified_b = {r['number']: r for r in b['rows'] if r['passed']}
    cde = json.loads((root/'artifacts/cde-tight-fit/expansion-report.json').read_text())
    qualified_cde = {(r['number'], r['stage']): r for r in cde['inventory'] if r['passed']}
    candidates = [('A', p) for p in sorted((root/'artifacts/bulged-cuff').glob('pose-*/state.json'))]
    candidates += [('B', p) for p in sorted((root/'artifacts/diverse-left-reaching/b-natural-toe/final-shapes').glob('pose-*/state.json'))]
    candidates += [(p.parent.name[-1], p) for p in sorted((root/'artifacts/cde-tight-fit').glob('pose-*/state.json'))]
    found = {}
    for stage, path in candidates:
        state = json.loads(path.read_text())
        p = state['pattern']
        number = p['number']
        digest = sha256(path)
        if stage == 'A':
            if not json.loads((path.parent/'report.json').read_text())['passed']:
                continue
        elif stage == 'B':
            if number not in qualified_b or digest != qualified_b[number]['state_sha256']:
                raise ValueError(f'Unqualified or changed B shape: {path}')
        elif (number, stage) not in qualified_cde or digest != qualified_cde[number, stage]['state_sha256']:
            raise ValueError(f'Unqualified or changed CDE shape: {path}')
        key = stage, p['pitch_deg'], p['roll_deg']
        if key in found:
            found[key]['saved_numbers'].append(number)
            continue
        found[key] = dict(id=f'{stage}-{number:02d}', stage=stage, number=number,
            pitch_deg=p['pitch_deg'], roll_deg=p['roll_deg'], state=str(path.relative_to(root)),
            state_sha256=digest, scene_number=first[p['pitch_deg'], p['roll_deg']], saved_numbers=[number])
    rows = sorted(found.values(), key=lambda r: (r['stage'], r['number']))
    if len(rows) != 70 or {s: sum(r['stage'] == s for r in rows) for s in 'ABCDE'} != dict(A=32, B=10, C=11, D=6, E=11):
        raise ValueError('Expected the 70 accepted opening-stage/ankle states')
    return rows


def load_state(root, row):
    path = Path(root)/row['state']
    if sha256(path) != row['state_sha256']:
        raise ValueError(f'Cloth state changed: {path}')
    state = json.loads(path.read_text())
    state.update(scene_number=row['scene_number'], applied_opening_stage=row['stage'])
    return state


def piecewise_knots(q, tolerance=1e-5):
    q = np.asarray(q, dtype=float)
    v = np.diff(q, axis=0)
    lengths = np.linalg.norm(v, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('Reference raw route has repeated samples')
    directions = v/lengths[:, None]
    turns = np.flatnonzero(np.linalg.norm(np.diff(directions, axis=0), axis=1) > tolerance)+1
    return np.r_[0, turns, len(q)-1].tolist()


def reflected_pose(position, rotation, plane_point, plane_normal):
    normal = np.asarray(plane_normal, dtype=float)
    normal /= np.linalg.norm(normal)
    reflection = np.eye(3)-2*np.outer(normal, normal)
    p = np.asarray(plane_point)+reflection@(np.asarray(position)-plane_point)
    # Reflect approach and closing axes, then restore a proper tool frame.
    R = reflection@rotation@np.diag([1., 1., -1.])
    return p, R


def reference(root, model, state, delta):
    root = Path(root)
    template_path = root/'artifacts/photo-grasp-adjustment/planned/target-left-0004.json'
    route_path = root/'artifacts/diverse-left-reaching/final/routes/pose-10-held-A-v00-trajectory.json'
    template = json.loads(template_path.read_text())['attempts'][0]['trajectory']['raw_q']
    knots = piecewise_knots(template)
    route = json.loads(route_path.read_text())
    if route['moving_side'] != 'left' or len(knots) != 4 or len(route['raw_q_rad']) != len(template):
        raise ValueError('Expected the accepted three-leg A reference route')
    index = knots[-2]
    q = np.asarray(route['raw_q_rad'][index])
    p, R = model.pose(q)
    foot = np.asarray(state['source']['surfaces'][0]['vertices'])+delta
    center = (foot.min(0)+foot.max(0))/2
    from .held_ankle_poses import foot_basis
    normal = foot_basis(state['source'])[:, 2]
    mirrored_p, mirrored_R = reflected_pose(p, R, center, normal)
    return dict(route=str(route_path.relative_to(root)), route_sha256=sha256(route_path),
        raw_template=str(template_path.relative_to(root)), template_sha256=sha256(template_path),
        template_knots=knots, penultimate_index=index, source_side='left',
        left=dict(position_m=p.tolist(), rotation=R.tolist(), q_rad=q.tolist()),
        right=dict(position_m=mirrored_p.tolist(), rotation=mirrored_R.tolist()),
        reflection_plane=dict(point_m=center.tolist(), normal=normal.tolist()),
        interpretation='Left uses the existing A penultimate waypoint; right is reflected across the foot sagittal plane, as clarified by the user')


def replace_mesh_obstacle(scene, name, vertices, faces):
    from .arm_reaching import triangle_object
    vertices = np.asarray(vertices)
    obj = triangle_object(vertices, faces)
    scene.obstacles = [(n, obj if n == name else old) for n, old in scene.obstacles]
    scene.obstacle_objects[name] = obj
    scene.obstacle_points[name] = vertices
    scene.base_obstacle_bounds[name] = vertices.min(0), vertices.max(0)
    scene.floor_minima[name] = float(vertices[:, 1].min())
    if name == 'human_full_skin':
        scene.floor_minima[name] += scene.left_foot_floor_allowance_m
    scene.fixed_pair_cache = {}
    scene.fixed_pair_valid_cache = {}


def join_meshes(meshes):
    vertices, faces, offset = [], [], 0
    for v, f in meshes:
        v = np.asarray(v)
        vertices.append(v)
        faces.append(np.asarray(f, dtype=int).reshape(-1, 3)+offset)
        offset += len(v)
    return np.concatenate(vertices), np.concatenate(faces)


def union_scene(root, rows, initial):
    """A conservative shared-start screen containing all 70 surfaces."""
    from search_fixed_torso_starts import context
    from search_centered_reaching import REGISTERED_MANIFEST
    from search_translated_reaching import load
    state = load_state(root, next(r for r in rows if r['stage'] == 'A' and r['number'] == 10))
    k, scene, *rest = context(10, state, REGISTERED_MANIFEST,
        left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
    cloth = [load_state(root, r) for r in rows]
    points, faces = join_meshes((s['particles_world'], s['triangles']) for s in cloth)
    replace_mesh_obstacle(scene, 'frozen_cloth', points, faces)
    native = [load(n) for n in sorted({r['scene_number'] for r in rows})]
    points, faces = join_meshes((s['human_skin']['vertices'], s['human_skin']['triangles']) for s in native)
    replace_mesh_obstacle(scene, 'human_full_skin', points, faces)
    posed = [next(o for o in s['obstacles'] if o['name'] == 'right_leg_actual_skin_rigid') for s in native]
    points, faces = join_meshes((o['vertices_world'], o['triangles']) for o in posed)
    replace_mesh_obstacle(scene, 'right_leg_actual_skin_rigid', points, faces)
    return k, scene, state


def own_arm_issues(check, side):
    opposite = 'left' if side == 'right' else 'right'
    own = (side+'_arm/', side+'_gripper/')
    other = (opposite+'_arm/', opposite+'_gripper/')
    return [r for r in check['issues'] if any(n.startswith(own) for n in r['pair'])
            and not any(n.startswith(other) for n in r['pair'])]


def orient(approach, closing):
    a = np.asarray(approach, dtype=float)
    a /= np.linalg.norm(a)
    b = np.asarray(closing, dtype=float)-a*np.dot(a, closing)
    if np.linalg.norm(b) < 1e-7:
        b = np.eye(3)[np.argmin(abs(a))]
        b -= a*np.dot(a, b)
    b /= np.linalg.norm(b)
    return np.column_stack([a, b, np.cross(a, b)])


def approach_targets(state, side, reference_rotation, delta, tip_clearance=.002):
    """Candidate physical tip-midplanes 2 mm outside the actual opening band.

    The calibrated TCP is 10 mm behind the physical finger tips. Compensate
    this explicitly; a TCP on the skin-tight rim would put the tips inside it.
    """
    from .grasp_geometry import candidate_samples
    sign = 1. if side == 'left' else -1.
    samples = candidate_samples(state, circumferential_samples=32, depths_m=(0., .005, .015))
    samples = [s for s in samples if sign*np.asarray(s['normal'])[0] > .1]
    samples.sort(key=lambda s: (s['depth_m'] != 0., -sign*s['position'][0], s['depth_m']))
    for sample in samples:
        normal = np.asarray(sample['normal'])
        along = np.asarray(sample['along'])
        # Prefer keeping the initial hand attitude when its approach points
        # toward this side of the opening. This avoids needless wrist turns.
        reference_rotation = np.asarray(reference_rotation)
        if np.dot(reference_rotation[:, 0], -normal) > .15:
            tip = np.asarray(sample['position'])+delta+normal*tip_clearance
            tcp = tip-.010*reference_rotation[:, 0]
            yield dict(sample=sample, tip_midplane_target_m=tip.tolist(), tcp_target_m=tcp.tolist(),
                rotation=reference_rotation.tolist(), finger_tip_inset_m=.010,
                tip_clearance_m=tip_clearance, tilt_deg=None, roll_deg=None,
                orientation_mode='preserve_initial_attitude')
        for tilt in (0., 25., -25., 45., -45.):
            a = -normal*np.cos(np.deg2rad(tilt))+along*np.sin(np.deg2rad(tilt))
            nominal = orient(a, np.asarray(reference_rotation)[:, 1])
            for roll in (0., 30., -30., 60., -60., 90., -90., 180.):
                R = nominal@Rotation.from_rotvec([np.deg2rad(roll), 0., 0.]).as_matrix()
                tip = np.asarray(sample['position'])+delta+normal*tip_clearance
                tcp = tip-.010*R[:, 0]
                yield dict(sample=sample, tip_midplane_target_m=tip.tolist(), tcp_target_m=tcp.tolist(),
                    rotation=R.tolist(), finger_tip_inset_m=.010, tip_clearance_m=tip_clearance,
                    tilt_deg=tilt, roll_deg=roll)


def edge_samples(a, b, maximum_joint_step_deg=.5):
    a, b = np.asarray(a), np.asarray(b)
    count = max(1, int(np.ceil(np.max(abs(b-a))/np.deg2rad(maximum_joint_step_deg))))
    return np.linspace(a, b, count+1)


def guided_approach_targets(state, side, model, start, delta, tip_clearance=.002):
    """Let the arm choose a reachable attitude while keeping the tip goal exact."""
    from scipy.optimize import least_squares
    from .grasp_geometry import candidate_samples
    sign = 1. if side == 'left' else -1.
    samples = candidate_samples(state, circumferential_samples=32, depths_m=(0., .005, .015))
    samples = [s for s in samples if sign*np.asarray(s['normal'])[0] > .1]
    samples.sort(key=lambda s: (s['depth_m'] != 0., -sign*s['position'][0], s['depth_m']))
    alternate = np.asarray(start).copy(); alternate[4] = -2.4; alternate[5] = .7; alternate[6] = .3
    rng = np.random.default_rng(20261108)
    for sample in samples:
        normal = np.asarray(sample['normal'])
        tip = np.asarray(sample['position'])+delta+normal*tip_clearance
        for seed in (start, alternate, np.clip(start+rng.normal(0., .6, 7), model.lower+1e-7, model.upper-1e-7)):
            seed = np.clip(seed, model.lower+1e-7, model.upper-1e-7)
            def values(q):
                p, R, Jp, Ja = model.fast_pose(q)
                direction_J = np.cross(Ja.T, R[:, 0]).T
                return p+.010*R[:, 0], R[:, 0], Jp+.010*direction_J, direction_J
            def residual(q):
                actual, direction, _, _ = values(q)
                return np.r_[20*(actual-tip), .015*(direction+normal), .0002*(q-seed)]
            def jacobian(q):
                _, _, Jtip, Jdirection = values(q)
                return np.vstack([20*Jtip, .015*Jdirection, .0002*np.eye(7)])
            fit = least_squares(residual, seed, jac=jacobian, bounds=(model.lower, model.upper),
                                max_nfev=100, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            # Enforce the physical tip location independently of the soft
            # attitude preference. The requested task does not prescribe roll.
            exact = least_squares(lambda q: values(q)[0]-tip, fit.x,
                jac=lambda q: values(q)[2], bounds=(model.lower, model.upper),
                max_nfev=60, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            actual_tip, direction, _, _ = values(exact.x)
            if np.linalg.norm(actual_tip-tip) > 8e-5 or np.dot(direction, -normal) < .05:
                continue
            _, R = model.pose(exact.x)
            tcp = tip-.010*R[:, 0]
            yield dict(sample=sample, tip_midplane_target_m=tip.tolist(), tcp_target_m=tcp.tolist(),
                rotation=R.tolist(), finger_tip_inset_m=.010, tip_clearance_m=tip_clearance,
                orientation_mode='arm_guided_exact_tip_position', seed_q_rad=exact.x.tolist())


def linear_quintic_profile(q, velocity, acceleration):
    """Preserve a joint polyline exactly, stopping smoothly at its true turns."""
    from scipy.optimize import brentq
    q = np.asarray(q)
    q = q[np.r_[True, np.linalg.norm(np.diff(q, axis=0), axis=1) > 1e-10]]
    q = q[piecewise_knots(q)]
    lengths = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.r_[0., np.cumsum(lengths)]; s /= s[-1]
    us, times, positions, speeds, accelerations, durations = [], [], [], [], [], []
    vbound, abound = np.zeros(q.shape[1]), np.zeros(q.shape[1])
    offset = 0.
    for i, dq in enumerate(np.diff(q, axis=0)):
        duration = max(.2, float(np.max(1.875*abs(dq)/(velocity*.1))),
                       float(np.sqrt(np.max(5.773503*abs(dq)/(acceleration*.1)))))
        durations.append(duration)
        count = max(2, int(np.ceil(np.max(abs(dq))/np.deg2rad(.25))))
        ratio = np.linspace(0., 1., count+1)
        if i < len(q)-2: ratio = ratio[:-1]
        tau = np.array([0. if r == 0 else 1. if r == 1 else brentq(
            lambda t: 10*t**3-15*t**4+6*t**5-r, 0., 1.) for r in ratio])
        gp = 30*tau**2-60*tau**3+30*tau**4
        gpp = 60*tau-180*tau**2+120*tau**3
        us.extend(s[i]+(s[i+1]-s[i])*ratio)
        times.extend(offset+duration*tau)
        positions.extend(q[i]+ratio[:, None]*dq)
        speeds.extend(gp[:, None]*dq/duration)
        accelerations.extend(gpp[:, None]*dq/duration**2)
        vbound = np.maximum(vbound, 1.875*abs(dq)/duration)
        abound = np.maximum(abound, 5.773503*abs(dq)/duration**2)
        offset += duration
    def curve(u):
        return np.column_stack([np.interp(u, s, q[:, j]) for j in range(q.shape[1])])
    return dict(q=np.asarray(positions), t=np.asarray(times), velocity=np.asarray(speeds),
        acceleration=np.asarray(accelerations), duration=offset, u=np.asarray(us), curve=curve,
        velocity_bound=vbound, acceleration_bound=abound, segment_durations_s=durations,
        knots_q_rad=q.tolist())
