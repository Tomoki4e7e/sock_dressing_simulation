#!/usr/bin/env python3
"""Bounded exploratory endpoint search with one shared robot configuration."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from search_fixed_torso_starts import context
from search_centered_reaching import REGISTERED_MANIFEST, write, geometric_check
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics, ControlledCollisionScene
from sock_dressing_simulation.grasp_geometry import candidate_samples, SkinBody, orientation
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances
from sock_dressing_simulation.preinsert import generate_preinsert
from sock_dressing_simulation.held_ankle_poses import foot_basis
from sock_dressing_simulation.pose_patterns import read_patterns
from dataclasses import asdict
from sock_dressing_simulation.reaching_dataset import file_sha256


def existing_geometry(numbers,output,initial_path):
    """Reuse the independently screened full-grid endpoints before resampling."""
    initial=yaml.safe_load(initial_path.read_text());delta=np.array(initial['translation_m']);start=np.array(initial['left_q_rad'])
    patterns=read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
    refs=json.loads((ROOT/'artifacts/centered-reaching/full-grid/geometry-summary.json').read_text())['accepted']
    prior=json.loads((ROOT/'artifacts/photo-grasp-adjustment/planned/target-left-0004.json').read_text())
    for n in numbers:
        path=output/f'pose-{n:02d}-held-A.json'
        if path.exists():
            if json.loads(path.read_text()).get('initial_sha256')!=file_sha256(initial_path):raise ValueError('Cached shared start differs: '+str(path))
            continue
        pattern=next(p for p in patterns if p.number==n)
        source_n=next(p.number for p in patterns if (p.pitch_deg,p.roll_deg)==(pattern.pitch_deg,pattern.roll_deg))
        state=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{source_n:02d}/state.json').read_text())
        state.update(scene_number=source_n,pattern=asdict(pattern),applied_opening_stage='A')
        base,scene,*_=context(n,state,REGISTERED_MANIFEST,left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
        scene.park_left(start,initial['left_gap_m']);model=ControlledArmKinematics(base,'left',initial['right_q_rad'],initial['right_gap_m']);checker=ControlledCollisionScene(scene,'left',initial['right_q_rad'],initial['right_gap_m'])
        check=checker.slacks(start,initial['left_gap_m'],delta,details=True);accepted=[];counts={};began=time.monotonic()
        comfortable=copy.copy(model);comfortable.lower=model.lower+np.deg2rad(5);comfortable.upper=model.upper-np.deg2rad(5)
        if check['passed']:
            samples=candidate_samples(state);rng=np.random.default_rng(20261105+n)
            for ref in refs:
                if ref['number']!=source_n:continue
                geometry=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']]
                if geometry['source_kind']!='bulged':continue
                c=geometry['candidate'];p=np.array(c['grasp_center_unity_world_m'])+delta;R=np.array(c['rotation_matrix'])
                seeds=[np.array(prior['attempts'][0]['q']),start]+list(rng.uniform(comfortable.lower,comfortable.upper,(6,7)))
                for seed in seeds:
                    sol=comfortable.solve(p,R,seed)
                    if sol['position_error_m']>1e-4 or sol['orientation_error_deg']>.5:
                        counts['ik_failed']=counts.get('ik_failed',0)+1;continue
                    q=np.array(sol['q'])
                    for opening in sorted(geometry['valid_openings'],key=lambda g:-g['gap_m']):
                        gap=opening['gap_m']
                        if not checker.slacks(q,gap,delta,details=True)['passed']:
                            counts['endpoint_collision']=counts.get('endpoint_collision',0)+1;continue
                        if not all(checker.slacks(q,float(g),delta,allow_cloth_overlap=True,details=True)['passed'] for g in np.linspace(gap,.001,21)):
                            counts['closure_collision']=counts.get('closure_collision',0)+1;continue
                        accepted.append(dict(q=q.tolist(),gap_m=gap,geometry=geometry,sample=samples[c['sample_index']]))
                        break
                    if accepted:break
                if accepted:break
        write(path,dict(number=n,stage='held-A',initial_configuration=str(initial_path),initial_sha256=file_sha256(initial_path),workbook_sha256=file_sha256(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx'),initial_check=check,accepted=accepted,counts=counts,elapsed_s=time.monotonic()-began,search_mode='existing_full_grid_geometry'))
        if accepted:write(output/'states'/path.name,state)
        print('EXISTING GRID',n,'found',bool(accepted),counts,'seconds',round(time.monotonic()-began,1),flush=True)


def run(numbers, stages, output, initial_path, seconds, arm_guided=False,retry=False):
    initial = yaml.safe_load(initial_path.read_text())
    initial_sha=file_sha256(initial_path)
    workbook_sha=file_sha256(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
    prior = json.loads((ROOT / 'artifacts/photo-grasp-adjustment/planned/target-left-0004.json').read_text())
    qref = np.array(prior['attempts'][0]['q'])
    Rref = np.array(prior['geometry_record']['candidate']['rotation_matrix'])
    start = np.array(initial['left_q_rad']); delta = np.array(initial['translation_m'])
    settings = yaml.safe_load((ROOT / 'config/geometric_closure.yaml').read_text())
    patterns = read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')
    reference_state = json.loads((ROOT/'artifacts/bulged-cuff/pose-10/state.json').read_text())
    for n in numbers:
        pattern = next(p for p in patterns if p.number == n)
        source_number = next(p.number for p in patterns if p.pitch_deg == pattern.pitch_deg and p.roll_deg == pattern.roll_deg)
        original = json.loads((ROOT / f'artifacts/bulged-cuff/pose-{source_number:02d}/state.json').read_text())
        for stage in stages:
            if stage == 'workbook': stage = 'held-A' if pattern.stage == 'A' else pattern.stage
            path = output / f'pose-{n:02d}-{stage}.json'
            if path.exists() and not retry:
                cached=json.loads(path.read_text())
                if cached.get('initial_sha256')!=initial_sha or cached.get('workbook_sha256',workbook_sha)!=workbook_sha:
                    raise ValueError('Cached endpoint inputs differ; use a new output directory: '+str(path))
                continue
            state = copy.deepcopy(original)
            state.update(scene_number=source_number, pattern=asdict(pattern))
            if stage != 'held-A':
                points, placement = generate_preinsert(state['source'], stage, clearance_m=.015)
                state.update(particles_world=points.tolist(), opening_normal=placement['opening_inward_normal_world'])
                state['placement'] = placement
                state['reference_geometry_mode'] = original['mode']
                state['mode'] = 'frozen_normal_preinsert'
                state.pop('bulge', None)
                state['settings'] = dict(generator='generate_preinsert',clearance_m=.015,above_heel_m=.05)
            state['applied_opening_stage']='A' if stage=='held-A' else stage
            base, scene, *rest = context(n, state, REGISTERED_MANIFEST, left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
            scene.park_left(start, .04)
            model = ControlledArmKinematics(base, 'left', initial['right_q_rad'], .04)
            checker = ControlledCollisionScene(scene, 'left', initial['right_q_rad'], .04)
            initial_check = checker.slacks(start, .04, delta, details=True)
            if not initial_check['passed']:
                write(path, dict(number=n, stage=stage, initial_configuration=str(initial_path), initial_sha256=initial_sha,workbook_sha256=workbook_sha, initial_check=initial_check, accepted=[]))
                print('PROBE initial rejected', n, stage, initial_check['issues'], flush=True)
                continue
            samples = candidate_samples(state)
            samples.sort(key=lambda s: -s['position'][1])
            skin = SkinBody(state['source']['skin_surface']); cloth = ClothScreen(state['particles_world'], state['triangles'])
            faces = np.array(state['triangles']).reshape(-1, 3); material = np.array(state['source']['rest_positions'])
            comfortable = copy.copy(model); comfortable.lower = model.lower + np.deg2rad(5); comfortable.upper = model.upper - np.deg2rad(5)
            counts = {}; accepted = []; began = time.monotonic()
            transported = foot_basis(state['source']) @ foot_basis(reference_state['source']).T @ Rref
            rotations = [transported, Rref] + [Rotation.from_rotvec(axis * np.deg2rad(a)).as_matrix() @ transported for a in (15, -15, 30, -30) for axis in np.eye(3)]
            for s in samples:
                if time.monotonic() - began > seconds: break
                if s['depth_m'] > .030: continue
                dist = material_distances(material, faces, faces[s['face']], s['weights'])
                local_rotations = [orientation(s,yaw,tilt) @ Rotation.from_rotvec([np.deg2rad(roll),0,0]).as_matrix() for yaw in (330,0,180,90,270,300,30,150,210,60,120,240) for tilt in (-45,0,45,-75,75) for roll in (135,90,180,0,45,225,270,315)]
                proposals=[]
                if arm_guided:
                    rng=np.random.default_rng(1729+s['index'])
                    target=np.array(s['position'])+delta
                    for sign in (1,-1):
                        normal=sign*np.array(s['normal'])
                        for seed in [start,qref]+list(start+rng.normal(0,.5,(6,7))):
                            def residual(q):
                                p,R=model.pose(q)
                                return np.r_[p-target,.15*(R[:,1]-normal)]
                            def jacobian(q):
                                _,R,Jp,Ja=model.fast_pose(q)
                                return np.vstack([Jp,.15*np.cross(Ja.T,R[:,1]).T])
                            fit=least_squares(residual,np.clip(seed,comfortable.lower+1e-9,comfortable.upper-1e-9),jac=jacobian,bounds=(comfortable.lower,comfortable.upper),max_nfev=100,ftol=1e-9,gtol=1e-9,xtol=1e-9)
                            if np.linalg.norm(fit.fun[:3])<1e-4 and np.linalg.norm(fit.fun[3:])<.005:proposals.append((model.pose(fit.x)[1],fit.x))
                else:proposals=[(R,None) for R in rotations+local_rotations]
                for R,guided_q in proposals:
                    if time.monotonic()-began>seconds:break
                    geometry, reason = geometric_check(model, state, s, R, skin, cloth, settings, dist, approach_distance=0.)
                    counts[reason] = counts.get(reason, 0) + 1
                    if not geometry: continue
                    sol = comfortable.solve(np.array(s['position']) + delta, R, np.clip(qref, comfortable.lower+1e-9, comfortable.upper-1e-9))
                    if guided_q is not None:sol=dict(q=guided_q.tolist(),position_error_m=float(np.linalg.norm(model.pose(guided_q)[0]-np.array(s['position'])-delta)),orientation_error_deg=0.)
                    if sol['position_error_m'] > 1e-4 or sol['orientation_error_deg'] > .5:
                        sol = comfortable.solve(np.array(s['position']) + delta, R, np.clip(start, comfortable.lower+1e-9, comfortable.upper-1e-9))
                    for seed in (np.deg2rad([140,-60,-100,90,-140,60,60]), np.deg2rad([100,-40,-80,90,140,60,0])):
                        if sol['position_error_m'] <= 1e-4 and sol['orientation_error_deg'] <= .5: break
                        sol = comfortable.solve(np.array(s['position']) + delta, R, np.clip(seed, comfortable.lower+1e-9, comfortable.upper-1e-9))
                    if sol['position_error_m'] > 1e-4 or sol['orientation_error_deg'] > .5:
                        counts['ik_failed'] = counts.get('ik_failed', 0) + 1; continue
                    for opening in sorted(geometry['valid_openings'], key=lambda g: -g['gap_m']):
                        gap = opening['gap_m']; q = np.array(sol['q'])
                        check = checker.slacks(q, gap, delta, details=True)
                        if not check['passed']:
                            counts['endpoint_collision'] = counts.get('endpoint_collision', 0) + 1; continue
                        closure = [checker.slacks(q, float(g), delta, allow_cloth_overlap=True, details=True) for g in np.linspace(gap, .001, 11)]
                        if not all(g['passed'] for g in closure): continue
                        accepted.append(dict(q=q.tolist(), gap_m=gap, geometry=geometry, sample=s))
                        print('PROBE ENDPOINT', n, stage, len(accepted), 'elapsed', round(time.monotonic()-began, 1), flush=True)
                        break
                    if accepted: break
                if accepted: break
            if file_sha256(initial_path)!=initial_sha:raise ValueError('Shared initial configuration changed during search')
            write(path, dict(number=n, stage=stage, initial_configuration=str(initial_path), initial_sha256=initial_sha,workbook_sha256=workbook_sha, initial_check=initial_check, accepted=accepted, counts=counts, elapsed_s=time.monotonic()-began))
            if accepted:
                write(output / 'states' / f'pose-{n:02d}-{stage}.json', state)
            print('PROBE', n, stage, counts, 'elapsed', round(time.monotonic()-began, 1), flush=True)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--numbers', type=int, nargs='+'); ap.add_argument('--stages', nargs='+', choices=['held-A','A','B','C','D','E','workbook'], default=['workbook']); ap.add_argument('--output', type=Path, default=ROOT/'artifacts/diverse-left-reaching/workbook-shared-probe'); ap.add_argument('--initial', type=Path, default=ROOT/'config/diverse_left_reaching_shared.yaml'); ap.add_argument('--seconds', type=float, default=45.); ap.add_argument('--arm-guided',action='store_true');ap.add_argument('--retry',action='store_true');ap.add_argument('--existing-geometry',action='store_true')
    args = ap.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0:ap.error('--seconds must be finite and positive')
    numbers=args.numbers or [p.number for p in read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx')]
    if args.existing_geometry:existing_geometry(numbers,args.output.resolve(),args.initial.resolve())
    else:run(numbers, args.stages, args.output.resolve(), args.initial.resolve(), args.seconds,args.arm_guided,args.retry)
