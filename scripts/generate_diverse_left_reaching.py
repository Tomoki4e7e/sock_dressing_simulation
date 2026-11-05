#!/usr/bin/env python3
"""Generate spatially distinct left-arm routes under one immutable shared start.

Endpoint files come from search_diverse_left_endpoints.py. Every proposed route
is checked with the registered whole-robot geometry and independently validated
after smoothing. Failed proposals remain diagnostic records, never successes.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from search_centered_reaching import write, REGISTERED_MANIFEST,geometric_check
from search_fixed_torso_starts import context
from validate_controlled_photo_grasp import validate
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics, ControlledCollisionScene
from sock_dressing_simulation.grasp_geometry import candidate_samples,SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances
from sock_dressing_simulation.held_ankle_poses import audit_actual_skin
from sock_dressing_simulation.reaching_dataset import file_sha256, spatial_route_distance,verify_dataset_inputs,load_endpoint_samples
from plan_fixed_torso_route import rrt
from sock_dressing_simulation.fitted_b_cuff import PROFILE as FITTED_B_PROFILE


def generate(args):
    initial = yaml.safe_load(args.initial.read_text())
    reference_path = ROOT/'artifacts/photo-grasp-adjustment/planned/target-left-0004.json'
    reference = json.loads(reference_path.read_text())
    template = np.array(reference['attempts'][0]['trajectory']['raw_q'])
    start = np.array(initial['left_q_rad']); delta = np.array(initial['translation_m'])
    variants = [(None, 0.), (0, 3.), (0, -3.), (5, 8.), (5, -8.), (3, 3.), (3, -3.), (1, 3.), (1, -3.)]
    known={}
    for path in sorted((args.output/'routes').glob('*-trajectory.json')):
        saved=json.loads(path.read_text())
        if not saved.get('offline_passed'):continue
        verify_dataset_inputs(saved,ROOT)
        if saved['applied_opening_stage']=='B':
            previous_state=json.loads((ROOT/saved['geometry_record']['state_file']).read_text())
            if previous_state.get('shape_profile')!=FITTED_B_PROFILE:continue
        p=saved['workbook_pattern'];key=(p['pitch_deg'],p['roll_deg'],saved['applied_opening_stage'])
        known.setdefault(key,[]).append((saved['tcp_world_m'],str(path.relative_to(ROOT))))
    files = sorted(args.probe.glob('pose-*.json'))
    if args.numbers:
        order={n:i for i,n in enumerate(args.numbers)}
        files.sort(key=lambda path:order.get(int(path.name.split('-')[1]),len(order)))
    for path in files:
        probe = json.loads(path.read_text())
        if args.numbers and probe['number'] not in args.numbers:continue
        if not probe['accepted']: continue
        stem = path.stem
        status_path = args.output/'search'/f'{stem}.json'
        state_path = args.probe/'states'/path.name
        state = json.loads(state_path.read_text())
        if probe['stage']=='B' and state.get('shape_profile')!=FITTED_B_PROFILE:
            raise ValueError('B requires its continuous forefoot/toe envelope and the 15 mm / +5 mm cuff bulge')
        p=state['pattern'];condition=(p['pitch_deg'],p['roll_deg'],'A' if probe['stage']=='held-A' else probe['stage'])
        if status_path.exists() and not args.retry:
            cached=json.loads(status_path.read_text())
            if probe['stage']=='B' and not known.get(condition) and cached.get('shape_profile')!=FITTED_B_PROFILE:
                raise ValueError('Cached B route search predates the fitted shape; choose a new output or --retry')
            if cached.get('state_sha256') and cached['state_sha256']!=file_sha256(state_path):
                raise ValueError('Cached route search uses a different cloth shape')
            continue
        if len(known.get(condition,[]))>=args.variants:
            write(status_path,dict(number=probe['number'],stage=probe['stage'],reused_condition_routes=[p for _,p in known[condition]],routes=[]))
            continue
        body, surface = audit_actual_skin(np.asarray(state['particles_world']), state['source']['skin_surface'], state['edges'], state['triangles'])
        status = dict(number=probe['number'], stage=probe['stage'],shape_profile=state.get('shape_profile'),state_sha256=file_sha256(state_path), state_body=body, state_surface=surface, routes=[], rejected=[])
        if not body['geometry_ok'] or not surface['surface_ok']:
            write(status_path,status); print('STATE AUDIT REJECTED', stem, flush=True); continue
        samples_path = args.output/'samples'/path.name
        samples=load_endpoint_samples(probe,state,ROOT)
        write(samples_path, samples)
        k, base, dl, du, contract, scene = context(probe['number'], state, REGISTERED_MANIFEST, left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
        base.park_left(start,initial['left_gap_m'])
        model = ControlledArmKinematics(k, 'left', initial['right_q_rad'], initial['right_gap_m'])
        checker = ControlledCollisionScene(base, 'left', initial['right_q_rad'], initial['right_gap_m'])
        successful_tcp = [np.array(tcp) for tcp,_ in known.get(condition,[])]
        faces=np.array(state['triangles']).reshape(-1,3);material=np.array(state['source']['rest_positions'])
        skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles'])
        closure_settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text())
        for endpoint in probe['accepted']:
            goal = np.array(endpoint['q']); gap = endpoint['gap_m']
            actual,R=model.pose(goal);sample=dict(samples[endpoint['geometry']['candidate']['sample_index']],position=(actual-delta).tolist())
            dist=material_distances(material,faces,faces[sample['face']],sample['weights'])
            geometry,reason=geometric_check(model,state,sample,R,skin,cloth,closure_settings,dist,approach_distance=0.)
            if not geometry:
                status['rejected'].append(dict(reason='actual_endpoint_geometry_failed',geometry_reason=reason));write(status_path,status);continue
            endpoint=dict(endpoint,geometry=dict(endpoint['geometry'],**geometry))
            progress = np.r_[0., np.cumsum(np.linalg.norm(np.diff(template,axis=0),axis=1))];progress/=progress[-1]
            blend=np.clip((progress-.45)/.55,0.,1.);blend=blend**3*(10-15*blend+6*blend**2)
            start_blend=np.clip(progress/.25,0.,1.);start_blend=1-start_blend**3*(10-15*start_blend+6*start_blend**2)
            corrected = template + blend[:,None]*(goal-template[-1]) + start_blend[:,None]*(start-template[0])
            outside=np.any(corrected<model.lower+np.deg2rad(5)-1e-9) or np.any(corrected>model.upper-np.deg2rad(5)+1e-9)
            if outside or not all(checker.is_clear(q,gap,delta) for q in corrected[::8]):
                route=rrt(model,checker,start,goal,gap,delta,iterations=args.iterations)
                if route['passed']:
                    corrected=np.array(route['raw_q'])
                    progress=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(corrected,axis=0),axis=1))];progress/=progress[-1]
                else:
                    status['rejected'].append(dict(reason='rrt_failed',details=route));write(status_path,status);continue
            bump=np.sin(np.pi*progress)**4
            for variant, (joint, amplitude) in enumerate(variants):
                raw = corrected.copy()
                if joint is not None: raw[:,joint] += np.deg2rad(amplitude)*bump
                raw[0] = start; raw[-1] = goal
                if np.any(raw<model.lower+np.deg2rad(5)-1e-9) or np.any(raw>model.upper-np.deg2rad(5)+1e-9):
                    status['rejected'].append(dict(variant=variant, reason='joint_margin')); continue
                sparse = np.unique(np.r_[np.arange(0,len(raw),4),len(raw)-1])
                if not all(checker.is_clear(raw[i],gap,delta) for i in sparse):
                    status['rejected'].append(dict(variant=variant, reason='route_collision')); continue
                tcp = np.array([model.pose(q)[0] for q in raw])
                distances = [spatial_route_distance(tcp, other) for other in successful_tcp]
                if distances and min(distances)<args.minimum_diversity:
                    status['rejected'].append(dict(variant=variant, reason='insufficient_spatial_diversity', rms_m=min(distances))); continue
                geometry = dict(endpoint['geometry'], number=probe['number'], state_file=str(state_path.relative_to(ROOT)), samples_file=str(samples_path.relative_to(ROOT)))
                inputs = [args.initial, ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx', state_path, samples_path, reference_path,ROOT/'config/geometric_closure.yaml']
                record = dict(number=probe['number'], geometry_record=geometry, central_contract=contract,
                    left_q_rad=initial['left_q_rad'], right_q_rad=initial['right_q_rad'], left_gap_m=initial['left_gap_m'], right_gap_m=initial['right_gap_m'],
                    moving_side='left', left_foot_floor_allowance_m=initial['left_foot_floor_allowance_m'], initial_opening_gap_m=initial['left_gap_m'],
                    endpoint_task='geometric_grasp_target', require_virtual_closure=True, close_after_reach=True, closing_gap_m=.001,
                    shared_initial_configuration=str(args.initial.relative_to(ROOT)),
                    dataset_inputs=[dict(file=str(p.relative_to(ROOT)),sha256=file_sha256(p)) for p in inputs],
                    applied_opening_stage='A' if probe['stage']=='held-A' else probe['stage'], workbook_pattern=state['pattern'],
                    ignored_workbook_conditions=['sock_id','shift_cm','diagonal_deg','opening','wrinkles','fold']+(['stage'] if ('A' if probe['stage']=='held-A' else probe['stage'])!=state['pattern']['stage'] else []),
                    human_and_torso_unchanged=True, route_variant=dict(joint_index=joint, amplitude_deg=amplitude, minimum_rms_tcp_separation_m=args.minimum_diversity),
                    attempts=[dict(q=goal.tolist(),gap_m=gap,translation_m=delta.tolist(),left_q_rad=initial['left_q_rad'],stage='endpoint_passed',trajectory=dict(passed=True,raw_q=raw.tolist(),stage='transferred_spatial_route'))])
                source = args.output/'planned'/f'{stem}-v{variant:02d}.json'
                write(source, record)
                validate(source, args.output, 1.)
                trajectory = args.output/'routes'/f'{source.stem}-trajectory.json'
                validation = json.loads(trajectory.with_name(trajectory.name.replace('trajectory','validation')).read_text())
                if validation['offline_passed']:
                    successful_tcp.append(tcp)
                    known.setdefault(condition,[]).append((tcp,str(trajectory.relative_to(ROOT))))
                    status['routes'].append(dict(trajectory=str(trajectory.relative_to(ROOT)),variant=variant,rms_separation_from_prior_routes_m=distances))
                    print('DIVERSE ROUTE PASSED',stem,variant,flush=True)
                else: status['rejected'].append(dict(variant=variant,reason='smoothed_validation_failed'))
                write(status_path,status)
                if not validation['actual_grasp_geometry_passed']:break
                if len(successful_tcp)>=args.variants: break
            if len(successful_tcp)>=args.variants: break
        write(status_path,status)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe',type=Path,required=True)
    ap.add_argument('--initial',type=Path,default=ROOT/'config/diverse_left_reaching_shared.yaml')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/final')
    ap.add_argument('--variants',type=int,default=2)
    ap.add_argument('--numbers',type=int,nargs='+')
    ap.add_argument('--minimum-diversity',type=float,default=.005)
    ap.add_argument('--retry',action='store_true')
    ap.add_argument('--iterations',type=int,default=1200)
    a=ap.parse_args()
    a.probe=a.probe.resolve();a.output=a.output.resolve();a.initial=a.initial.resolve()
    if a.variants<1 or not np.isfinite(a.minimum_diversity) or a.minimum_diversity<=0: ap.error('Require positive variant count and finite positive diversity threshold')
    generate(a)
