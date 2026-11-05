#!/usr/bin/env python3
"""Search additional jaw roll, then centered placements and right-arm routes.

All outputs have their own directory. Existing candidates and trajectories are
read-only inputs. Geometry screening retains the established 1mm/4mm2 policy.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
import hashlib
import json
import sys
import time
from uuid import uuid4
from collections import Counter
from pathlib import Path
import numpy as np
import yaml
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.centered_reaching import roll_tool, placement_bounds
from sock_dressing_simulation.grasp_geometry import SkinBody, orientation
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances, projected_cloth_area
from search_compound_reaching import target_data, pose_constraint, Constraints, trace
from search_translated_reaching import load
from apron_reaching_common import overlay_robot, AuditedScene

URDF = ROOT / 'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf'
ROBOT = ROOT / 'artifacts/apron-reaching/robot-scene.json'
FAST_MANIFEST = ROOT / 'artifacts/compound-reaching/decomposition-original-parts.json'
REGISTERED_MANIFEST = ROOT / 'artifacts/compound-reaching/registration/candidate/manifest.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def geometric_check(k, state, sample, R, skin, cloth, settings, dist, approach_distance=.05):
    """Check capture/skin, plus the requested straight tool-only approach.

    A zero distance screens endpoints only; a robot trajectory must still be
    checked separately before accepting such a proposal as reachable.
    """
    if not np.isfinite(approach_distance) or approach_distance<0:
        raise ValueError('Approach distance must be finite and nonnegative')
    h = k.hand
    p = np.asarray(sample['position'])
    if not skin.inspect(h.bank(.001), R, p, .001)[0]:
        return None, 'closed_foot'
    points = np.asarray(state['particles_world'])
    faces = np.asarray(state['triangles']).reshape(-1, 3)
    rest = np.asarray(state['source']['rest_positions'])
    depth = rest[:, 2] - rest[:, 2].min()
    fingers = [v for n, v in h.vertices(.001).items() if n.endswith('/left_finger_link') or n.endswith('/right_finger_link')]
    projection = projected_cloth_area((points-p)@R, faces, fingers, dist, depth,
        **{n: settings[n] for n in ('half_extent_m', 'edge_clearance_m', 'material_radius_m', 'band_width_m', 'topology_tolerance_m')})
    if (min(projection['area_m2'], projection['largest_component_area_m2']) < 4e-6
            or projection['area_fraction'] < .05
            or min(projection['span_x_m'], projection['span_z_m']) < .0005):
        return None, 'area'
    from reevaluate_geometric_closure import insertion
    c = dict(sample_index=sample['index'], rotation_matrix=R.tolist(), grasp_center_unity_world_m=p.tolist())
    gaps = []
    for gap in (.0015, .002, .003, .004, .005, .0075, .01, .015, .02, .025, .03, .04, .05):
        bank = h.bank(gap, 'approach', approach=approach_distance)
        if not skin.inspect(bank, R, p, .001)[0] or cloth.intersections(bank, R, p):
            continue
        ins = insertion(h, skin, dict(c, gap_m=gap), sample)
        if ins['passed'] and skin.inspect(h.bank(gap, 'closure', pinch_gap=.001), R, p, .001)[0]:
            gaps.append(dict(gap_m=gap, insertion=ins))
    if not gaps:
        return None, 'opening_or_approach'
    return dict(candidate=c, projection=projection, valid_openings=gaps), 'accepted'


def geometry(args):
    old = json.loads((ROOT / 'artifacts/translated-reaching/summary.json').read_text())['results']
    old = [r for r in old if not args.numbers or r['number'] in args.numbers]
    settings = yaml.safe_load((ROOT / 'config/geometric_closure.yaml').read_text())
    counts = Counter()
    robot = json.loads(ROBOT.read_text())
    grouped = {}
    for r in old:
        grouped.setdefault(r['number'], []).append(r)
    for number, rows in sorted(grouped.items()):
        output = args.output / 'geometry' / f'pose-{number:02d}.json'
        if output.exists():
            continue
        contexts = {}
        seen = set()
        accepted = []
        local = Counter()
        for row in rows:
            c, state, _ = target_data(row)
            # Legacy closure records store their source geometry separately.
            if 'source' not in state:
                prior = json.loads((ROOT / f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text())
                base = json.loads((ROOT / prior['input_state']).read_text())
                state = dict(state, source=base['source'])
            state_id = 'bulged' if 'bulged-cuff' in row['source'] else 'original'
            if state_id not in contexts:
                scene = overlay_robot(load(number), robot)
                scene['particles_world'] = state['particles_world']
                k = Kinematics(URDF, scene)
                if state_id == 'bulged':
                    samples = json.loads((ROOT / f'artifacts/bulged-cuff/pose-{number:02d}/samples.json').read_text())
                else:
                    samples = json.loads((ROOT / f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text())['samples']
                contexts[state_id] = (k, SkinBody(state['source']['skin_surface']), ClothScreen(state['particles_world'], state['triangles']), samples, {})
            k, skin, cloth, samples, distances = contexts[state_id]
            sample = samples[c['sample_index']]
            if sample['index'] not in distances:
                distances[sample['index']] = material_distances(np.asarray(state['source']['rest_positions']), np.asarray(state['triangles']).reshape(-1, 3), np.asarray(state['triangles']).reshape(-1, 3)[sample['face']], sample['weights'])
            for roll in range(-180, 180, args.roll_step):
                R = roll_tool(c['rotation_matrix'], roll)
                key = (state_id, sample['index'], tuple(np.round(R.ravel(), 8)))
                if key in seen:
                    continue
                seen.add(key)
                value, reason = geometric_check(k, state, sample, R, skin, cloth, settings, distances[sample['index']])
                local['tested'] += 1
                local[reason] += 1
                if value:
                    value.update(number=number, source=row['source'], source_kind=state_id,
                        base_candidate=c, additional_roll_deg=roll,
                        source_state_sha256=hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest())
                    accepted.append(value)
        counts.update(local)
        write(output, dict(number=number, counts=dict(local), accepted=accepted, roll_step_deg=args.roll_step))
        print('GEOMETRY', number, dict(local), flush=True)
    records = [json.loads(p.read_text()) for p in sorted((args.output / 'geometry').glob('pose-*.json'))]
    total = Counter()
    for r in records:
        total.update(r['counts'])
    write(args.output / 'geometry-summary.json', dict(counts=dict(total), records=len(records),
        roll_step_deg=args.roll_step, scope='Additional full-circle roll about existing approach directions; original and bulged saved cloth states',
        accepted=[dict(file=str((args.output/'geometry'/f"pose-{r['number']:02d}.json").relative_to(ROOT)), candidate_index=i, number=r['number'], additional_roll_deg=v['additional_roll_deg']) for r in records for i,v in enumerate(r['accepted'])]))


def state_for(record):
    if 'state_file' in record:
        return json.loads((ROOT/record['state_file']).read_text())
    c, state, _ = target_data(record)
    if 'source' not in state:
        prior = json.loads((ROOT / f"artifacts/right-grasp-geometry/pose-{record['number']:02d}/report.json").read_text())
        state = dict(state, source=json.loads((ROOT / prior['input_state']).read_text())['source'])
    return state


def grid(args):
    """Broaden approach directions and grasp points as well as independent roll."""
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text())
    robot=json.loads(ROBOT.read_text())
    for number in args.numbers or [10,38,50]:
        output=args.output/'geometry'/f'pose-{number:02d}.json'
        if output.exists():
            continue
        state_file=f'artifacts/bulged-cuff/pose-{number:02d}/state.json'
        grid_settings=dict(number=number,theta_step=args.theta_step,roll_step=args.roll_step,depths_m=args.depths,
            state_sha256=hashlib.sha256((ROOT/state_file).read_bytes()).hexdigest())
        settings_file=args.output/f'grid-settings-{number:02d}.json'
        if args.resume and (args.output/f'grid-candidates-{number:02d}.json').exists() and not settings_file.exists():
            raise ValueError('Resume requires saved grid settings and source hash')
        if settings_file.exists() and json.loads(settings_file.read_text())!=grid_settings:
            raise ValueError('Output directory contains a different grid or source state')
        write(settings_file,grid_settings)
        state=json.loads((ROOT/state_file).read_text())
        scene=overlay_robot(load(number),robot);scene['particles_world']=state['particles_world']
        k=Kinematics(URDF,scene);skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles'])
        samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{number:02d}/samples.json').read_text())
        faces=np.asarray(state['triangles']).reshape(-1,3);rest=np.asarray(state['source']['rest_positions'])
        accepted=[];counts=Counter();began=time.monotonic();completed=0
        selected=[s for s in samples if not s['theta_index']%args.theta_step and any(abs(s['depth_m']-v)<1e-8 for v in args.depths)]
        checkpoint=args.output/f'grid-candidates-{number:02d}.json'
        if args.resume and checkpoint.exists():
            saved=json.loads(checkpoint.read_text());accepted=saved['accepted'];counts=Counter(saved['counts'])
            per_point=12*11*(360//args.roll_step)
            if counts['tested']%per_point:
                raise ValueError('Checkpoint ends inside a material point')
            completed=counts['tested']//per_point
            if completed>len(selected):
                raise ValueError('Checkpoint is larger than the requested grid')
        for sample in selected[completed:]:
            dist=material_distances(rest,faces,faces[sample['face']],sample['weights'])
            for yaw in range(0,360,30):
                for tilt in range(-75,76,15):
                    base=orientation(sample,yaw,tilt,1)
                    for roll in range(-180,180,args.roll_step):
                        R=roll_tool(base,roll)
                        result,reason=geometric_check(k,state,sample,R,skin,cloth,settings,dist)
                        counts['tested']+=1;counts[reason]+=1
                        if result:
                            c=dict(sample_index=sample['index'],yaw_deg=yaw,tilt_deg=tilt,closing_sign=1,rotation_matrix=base.tolist(),grasp_center_unity_world_m=sample['position'])
                            result.update(number=number,source=state_file,state_file=state_file,source_kind='bulged',base_candidate=c,additional_roll_deg=roll)
                            accepted.append(result)
            write(args.output/f'grid-progress-{number:02d}.json',dict(number=number,sample_index=sample['index'],counts=dict(counts),accepted=len(accepted),elapsed_s=time.monotonic()-began))
            # Checkpoint accepted poses after each point for bounded resume loss.
            write(args.output/f'grid-candidates-{number:02d}.json',dict(number=number,counts=dict(counts),accepted=accepted))
            print('GRID',number,sample['index'],dict(counts),'seconds',round(time.monotonic()-began,1),flush=True)
        write(output,dict(number=number,counts=dict(counts),accepted=accepted,roll_step_deg=args.roll_step,
            theta_step=args.theta_step,depths_m=args.depths,elapsed_s=time.monotonic()-began))
    records=[json.loads(p.read_text()) for p in sorted((args.output/'geometry').glob('pose-*.json'))]
    counts=Counter()
    for r in records:counts.update(r['counts'])
    write(args.output/'geometry-summary.json',dict(counts=dict(counts),records=len(records),roll_step_deg=args.roll_step,
        scope='Independent roll, full yaw and tilt grid, and expanded material points on bulged states',
        accepted=[dict(file=str((args.output/'geometry'/f"pose-{r['number']:02d}.json").relative_to(ROOT)),candidate_index=i,number=r['number'],additional_roll_deg=v['additional_roll_deg']) for r in records for i,v in enumerate(r['accepted'])]))


def search(args):
    summary = json.loads((args.geometry_summary or args.output / 'geometry-summary.json').read_text())
    # New rolls first; preserve both jaw assignments and all ankle states.
    records = summary['accepted']
    records.sort(key=lambda v: (abs(v['additional_roll_deg']) in (0, 180), abs(v['additional_roll_deg']), v['number']))
    rng = np.random.default_rng(20261105)
    robot = json.loads(ROBOT.read_text())
    contexts = {}
    passes = 0
    for index, ref in enumerate(records):
        if index % args.shards != args.shard_index:
            continue
        rng=np.random.default_rng(20261105+index)
        output = args.output / 'search' / f'target-{index:04d}.json'
        if output.exists():
            cached = json.loads(output.read_text())
            passes += any(a.get('trajectory', {}).get('passed') for a in cached['attempts'])
            if passes >= args.max_routes:
                break
            continue
        record = json.loads((ROOT / ref['file']).read_text())['accepted'][ref['candidate_index']]
        state = state_for(record)
        key = (record['number'], record['source_kind'])
        if key not in contexts:
            scene = overlay_robot(load(record['number']), robot)
            scene['particles_world'] = state['particles_world']
            k = Kinematics(URDF, scene)
            checker = AuditedScene(k, scene, state['triangles'], FAST_MANIFEST, ROOT)
            dl, du, contract = placement_bounds(scene, state,edge_margin=args.central_edge_margin)
            if args.max_forward_delta_z is not None:
                du=du.copy();du[2]=min(du[2],args.max_forward_delta_z)
                contract=dict(contract,translation_upper_m=du.tolist(),forward_search_restriction_m=args.max_forward_delta_z)
            contexts[key] = k, checker, dl, du, contract
        k, checker, dl, du, contract = contexts[key]
        pad = np.deg2rad(5)
        qlo, qhi = k.lower+pad, k.upper-pad
        qhi = qhi.copy()
        qhi[0] = min(qhi[0], np.deg2rad(109.5))
        lower, upper = np.r_[qlo, dl], np.r_[qhi, du]
        c = record['candidate']
        p, R = np.asarray(c['grasp_center_unity_world_m']), np.asarray(c['rotation_matrix'])
        attempts = []
        for seed in range(args.seeds):
            warm=ref.get('warm_start') if seed==0 else None
            q = np.asarray(warm['q']) if warm else (k.start.copy() if seed == 0 else rng.uniform(qlo, qhi))
            q = np.clip(q, qlo+1e-9, qhi-1e-9)
            delta = np.clip(np.asarray(warm['translation_m']) if warm else k.pose(q)[0]-p, dl+1e-9, du-1e-9)
            fit = least_squares(lambda x: pose_constraint(k,p,R,x)*np.r_[np.full(3,5.),np.full(3,.75)],
                np.r_[q,delta], jac=lambda x:pose_constraint(k,p,R,x,True)[1]*np.r_[np.full(3,5.),np.full(3,.75)][:,None],
                bounds=(lower,upper), max_nfev=150, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            x = fit.x
            res = pose_constraint(k,p,R,x)
            a = dict(seed=seed, q=x[:7].tolist(), translation_m=x[7:].tolist(),
                position_error_m=float(np.linalg.norm(res[:3])), orientation_error_deg=float(np.rad2deg(np.linalg.norm(res[3:]))))
            attempts.append(a)
            if a['position_error_m']>1e-4 or a['orientation_error_deg']>.5:
                a['stage']='ik_failed'
                continue
            if args.ik_only:
                a['stage']='ik_passed'
                continue
            for opening in sorted(record['valid_openings'], key=lambda v: -v['gap_m']):
                gap = opening['gap_m']
                constraint = Constraints(k,checker,gap)
                # Restore collisions while keeping the pose and centered bounds.
                g = constraint.fun(x)
                if g.min() < -1e-9 and not args.skip_restoration:
                    weights=np.r_[np.full(3,5.),np.full(3,.75)]
                    def restore(z):return np.r_[pose_constraint(k,p,R,z)*weights,5*np.minimum(constraint.fun(z),0)]
                    def jac(z):
                        v,J=constraint.evaluate(z)
                        return np.vstack([pose_constraint(k,p,R,z,True)[1]*weights[:,None],5*J*(v<0)[:,None]])
                    restored=least_squares(restore,x,jac=jac,bounds=(lower,upper),max_nfev=args.iterations,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                    x=restored.x
                    preferred=np.r_[np.clip(k.start,qlo,qhi),np.array([(dl[0]+du[0])/2,0.,-.45])]
                    fit=minimize(lambda z:.002*np.sum((z[:7]-preferred[:7])**2)+.02*np.sum((z[7:]-preferred[7:])**2),x,
                        jac=lambda z:np.r_[.004*(z[:7]-preferred[:7]),.04*(z[7:]-preferred[7:])],method='SLSQP',bounds=list(zip(lower,upper)),
                        constraints=[dict(type='eq',fun=lambda z:pose_constraint(k,p,R,z),jac=lambda z:pose_constraint(k,p,R,z,True)[1]),dict(type='ineq',fun=constraint.fun,jac=constraint.jac)],options=dict(maxiter=args.iterations,ftol=1e-10))
                    x=fit.x
                res=pose_constraint(k,p,R,x)
                a.update(q=x[:7].tolist(),translation_m=x[7:].tolist(),gap_m=gap,
                    position_error_m=float(np.linalg.norm(res[:3])),orientation_error_deg=float(np.rad2deg(np.linalg.norm(res[3:]))),
                    endpoint=checker.slacks(x[:7],gap,x[7:],details=True),initial_pose=checker.slacks(k.start,gap,x[7:],details=True))
                good=a['position_error_m']<=1e-4 and a['orientation_error_deg']<=.5 and a['endpoint']['passed'] and a['initial_pose']['passed']
                a['stage']='endpoint_passed' if good else 'endpoint_failed'
                if good:
                    a['trajectory']=trace(k,checker,c,gap,x)
                    print('ENDPOINT',index,record['number'],record['additional_roll_deg'],a['trajectory']['stage'],x[7:],flush=True)
                    if a['trajectory']['passed']:
                        passes+=1
                        break
            if a.get('trajectory',{}).get('passed'):
                break
        write(output,dict(index=index,number=record['number'],geometry_reference=ref,geometry_record=record,
            source=record['source'],central_contract=contract,attempts=attempts,
            search_geometry=str(FAST_MANIFEST.relative_to(ROOT)),
            settings=dict(seeds=args.seeds,iterations=args.iterations,random_seed=20261105+index,
                skip_restoration=args.skip_restoration,ik_only=args.ik_only,max_forward_delta_z=args.max_forward_delta_z),executed=False))
        print('SEARCH',index,'/',len(records),'pose',record['number'],'roll',record['additional_roll_deg'],'routes',passes,'stages',dict(Counter(a['stage'] for a in attempts)),flush=True)
        if passes>=args.max_routes:
            break
    saved=[json.loads(p.read_text()) for p in sorted((args.output/'search').glob('*.json'))]
    summary_name='search-summary.json' if args.shards==1 else f'search-summary-shard-{args.shard_index}.json'
    write(args.output/summary_name,dict(targets=len(saved),available_geometry_candidates=len(records),
        trials=sum(len(r['attempts']) for r in saved),sampled_route_passes=sum(a.get('trajectory',{}).get('passed',False) for r in saved for a in r['attempts']),
        complete=len(saved)==len(records),stopped_after_requested_routes=passes>=args.max_routes,executed=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['geometry','grid','search'])
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/centered-reaching')
    ap.add_argument('--numbers',type=int,nargs='+')
    ap.add_argument('--roll-step',type=int,default=15)
    ap.add_argument('--theta-step',type=int,default=2)
    ap.add_argument('--depths',type=float,nargs='+',default=[.01,.015,.02])
    ap.add_argument('--seeds',type=int,default=4)
    ap.add_argument('--iterations',type=int,default=80)
    ap.add_argument('--max-routes',type=int,default=3)
    ap.add_argument('--geometry-summary',type=Path)
    ap.add_argument('--max-forward-delta-z',type=float)
    ap.add_argument('--ik-only',action='store_true')
    ap.add_argument('--skip-restoration',action='store_true',help='Screen multistart IK and paths without collision-restoration optimization')
    ap.add_argument('--resume',action='store_true',help='Resume a grid from completed material-point checkpoints')
    ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--shard-index',type=int,default=0)
    ap.add_argument('--central-edge-margin',type=float,default=.01,help='Extra lateral margin inside starting gripper planes, metres')
    args=ap.parse_args()
    args.output=args.output.resolve()
    if args.roll_step<=0 or 360%args.roll_step:
        ap.error('--roll-step must be a positive divisor of 360')
    if args.shards<1 or not 0<=args.shard_index<args.shards:
        ap.error('--shard-index must be within --shards')
    if args.central_edge_margin<0:
        ap.error('--central-edge-margin must be nonnegative')
    {'geometry':geometry,'grid':grid,'search':search}[args.stage](args)


if __name__=='__main__':
    main()
