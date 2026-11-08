#!/usr/bin/env python3
"""Recheck hand-only candidates at bounded IK with a common candidate arm pair."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,copy,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from shapely.errors import GEOSException
import time
import numpy as np,yaml
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,REGISTERED_MANIFEST,geometric_check
from search_fixed_torso_starts import context
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances


def one(payload):
    row,geometries,initial_path,out,pass_name,guided,loose=payload;initial=yaml.safe_load(initial_path.read_text());state=load_state(ROOT,row);k,scene,*_=context(row['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=initial['left_foot_floor_allowance_m']);scene.use_spatial_aabb=True;delta=np.array(initial['translation_m']);skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());rows=[]
    for side in ('left','right'):
        scene.park_left(initial['left_q_rad'],.04);model=ControlledArmKinematics(k,side,initial['right_q_rad'],.04);checker=ControlledCollisionScene(scene,side,initial['right_q_rad'],.04);comfortable=copy.copy(model);comfortable.lower=model.lower+np.deg2rad(5);comfortable.upper=model.upper-np.deg2rad(5);start=np.array(initial[side+'_q_rad']);rng=np.random.default_rng(20261110+row['number']+(side=='right'));found=[];counts={};attempts=[]
        began=time.monotonic();position_limit=.010 if loose else 2e-6;rotation_limit=30 if loose else .05
        prior=out/'arm-guided-endpoints'/side/f'{row["id"]}.json'
        if loose and prior.exists():
            cached=json.loads(prior.read_text())
            if cached['accepted'] and cached['initial_sha256']==sha256(initial_path) and cached['condition']==row:
                write(out/pass_name/side/f'{row["id"]}.json',dict(cached,reused_verified_endpoint=True));rows.append(dict(id=row['id'],side=side,accepted=len(cached['accepted']),counts=cached['counts']));continue
        seeds=[start]
        for root in [out/'start-candidates',ROOT/'artifacts/common-opening-approach/negative-wrist-start']:
            path=root/f'{side}-start-candidates.json'
            if path.exists():seeds.extend(np.array(r['q_rad']) for r in json.loads(path.read_text())['candidates'][::6])
        for path in sorted((ROOT/'artifacts/diverse-left-reaching/final/routes').glob(f'pose-{row["number"]:02d}-held-A*-trajectory.json')):
            d=json.loads(path.read_text())
            if d['moving_side']==side:seeds.insert(0,np.array(d['q_rad'][-1]))
        seeds+=list(rng.uniform(comfortable.lower,comfortable.upper,(8,7)))
        for index,source in enumerate(geometries):
            candidate=source['candidate'];p=np.array(candidate['position'])+delta;R=np.array(candidate['rotation']);sample=candidate['sample'];dist=material_distances(rest,faces,faces[sample['face']],sample['weights']);opening=candidate['geometry']['valid_openings'][0]['gap_m'] if candidate.get('geometry') else .04
            if loose and time.monotonic()-began>30:break
            for seed in seeds:
                if loose and time.monotonic()-began>30:break
                seed=np.clip(seed,comfortable.lower+1e-9,comfortable.upper-1e-9)
                sol=comfortable.solve(p,R,seed);counts['ik']=counts.get('ik',0)+1
                q=np.array(sol['q']);relaxed=loose
                if sol['position_error_m']>position_limit or sol['orientation_error_deg']>rotation_limit:
                    if not guided:continue
                    closing=R[:,1]
                    def residual(q):
                        actual,rotation=model.pose(q);return np.r_[actual-p,.15*(rotation[:,1]-closing)]
                    def jacobian(q):
                        _,rotation,Jp,Ja=model.fast_pose(q);return np.vstack([Jp,.15*np.cross(Ja.T,rotation[:,1]).T])
                    fit=least_squares(residual,seed,jac=jacobian,bounds=(comfortable.lower,comfortable.upper),max_nfev=120,ftol=1e-11,gtol=1e-11,xtol=1e-11)
                    if np.linalg.norm(fit.fun[:3])>position_limit:continue
                    q=fit.x;relaxed=True;counts['arm_guided']=counts.get('arm_guided',0)+1
                actual,actual_R=model.pose(q)
                try:geometry,reason=geometric_check(model,state,dict(sample,position=(actual-delta).tolist()),actual_R,skin,cloth,settings,dist,approach_distance=0,opening_gaps=None if relaxed else [opening])
                except (ValueError,GEOSException):geometry=None;reason='numerical_rejected'
                if loose and geometry is None and reason=='opening_or_approach':
                    lo=.001;hi=.05
                    for _ in range(15):
                        mid=(lo+hi)/2
                        if skin.inspect(model.hand.bank(mid,'closure',pinch_gap=.001),actual_R,actual-delta,.001)[0]:lo=mid
                        else:hi=mid
                    gap=max(.00100001,lo-2e-7)
                    try:geometry,reason=geometric_check(model,state,dict(sample,position=(actual-delta).tolist()),actual_R,skin,cloth,settings,dist,approach_distance=0,opening_gaps=[gap])
                    except (ValueError,GEOSException):geometry=None;reason='numerical_rejected'
                    if len(model.hand._banks)>200:model.hand._banks={k:v for k,v in model.hand._banks.items() if k[0]==.001}
                if not geometry:counts[reason]=counts.get(reason,0)+1;continue
                for option in sorted(geometry['valid_openings'],key=lambda r:-r['gap_m']):
                    gap=option['gap_m'];checker.prepare(q,gap)
                    if not checker.is_clear(q,gap,delta):counts['body_rejected']=counts.get('body_rejected',0)+1;continue
                    checks=[checker.slacks(q,float(g),delta,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,21)]
                    if not all(c['passed'] for c in checks):counts['closure_rejected']=counts.get('closure_rejected',0)+1;continue
                    found.append(dict(q_rad=q.tolist(),opening_gap_m=gap,geometry=geometry,sample=sample,position_world_m=actual.tolist(),rotation=actual_R.tolist(),source=source['source'],closure_checks=checks,arm_guided_rotation=relaxed));break
                if found:break
            if found:break
        result=dict(condition=row,side=side,initial_source=str(initial_path.relative_to(ROOT)),initial_sha256=sha256(initial_path),geometry_manifest_sha256=sha256(REGISTERED_MANIFEST),criteria_sha256=sha256(ROOT/'config/geometric_closure.yaml'),accepted=found,counts=counts,loose_proposal_search=loose,proposal_budget_exhausted=bool(loose and time.monotonic()-began>30),route_verified=False,native_verified=False,scope='Candidate pair fixed; actual bounded IK pose plus whole-scene open and 21-width closure checks. No path or native replay inferred')
        write(out/pass_name/side/f'{row["id"]}.json',result);rows.append(dict(id=row['id'],side=side,accepted=len(found),counts=counts));print('ARM GRASP ENDPOINT',row['id'],side,len(found),counts,flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/common-grasp-start-search');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--pass-name',default='arm-endpoints');ap.add_argument('--guided',action='store_true');ap.add_argument('--loose',action='store_true');a=ap.parse_args();out=a.output.resolve();groups={};inputs=inventory(ROOT)
    for folder in [out,out/'transport',out/'transfer-all',out/'symmetry',out/'foot-reflection']:
        for path in sorted((folder/'geometry').glob('*.json')):
            d=json.loads(path.read_text());row=d['condition']
            for c in d['accepted']+(d.get('open_promising',[])[:2]+d.get('promising',[])[:2] if a.loose else []):groups.setdefault(row['id'],[]).append(dict(candidate=c,source=str(path.relative_to(ROOT))))
    qualified_geometry_ids={identifier for identifier,records in groups.items() if any(r['candidate'].get('geometry') for r in records)}
    if a.loose:
        from sock_dressing_simulation.grasp_geometry import candidate_samples,orientation
        for row in inputs:
            if row['id'] in groups:continue
            for sample in candidate_samples(load_state(ROOT,row),16,(.005,.015)):
                groups.setdefault(row['id'],[]).append(dict(candidate=dict(sample=sample,position=sample['position'],rotation=orientation(sample,0,-10).tolist()),source=row['state']))
    initial_path=out/'start-candidates/initial.yaml';done=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,(r,groups[r['id']],initial_path,out,a.pass_name,a.guided,a.loose)) for r in inputs if r['id'] in groups]
        for f in as_completed(futures):done.extend(f.result());write(out/(a.pass_name+'-progress.json'),dict(rows=done,with_endpoint=sum(r['accepted']>0 for r in done),evaluated=len(done),required=140,states_without_qualified_tool_geometry_at_start=[r['id'] for r in inputs if r['id'] not in qualified_geometry_ids]))

if __name__=='__main__':main()
