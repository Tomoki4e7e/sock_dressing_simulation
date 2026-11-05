#!/usr/bin/env python3
"""Propose continuous wrist orientations from centered, attainable arm poses.

Unlike the direction grid, tool rotation is a soft objective here. Only the
position constraint is exact during proposal generation. Every proposed actual
rotation must subsequently pass the full cloth/skin geometric inclusion test.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.optimize import least_squares,minimize
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import state_for,write,geometric_check,URDF,ROBOT,FAST_MANIFEST
from search_compound_reaching import pose_constraint,trace
from search_translated_reaching import load
from apron_reaching_common import overlay_robot,AuditedScene
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.centered_reaching import placement_bounds
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances


def diagnose_rotating_endpoints(records,args):
    """Drop the straight-sweep gate, retaining endpoint insertion/closure rules."""
    from reevaluate_geometric_closure import insertion
    contexts={};total=Counter();valid=[];tested=0
    for path in sorted((args.output/'proposals').glob('target-*.json')):
        index=int(path.stem.split('-')[1]);row=json.loads(path.read_text());r=records[index];n=r['number'];state=state_for(r)
        if not any(a['stage']=='opening_or_approach' for a in row['proposals']):continue
        if n not in contexts:
            scene=overlay_robot(load(n),json.loads(ROBOT.read_text()));scene['particles_world']=state['particles_world'];k=Kinematics(URDF,scene)
            contexts[n]=(k,SkinBody(state['source']['skin_surface']),ClothScreen(state['particles_world'],state['triangles']),json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/samples.json').read_text()))
        k,skin,cloth,samples=contexts[n];sample=samples[r['candidate']['sample_index']];p=np.asarray(r['candidate']['grasp_center_unity_world_m'])
        for ai,a in enumerate(row['proposals']):
            if a['stage']!='opening_or_approach':continue
            tested+=1;R=np.asarray(a['rotation_matrix']);c=dict(sample_index=sample['index'],rotation_matrix=R.tolist(),grasp_center_unity_world_m=p.tolist());gaps=[]
            for gap in (.0015,.002,.003,.004,.005,.0075,.01,.015,.02,.025,.03,.04,.05):
                if not skin.inspect(k.hand.bank(gap),R,p,.001)[0]:total['endpoint_skin']+=1;continue
                if cloth.intersections(k.hand.bank(gap),R,p):total['endpoint_cloth']+=1;continue
                if not insertion(k.hand,skin,dict(c,gap_m=gap),sample)['passed']:total['insertion']+=1;continue
                if not skin.inspect(k.hand.bank(gap,'closure',pinch_gap=.001),R,p,.001)[0]:total['closure_skin']+=1;continue
                gaps.append(gap)
            if gaps:valid.append(dict(proposal_file=str(path.relative_to(ROOT)),proposal_index=ai,record=r,endpoint_gaps_m=gaps))
    write(args.output.parent/'rotating-approach-diagnostics.json',dict(counts=dict(total),tested_proposals=tested,valid_endpoints=valid,
        scope='Only endpoint/opening/insertion/closure checks; a failed endpoint cannot be repaired by a rotating approach route'))
    print('ROTATING APPROACH ENDPOINTS',tested,dict(total),'accepted',len(valid),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--geometry-summary',type=Path,default=ROOT/'artifacts/centered-reaching/geometry-summary.json')
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/centered-reaching/continuous')
    ap.add_argument('--seeds',type=int,default=6)
    ap.add_argument('--numbers',type=int,nargs='+')
    ap.add_argument('--endpoint-diagnostics',action='store_true')
    args=ap.parse_args();args.output=args.output.resolve()
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());robot=json.loads(ROBOT.read_text())
    refs=json.loads(args.geometry_summary.read_text())['accepted'];records=[];seen=set()
    for ref in refs:
        r=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']]
        if r['source_kind']!='bulged' or (args.numbers and r['number'] not in args.numbers):continue
        # Both jaw assignments are included through separate rotation targets.
        c=r['candidate'];key=(r['number'],c['sample_index'],tuple(np.round(np.asarray(c['rotation_matrix']).ravel(),7)))
        if key in seen:continue
        seen.add(key);records.append(r)
    if args.endpoint_diagnostics:
        diagnose_rotating_endpoints(records,args)
        return
    rng=np.random.default_rng(20261106);contexts={};total=Counter()
    for index,record in enumerate(records):
        path=args.output/'proposals'/f'target-{index:04d}.json'
        if path.exists():continue
        state=state_for(record);number=record['number']
        if number not in contexts:
            scene=overlay_robot(load(number),robot);scene['particles_world']=state['particles_world'];k=Kinematics(URDF,scene)
            checker=AuditedScene(k,scene,state['triangles'],FAST_MANIFEST,ROOT);dl,du,contract=placement_bounds(scene,state)
            # This supplement explicitly targets the forward, start-clear region.
            du=du.copy();du[2]=-.25
            samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{number:02d}/samples.json').read_text())
            contexts[number]=(k,checker,dl,du,contract,samples,SkinBody(state['source']['skin_surface']),ClothScreen(state['particles_world'],state['triangles']))
        k,checker,dl,du,contract,samples,skin,cloth=contexts[number]
        c=record['candidate'];p=np.asarray(c['grasp_center_unity_world_m']);desired=np.asarray(c['rotation_matrix']);sample=samples[c['sample_index']]
        faces=np.asarray(state['triangles']).reshape(-1,3);dist=material_distances(np.asarray(state['source']['rest_positions']),faces,faces[sample['face']],sample['weights'])
        ql=k.lower+np.deg2rad(5);qu=k.upper-np.deg2rad(5);qu[0]=min(qu[0],np.deg2rad(109.5));lower=np.r_[ql,dl];upper=np.r_[qu,du]
        proposals=[];counts=Counter()
        for seed in range(args.seeds):
            q=np.clip(k.start,ql+1e-9,qu-1e-9) if seed==0 else rng.uniform(ql,qu)
            delta=np.clip(k.pose(q)[0]-p,dl+1e-9,du-1e-9);x0=np.r_[q,delta]
            fit=least_squares(lambda x:pose_constraint(k,p,desired,x)[:3],x0,jac=lambda x:pose_constraint(k,p,desired,x,True)[1][:3],bounds=(lower,upper),max_nfev=100)
            def objective(x):
                e,J=pose_constraint(k,p,desired,x,True)
                return .5*np.sum(e[3:]**2)+.0001*np.sum((x[:7]-k.start)**2),J[3:].T@e[3:]+np.r_[.0002*(x[:7]-k.start),np.zeros(3)]
            opt=minimize(objective,fit.x,jac=True,method='SLSQP',bounds=list(zip(lower,upper)),
                constraints=[dict(type='eq',fun=lambda x:pose_constraint(k,p,desired,x)[:3],jac=lambda x:pose_constraint(k,p,desired,x,True)[1][:3])],options=dict(maxiter=100,ftol=1e-10))
            actual,R=k.pose(opt.x[:7]);counts['proposals']+=1
            error=float(np.linalg.norm(actual-p-opt.x[7:]));entry=dict(seed=seed,q=opt.x[:7].tolist(),translation_m=opt.x[7:].tolist(),position_error_m=error,rotation_matrix=R.tolist(),orientation_deviation_deg=float(np.rad2deg(np.linalg.norm(pose_constraint(k,p,desired,opt.x)[3:]))))
            proposals.append(entry)
            if error>1e-4:entry['stage']='position_failed';counts['position_failed']+=1;continue
            geometry,reason=geometric_check(k,state,sample,R,skin,cloth,settings,dist)
            entry['stage']=reason;counts[reason]+=1
            if geometry:
                continuous=dict(geometry,number=number,source=f'artifacts/bulged-cuff/pose-{number:02d}/state.json',state_file=f'artifacts/bulged-cuff/pose-{number:02d}/state.json',source_kind='bulged',additional_roll_deg=0.,generator='continuous_centered_arm_orientation',base_candidate=c)
                for opening in geometry['valid_openings']:
                    gap=opening['gap_m'];end=checker.slacks(opt.x[:7],gap,opt.x[7:],details=True);start=checker.slacks(k.start,gap,opt.x[7:],details=True)
                    if not end['passed'] or not start['passed']:continue
                    route=trace(k,checker,geometry['candidate'],gap,opt.x);entry['trajectory']=route
                    result=dict(index=index,number=number,source=continuous['source'],geometry_record=continuous,central_contract=contract,attempts=[dict(entry,gap_m=gap,endpoint=end,initial_pose=start)],executed=False)
                    write(args.output/'search'/f'target-{index:04d}.json',result)
                    if route['passed']:
                        write(path,dict(number=number,counts=dict(counts),proposals=proposals));print('CONTINUOUS ROUTE',number,index,flush=True);return
        total.update(counts);write(path,dict(number=number,counts=dict(counts),proposals=proposals))
        print('CONTINUOUS',index,len(records),number,dict(counts),flush=True)
    saved=[json.loads(p.read_text()) for p in (args.output/'proposals').glob('*.json')];total=Counter()
    for r in saved:total.update(r['counts'])
    write(args.output/'summary.json',dict(targets=len(saved),counts=dict(total),scope='Continuous attainable wrist orientation closest to each saved geometric target, in the forward central region',executed=False))


if __name__=='__main__':main()
