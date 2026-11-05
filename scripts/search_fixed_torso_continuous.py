#!/usr/bin/env python3
"""Use collision-aware continuous wrist orientations near fixed-torso IK limits."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import yaml
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_fixed_torso_starts import context,retract_route
from search_centered_reaching import state_for,write,geometric_check
from search_compound_reaching import pose_constraint
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');ap.add_argument('--index',type=int);ap.add_argument('--folder',default='continuous');ap.add_argument('--extra-clearance',type=float,default=0.);ap.add_argument('--body-clearance',type=float);a=ap.parse_args();a.output=a.output.resolve()
    near=json.loads((a.output/'near-ik-polish.json').read_text())['results'];layouts=json.loads((a.output/'initial-configurations.json').read_text())['candidates'];settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text())
    for ni,source in enumerate(near):
        for mode in (0,1):
            index=ni*2+mode
            if a.index is not None and index!=a.index:continue
            output=a.output/a.folder/f'target-{index:04d}.json'
            if output.exists():continue
            row=json.loads((ROOT/source['source']).read_text());ref=row['geometry_reference'];gr=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']];state=state_for(gr);n=row['number'];k,checker,dl,du,contract,scene=context(n,state)
            initial=layouts[source['layout_index']];checker.park_left(initial['left_q_rad']);p=np.array(gr['candidate']['grasp_center_unity_world_m']);desired=np.array(gr['candidate']['rotation_matrix']);gap=max(v['gap_m'] for v in gr['valid_openings']);pad=np.deg2rad(5);lo=np.r_[k.lower+pad,dl];hi=np.r_[k.upper-pad,du]
            x0=np.clip(np.r_[source['q'],source['translation_m']],lo+1e-10,hi-1e-10);preferred=x0.copy();preferred[9]-=.02 if mode==0 else .05;cache={};calls=0
            clearance=np.full(len(checker.pairs)+len(checker.floor_minima),a.extra_clearance)
            if a.body_clearance is not None:
                for pi,(aa,bb,mm) in enumerate(checker.pairs):
                    if not (aa.startswith('right_gripper/') or bb.startswith('right_gripper/')):continue
                    other=bb if aa.startswith('right_gripper/') else aa
                    if other in checker.groups and not other.startswith('right_gripper/') and other not in {'right_arm/link_5','right_arm/link_6','right_arm/link_7'}:clearance[pi]=a.body_clearance
            def constraints(x):
                nonlocal calls
                if 'x' not in cache or not np.array_equal(cache['x'],x):
                    calls+=1;g,J=checker.slacks(x[:7],gap,x[7:],with_jacobian=True);cache.update(x=x.copy(),g=g,J=J)
                return cache['g'],cache['J']
            def objective(x):
                e,J=pose_constraint(k,p,desired,x,True);w=1. if mode==0 else .1
                return w*float(e[3:]@e[3:])+.01*np.sum((x[7:]-preferred[7:])**2)+1e-6*np.sum((x[:7]-x0[:7])**2),2*w*J[3:].T@e[3:]+np.r_[2e-6*(x[:7]-x0[:7]),.02*(x[7:]-preferred[7:])]
            began=time.monotonic();fit=minimize(objective,x0,jac=True,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=lambda x:pose_constraint(k,p,desired,x)[:3],jac=lambda x:pose_constraint(k,p,desired,x,True)[1][:3]),dict(type='ineq',fun=lambda x:constraints(x)[0]-clearance,jac=lambda x:constraints(x)[1])],options=dict(maxiter=240,ftol=1e-12));x=fit.x;e=pose_constraint(k,p,desired,x);actual,R=k.pose(x[:7]);end=checker.slacks(x[:7],gap,x[7:],details=True)
            attempt=dict(q=x[:7].tolist(),left_q_rad=initial['left_q_rad'],translation_m=x[7:].tolist(),gap_m=gap,position_error_m=float(np.linalg.norm(e[:3])),orientation_change_deg=float(np.rad2deg(np.linalg.norm(e[3:]))),rotation_matrix=R.tolist(),endpoint=end,solver_success=bool(fit.success),solver_message=str(fit.message),constraint_evaluations=calls,stage='endpoint_failed')
            record=dict(index=index,number=n,geometry_reference=ref,geometry_record=gr,central_contract=contract,torso_joints_rad=checker.reference_torso,left_gap_m=.04,attempts=[attempt],input=source,extra_clearance_m=a.extra_clearance,body_clearance_m=a.body_clearance,elapsed_s=time.monotonic()-began,executed=False)
            if np.linalg.norm(e[:3])<=1e-4 and end['passed']:
                samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/samples.json').read_text());sample=samples[gr['candidate']['sample_index']];faces=np.array(state['triangles']).reshape(-1,3);dist=material_distances(np.array(state['source']['rest_positions']),faces,faces[sample['face']],sample['weights'])
                geo,reason=geometric_check(k,state,sample,R,SkinBody(state['source']['skin_surface']),ClothScreen(state['particles_world'],state['triangles']),settings,dist);attempt['geometry_stage']=reason;attempt['stage']='geometry_failed'
                if geo:
                    record['geometry_record']=dict(gr,**geo,generator='fixed_torso_collision_guided_continuous');attempt['stage']='endpoint_passed'
                    for opening in geo['valid_openings']:
                        gap=opening['gap_m'];check=checker.slacks(x[:7],gap,x[7:],details=True)
                        if not check['passed']:continue
                        route=retract_route(k,checker,geo['candidate'],x[:7],x[7:],gap);attempt.update(gap_m=gap,endpoint=check,trajectory=route)
                        if route['passed']:break
            write(output,record);print('CONTINUOUS',index,n,attempt['stage'],attempt.get('geometry_stage'),attempt['position_error_m'],attempt['orientation_change_deg'],end['minimum_slack_m'],attempt.get('trajectory',{}).get('stage'),flush=True)


if __name__=='__main__':main()
