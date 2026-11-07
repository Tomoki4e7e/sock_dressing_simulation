#!/usr/bin/env python3
"""Enforce exact TCP position before judging near-limit fixed-layout IK."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_fixed_torso_starts import context
from search_centered_reaching import state_for,write
from search_compound_reaching import pose_constraint


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/fixed-torso-start-search');a=ap.parse_args();a.output=a.output.resolve();ranked=[]
    for f in (a.output/'placed-ik').glob('*.json'):
        row=json.loads(f.read_text())
        for layout in row['layouts']:
            for ai,attempt in enumerate(layout['attempts']):
                if attempt['position_error_m']<.002 and attempt['orientation_error_deg']<2:
                    ranked.append((attempt['position_error_m']+.15*np.deg2rad(attempt['orientation_error_deg']),f,row,layout,ai,attempt))
    layouts=json.loads((a.output/'initial-configurations.json').read_text())['candidates'];results=[]
    for score,f,row,layout,ai,attempt in sorted(ranked,key=lambda v:v[0])[:24]:
        ref=row['geometry_reference'];gr=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']];state=state_for(gr);k,checker,dl,du,contract,scene=context(row['number'],state);initial=layouts[layout['layout_index']];checker.park_left(initial['left_q_rad']);d=np.array(layout['translation_m']);c=gr['candidate'];p=np.array(c['grasp_center_unity_world_m'])+d;R=np.array(c['rotation_matrix'])
        def residual(q,jac=False):return pose_constraint(k,p,R,np.r_[q,np.zeros(3)],jac)
        def objective(q):
            e,J=residual(q,True);return float(e[3:]@e[3:]),2*J[3:,:7].T@e[3:]
        fit=minimize(objective,np.array(attempt['q']),jac=True,method='SLSQP',bounds=list(zip(k.lower+np.deg2rad(5),k.upper-np.deg2rad(5))),constraints=[dict(type='eq',fun=lambda q:residual(q)[:3],jac=lambda q:residual(q,True)[1][:3,:7])],options=dict(maxiter=400,ftol=1e-14))
        e=residual(fit.x);gap=max(v['gap_m'] for v in gr['valid_openings']);check=checker.slacks(fit.x,gap,d,details=True)
        result=dict(number=row['number'],source=str(f.relative_to(ROOT)),layout_index=layout['layout_index'],attempt_index=ai,q=fit.x.tolist(),translation_m=d.tolist(),position_error_m=float(np.linalg.norm(e[:3])),orientation_error_deg=float(np.rad2deg(np.linalg.norm(e[3:]))),endpoint=check,solver_success=bool(fit.success),solver_message=str(fit.message));results.append(result)
        print('POLISH',row['number'],result['position_error_m'],result['orientation_error_deg'],check['passed'],flush=True)
    write(a.output/'near-ik-polish.json',dict(results=results))


if __name__=='__main__':main()
