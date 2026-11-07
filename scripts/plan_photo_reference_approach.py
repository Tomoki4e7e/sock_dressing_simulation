#!/usr/bin/env python3
"""Generate a short approach from the photo-like start, keeping the torso fixed.

This endpoint is a collision-free approach to the cuff. It is explicitly not
an accepted closed-jaw grasp; closure is recorded as a separate diagnostic.
"""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_photo_reference_reaching import photo_context
from search_centered_reaching import write,REGISTERED_MANIFEST
from search_compound_reaching import pose_constraint


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/photo-reference-reaching');ap.add_argument('--initial',type=Path);a=ap.parse_args();a.output=a.output.resolve()
    source=(a.initial or a.output/'initial-configurations.json').resolve();initial=json.loads(source.read_text())['candidates'][0];n=initial['number'];state=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/state.json').read_text());allowance=initial['left_foot_floor_allowance_m'];k,c,dl,du,contract,scene=photo_context(n,state,allowance)
    c.park_left(initial['left_q_rad'],initial['left_gap_m']);start=np.array(initial['right_q_rad']);p0,R=k.pose(start);d=np.array(initial['translation_m']);target=np.array(initial['cuff_center_world_m']);gap=initial['right_gap_m'];lo=k.lower+np.deg2rad(10);hi=k.upper-np.deg2rad(10)
    # The photograph calls for a nearby start. Follow small Cartesian steps
    # toward the cuff and retain only the collision-free prefix.
    directions=[target-p0,np.array([1.,0.,0.]),R[:,0]];routes=[]
    for di,direction in enumerate(directions):
        direction=direction/np.linalg.norm(direction);rows=[start];failure=None
        for distance in np.arange(.002,.06001,.002):
            p=p0+direction*distance;prev=rows[-1];cache={}
            def eq(q):return pose_constraint(k,p,R,np.r_[q,np.zeros(3)])
            def eqjac(q):return pose_constraint(k,p,R,np.r_[q,np.zeros(3)],True)[1][:,:7]
            def constraints(q):
                if 'q' not in cache or not np.array_equal(cache['q'],q):
                    g,J=c.slacks(q,gap,d,with_jacobian=True);cache.update(q=q.copy(),g=g,J=J[:,:7])
                return cache['g'],cache['J']
            comfortable=copy.copy(k);comfortable.lower=lo;comfortable.upper=hi;sol=comfortable.solve(p,R,prev);q=np.array(sol['q'])
            if not c.is_clear(q,gap,d) or sol['position_error_m']>1e-4 or sol['orientation_error_deg']>.5:
                opt=minimize(lambda q:np.sum((q-prev)**2),q,jac=lambda q:2*(q-prev),method='SLSQP',bounds=list(zip(lo,hi)),
                    constraints=[dict(type='eq',fun=eq,jac=eqjac),dict(type='ineq',fun=lambda q:constraints(q)[0]-.0005,jac=lambda q:constraints(q)[1])],options=dict(maxiter=60,ftol=1e-11));q=opt.x
            e=eq(q);check=c.slacks(q,gap,d,details=True)
            if np.linalg.norm(e[:3])>1e-4 or np.rad2deg(np.linalg.norm(e[3:]))>.5 or not check['passed']:
                failure=dict(distance_m=float(distance),check=check,position_error_m=float(np.linalg.norm(e[:3])));break
            segment=np.linspace(prev,q,max(2,int(np.ceil(np.max(abs(q-prev))/np.deg2rad(.25))))+1)
            if not all(c.is_clear(v,gap,d) for v in segment):failure=dict(distance_m=float(distance),stage='edge_collision');break
            rows.append(q)
        route=dict(direction_index=di,direction=direction.tolist(),distance_m=float((len(rows)-1)*.002),raw_q=[q.tolist() for q in rows],failure=failure,
            start_distance_to_cuff_center_m=float(np.linalg.norm(p0-target)),end_distance_to_cuff_center_m=float(np.linalg.norm(k.pose(rows[-1])[0]-target)));routes.append(route);write(a.output/'approach-planning.json',dict(routes=routes));print('PHOTO APPROACH',di,route['distance_m'],flush=True)
    viable=[r for r in routes if r['distance_m']>=.01 and r['end_distance_to_cuff_center_m']<=r['start_distance_to_cuff_center_m']-.01]
    if not viable:raise RuntimeError('No collision-free approach of at least 10 mm')
    route=min(viable,key=lambda r:r['end_distance_to_cuff_center_m']);qgoal=np.array(route['raw_q'][-1]);goal,R=k.pose(qgoal)
    _,registered,*_=photo_context(n,state,allowance,REGISTERED_MANIFEST);registered.park_left(initial['left_q_rad'],initial['left_gap_m'])
    checks=[registered.slacks(np.array(q),gap,d,details=True) for q in route['raw_q']]
    if not all(v['passed'] for v in checks):raise RuntimeError('Approach failed registered shape check')
    grasp=dict(candidate=dict(grasp_center_unity_world_m=(goal-d).tolist(),rotation_matrix=R.tolist()),number=n,state_file=f'artifacts/bulged-cuff/pose-{n:02d}/state.json',source_kind='bulged',generator='photo_reference_pregrasp_approach')
    closure=[registered.slacks(qgoal,float(g),d,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,61)]
    contract.update(minimum_start_tcp_distance_m=route['distance_m'],minimum_start_skin_cloth_clearance_m=.005)
    record=dict(number=n,geometry_record=grasp,central_contract=contract,torso_joints_rad=c.reference_torso,left_gap_m=initial['left_gap_m'],left_foot_floor_allowance_m=allowance,
        endpoint_task='pregrasp_reaching_only',require_virtual_closure=False,minimum_start_tcp_distance_m=route['distance_m'],minimum_start_skin_cloth_clearance_m=.005,
        initial_configuration_source=str(source.relative_to(ROOT)),attempts=[dict(q=qgoal.tolist(),left_q_rad=initial['left_q_rad'],translation_m=d.tolist(),gap_m=gap,trajectory=dict(route,passed=True))],
        virtual_closure_passed=all(v['passed'] for v in closure),virtual_closure_diagnostic=closure)
    write(a.output/'planned/target-photo-approach.json',record)


if __name__=='__main__':main()
