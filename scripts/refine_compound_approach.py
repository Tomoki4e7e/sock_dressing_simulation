#!/usr/bin/env python3
"""Joint endpoint/preapproach/translation refinement of floor-aware survivors."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,copy,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.optimize import minimize,least_squares
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data,pose_constraint,trace
from search_translated_reaching import load,margin
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def task(arg):
    path,trial=arg;row=json.loads(Path(path).read_text());a=row['attempts'][trial];c,state,_=target_data(row);scene=load(row['number']);scene['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',scene);checker=CompoundScene(k,scene,state['triangles'],OUT/'decomposition.json',ROOT);p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);pre=p-R[:,0]*.05;gap=a['gap_m'];pad=np.deg2rad(5);bound=np.array([.8,.3,.8]);lo=-bound;lo[1]=max(lo[1],checker.floor_top-min(checker.floor_minima.values()));comfortable=copy.copy(k);comfortable.lower=k.lower+pad;comfortable.upper=k.upper-pad
    q=np.array(a['q']);d=np.array(a['translation_m']);qpre=np.array(comfortable.solve(pre+d,R,q)['q']);x0=np.r_[q,qpre,d];lower=np.r_[comfortable.lower,comfortable.lower,lo];upper=np.r_[comfortable.upper,comfortable.upper,bound];x0=np.clip(x0,lower+1e-10,upper-1e-10);cache={};calls=0
    def poses(x,jac=False):
        e,E=pose_constraint(k,p,R,np.r_[x[:7],x[14:]],True);v,V=pose_constraint(k,pre,R,np.r_[x[7:14],x[14:]],True);J=np.zeros((12,17));J[:6,:7]=E[:,:7];J[:6,14:]=E[:,7:];J[6:,7:14]=V[:,:7];J[6:,14:]=V[:,7:]
        return (np.r_[e,v],J) if jac else np.r_[e,v]
    def constraints(x):
        nonlocal calls
        if 'x' not in cache or not np.array_equal(cache['x'],x):
            calls+=1;parts=[];Js=[]
            for q,ids in [(x[:7],range(7)),(x[7:14],range(7,14)),(k.start,[])]:
                v,J=checker.slacks(q,gap,x[14:],with_jacobian=True);G=np.zeros((len(v),17))
                if len(ids):G[:,ids]=J[:,:7]
                G[:,14:]=J[:,7:];parts.append(v);Js.append(G)
            cache.update(x=x.copy(),v=np.concatenate(parts),J=np.vstack(Js))
        return cache['v'],cache['J']
    weights=np.tile(np.r_[np.full(3,5.),np.full(3,.75)],2)
    def restore(x):return np.r_[poses(x)*weights,5*np.minimum(constraints(x)[0],0)]
    def restore_jac(x):
        _,P=poses(x,True);v,J=constraints(x);return np.vstack([P*weights[:,None],5*J*(v<0)[:,None]])
    start=time.monotonic();rest=least_squares(restore,x0,jac=restore_jac,bounds=(lower,upper),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    def objective(x):return .001*(np.sum((x[:7]-k.start)**2)+np.sum((x[7:14]-x[:7])**2))+.01*np.sum(x[14:]**2)
    fit=minimize(objective,rest.x,method='SLSQP',bounds=list(zip(lower,upper)),constraints=[dict(type='eq',fun=poses,jac=lambda x:poses(x,True)[1]),dict(type='ineq',fun=lambda x:constraints(x)[0],jac=lambda x:constraints(x)[1])],options=dict(maxiter=120,ftol=1e-10));x=fit.x;v=poses(x);g=constraints(x)[0];valid=np.linalg.norm(v[:3])<=1e-4 and np.linalg.norm(v[6:9])<=1e-4 and max(np.linalg.norm(v[3:6]),np.linalg.norm(v[9:]))<=np.deg2rad(.5) and g.min()>=-1e-9
    result=dict(source=row['source'],number=row['number'],input_file=str(Path(path).relative_to(ROOT)),input_trial=trial,q=x[:7].tolist(),preapproach_q=x[7:14].tolist(),translation_m=x[14:].tolist(),translation_bounds_m=bound.tolist(),gap_m=gap,minimum_slack_m=float(g.min()),pose_residual=v.tolist(),endpoint_and_preapproach_passed=bool(valid),solver_message=str(fit.message),elapsed_s=time.monotonic()-start,constraint_evaluations=calls,floor_constraint_included=True)
    if valid:result['trajectory']=trace(k,checker,c,gap,np.r_[x[:7],x[14:]])
    destination=OUT/'coupled-approach';destination.mkdir(exist_ok=True);name=Path(path).stem+f'-trial-{trial}.json';(destination/name).write_text(json.dumps(result,indent=2));print(name,valid,result.get('trajectory',{}).get('stage'),result['minimum_slack_m'],round(result['elapsed_s'],1),flush=True);return name

def main():
    tasks=[]
    for path in sorted((OUT/'floor-aware/search').glob('*.json')):
        row=json.loads(path.read_text())
        for i,a in enumerate(row['attempts']):
            if a['endpoint_and_start_passed']:tasks.append((str(path),i))
    with ProcessPoolExecutor(3) as pool:list(pool.map(task,tasks))
if __name__=='__main__':main()
