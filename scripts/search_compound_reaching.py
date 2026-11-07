#!/usr/bin/env python3
"""Jointly optimize scene translation and arm pose with signed clearance constraints."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,time,copy
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,Bounds,least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load,margin,good
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def target_data(row):
    source,sep,idx=row['source'].partition(':');raw=json.loads((ROOT/source).read_text())
    if sep:
        r=next(r for r in raw['candidates'] if r['candidate_index']==int(idx));c=r['candidate'];state=json.loads((ROOT/r['initial_state_source']).read_text());gaps=[r['opening_gap_m']]
    else:c=raw['candidate'];state=json.loads((ROOT/raw['source_state']).read_text());gaps=[r['gap_m'] for r in raw['valid_openings']]
    if 'particles_world' not in state:state['particles_world']=state['initial_particles_world']
    return c,state,gaps

def pose_constraint(k,p,R,x,with_jacobian=False):
    actual,rot,jp,ja=k.fast_pose(x[:7]);phi=Rotation.from_matrix(rot@R.T).as_rotvec();v=np.r_[actual-p-x[7:],phi]
    if not with_jacobian:return v
    theta=np.linalg.norm(phi);a,b,c=phi;skew=np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]]);factor=1/12 if theta<1e-5 else (1-.5*theta/np.tan(.5*theta))/theta**2
    inv=np.eye(3)-.5*skew+factor*skew@skew
    return v,np.block([[jp,-np.eye(3)],[inv@ja,np.zeros((3,3))]])

class Constraints:
    def __init__(self,k,checker,gap):self.k=k;self.checker=checker;self.gap=gap;self.last=None;self.calls=0
    def evaluate(self,x):
        if self.last is None or not np.array_equal(x,self.last):
            self.last=x.copy();self.calls+=1
            v,J=self.checker.slacks(x[:7],self.gap,x[7:],with_jacobian=True)
            start,K=self.checker.slacks(self.k.start,self.gap,x[7:],with_jacobian=True);K[:,:7]=0
            self.v=np.r_[v,start];self.J=np.vstack([J,K])
        return self.v,self.J
    def fun(self,x):return self.evaluate(x)[0]
    def jac(self,x):return self.evaluate(x)[1]

def trace(k,checker,c,gap,x,pre_seed=None):
    comfortable=copy.copy(k);comfortable.lower=k.lower+np.deg2rad(5);comfortable.upper=k.upper-np.deg2rad(5)
    p=np.array(c['grasp_center_unity_world_m'])+x[7:];R=np.array(c['rotation_matrix']);q=x[:7];rows=[q];failure=None
    for distance in np.linspace(.0025,.05,20):
        target=p-R[:,0]*distance;preferred=rows[-1] if pre_seed is None else (1-distance/.05)*q+(distance/.05)*np.array(pre_seed)
        r=comfortable.solve(target,R,preferred);a=np.array(r['q'])
        collision=checker.slacks(a,gap,x[7:],details=True)
        if not good(r) or margin(k,a)<5-1e-7 or not collision['passed']:
            constraints=Constraints(k,checker,gap)
            def eq(angles):return pose_constraint(k,target,R,np.r_[angles,np.zeros(3)])
            def eq_jac(angles):return pose_constraint(k,target,R,np.r_[angles,np.zeros(3)],True)[1][:,:7]
            fit=minimize(lambda angles:np.sum((angles-preferred)**2),a,jac=lambda angles:2*(angles-preferred),method='SLSQP',bounds=list(zip(comfortable.lower,comfortable.upper)),constraints=[dict(type='eq',fun=eq,jac=eq_jac),dict(type='ineq',fun=lambda angles:constraints.fun(np.r_[angles,x[7:]]),jac=lambda angles:constraints.jac(np.r_[angles,x[7:]])[:,:7])],options=dict(maxiter=80,ftol=1e-10))
            a=fit.x;res=eq(a);r=dict(q=a.tolist(),position_error_m=float(np.linalg.norm(res[:3])),orientation_error_deg=float(np.rad2deg(np.linalg.norm(res[3:]))),solver_message=str(fit.message));collision=checker.slacks(a,gap,x[7:],details=True)
        if not good(r) or margin(k,a)<5-1e-7:failure='preapproach_ik_or_margin';break
        if not collision['passed']:failure='approach_collision';break
        if np.max(abs(a-rows[-1]))>np.deg2rad(10):failure='branch_jump';break
        rows.append(a)
    if failure:return dict(passed=False,stage=failure,approach_samples=len(rows),last_ik_result=r)
    closure=[checker.slacks(q,float(g),x[7:],allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,21)]
    if not all(r['passed'] for r in closure):return dict(passed=False,stage='closure_collision',closure_checks=closure)
    pre=rows[-1];steps=max(2,int(np.ceil(np.max(abs(pre-k.start))/np.deg2rad(.5))));path=[]
    for t in np.linspace(0,1,steps+1):
        a=k.start*(1-t)+pre*t;check=checker.slacks(a,gap,x[7:],details=True)
        if not check['passed']:return dict(passed=False,stage='start_route_collision',approach_q=[a.tolist() for a in rows[::-1]],first_blocked_check=check)
        path.append(a.tolist())
    return dict(passed=True,stage='sampled_route_passed',approach_q=[a.tolist() for a in rows[::-1]],start_route_q=path,closure_checks=closure,continuous_collision_certified=False)

def main():
    global OUT,load,CompoundScene
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=2);ap.add_argument('--iterations',type=int,default=80);ap.add_argument('--indices',type=int,nargs='+');ap.add_argument('--shards',type=int,default=1);ap.add_argument('--shard-index',type=int,default=0);ap.add_argument('--translation-bounds',type=float,nargs=3,default=[.3,.15,.3]);ap.add_argument('--output',default='artifacts/compound-reaching');ap.add_argument('--max-joint1-deg',type=float);ap.add_argument('--robot-scene');ap.add_argument('--audited-shoulder',action='store_true');ap.add_argument('--geometry-manifest',default='artifacts/compound-reaching/decomposition.json');a=ap.parse_args();OUT=ROOT/a.output;OUT.mkdir(exist_ok=True)
    if a.robot_scene:
        from apron_reaching_common import overlay_robot,AuditedScene
        robot_scene=json.loads((ROOT/a.robot_scene).read_text());previous_load=load
        load=lambda n:overlay_robot(previous_load(n),robot_scene)
    if a.audited_shoulder:
        from apron_reaching_common import AuditedScene
        CompoundScene=AuditedScene
    (OUT/'search').mkdir(exist_ok=True);prior=json.loads((ROOT/'artifacts/translated-reaching/summary.json').read_text());manifest=ROOT/a.geometry_manifest;assert json.loads(manifest.read_text())['passed'];bounds=np.array(a.translation_bounds);pad=np.deg2rad(5);k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',load(39));results=[]
    planning_upper=k.upper.copy()
    if a.max_joint1_deg is not None:planning_upper[0]=min(planning_upper[0],np.deg2rad(a.max_joint1_deg)+pad)
    for index,row in enumerate(prior['results']):
        if (a.indices is not None and index not in a.indices) or index%a.shards!=a.shard_index:continue
        output=OUT/f'search/target-{index:03d}.json'
        if output.exists():results.append(json.loads(output.read_text()));continue
        c,state,gaps=target_data(row);scene=load(row['number']);scene['particles_world']=state['particles_world'];checker=CompoundScene(k,scene,state['triangles'],manifest,ROOT);lower_delta=-bounds.copy();lower_delta[1]=max(lower_delta[1],checker.floor_top-min(checker.floor_minima.values()));p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);old=json.loads((ROOT/f'artifacts/translated-reaching/target-{index:03d}.json').read_text());attempts=[]
        seeds=sorted(old['attempts'],key=lambda r:r['position_error_m']+.15*np.deg2rad(r['orientation_error_deg']))[:a.seeds]
        if a.robot_scene:
            seeds=[dict(q=k.start.tolist(),translation_m=np.clip(k.pose(k.start)[0]-p,lower_delta,bounds).tolist())]+seeds[:max(0,a.seeds-1)]
        for gap in gaps:
          constraint=Constraints(k,checker,gap)
          for seed in seeds:
            start=time.monotonic();x0=np.r_[seed['q'],seed['translation_m']];x0=np.clip(x0,np.r_[k.lower+pad,lower_delta],np.r_[planning_upper-pad,bounds]);initial=constraint.fun(x0).copy()
            def obj(x):return .002*np.sum((x[:7]-k.start)**2)+.02*np.sum(x[7:]**2)
            def grad(x):return np.r_[.004*(x[:7]-k.start),.04*x[7:]]
            def restore_fun(x):
                pose=pose_constraint(k,p,R,x);g=constraint.fun(x)
                return np.r_[5*pose[:3],.75*pose[3:],5*np.minimum(g,0)]
            def restore_jac(x):
                _,P=pose_constraint(k,p,R,x,True);g,G=constraint.evaluate(x)
                return np.vstack([5*P[:3],.75*P[3:],5*G*(g<0)[:,None]])
            restoration=least_squares(restore_fun,x0,jac=restore_jac,bounds=(np.r_[k.lower+pad,lower_delta],np.r_[planning_upper-pad,bounds]),max_nfev=60,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            x0=restoration.x
            fit=minimize(obj,x0,jac=grad,method='SLSQP',bounds=Bounds(np.r_[k.lower+pad,lower_delta],np.r_[planning_upper-pad,bounds]),constraints=[dict(type='eq',fun=lambda x:pose_constraint(k,p,R,x),jac=lambda x:pose_constraint(k,p,R,x,True)[1]),dict(type='ineq',fun=constraint.fun,jac=constraint.jac)],options=dict(maxiter=a.iterations,ftol=1e-10))
            x=fit.x;pose=pose_constraint(k,p,R,x);slacks=constraint.fun(x);valid=np.linalg.norm(pose[:3])<=1e-4 and np.rad2deg(np.linalg.norm(pose[3:]))<=.5 and slacks.min()>=-1e-9
            entry=dict(q=x[:7].tolist(),translation_m=x[7:].tolist(),gap_m=gap,initial_minimum_slack_m=float(initial.min()),minimum_slack_m=float(slacks.min()),position_error_m=float(np.linalg.norm(pose[:3])),orientation_error_deg=float(np.rad2deg(np.linalg.norm(pose[3:]))),joint_margin_deg=margin(k,x[:7]),restoration_cost=float(restoration.cost),solver_success=bool(fit.success),solver_message=str(fit.message),iterations=fit.nit,constraint_evaluations=constraint.calls,elapsed_s=time.monotonic()-start,endpoint_and_start_passed=bool(valid),endpoint=checker.slacks(x[:7],gap,x[7:],details=True),initial_pose=checker.slacks(k.start,gap,x[7:],details=True))
            if valid:entry['trajectory']=trace(k,checker,c,gap,x)
            attempts.append(entry);print('trial',index,len(attempts),'valid',valid,'slack',entry['minimum_slack_m'],'s',round(entry['elapsed_s'],1),flush=True)
          if any(r.get('trajectory',{}).get('passed') for r in attempts):break
        record=dict(max_joint1_deg=a.max_joint1_deg,robot_scene=a.robot_scene,audited_shoulder=a.audited_shoulder,start_q_rad=k.start.tolist(),geometry_manifest=a.geometry_manifest,floor_constraint_included=True,translation_lower_bound_m=lower_delta.tolist(),translation_bounds_m=bounds.tolist(),seeds=a.seeds,maximum_iterations=a.iterations,index=index,source=row['source'],number=row['number'],attempts=attempts);output.write_text(json.dumps(record,indent=2));results.append(record)
    (OUT/f'search-summary-{a.shard_index}.json').write_text(json.dumps(dict(targets=len(results),trials=sum(len(r['attempts']) for r in results),endpoint_passes=sum(t['endpoint_and_start_passed'] for r in results for t in r['attempts']),trajectory_passes=sum(t.get('trajectory',{}).get('passed',False) for r in results for t in r['attempts']),executed=False),indent=2))
if __name__=='__main__':main()
