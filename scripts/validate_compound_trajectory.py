#!/usr/bin/env python3
"""Build and independently recheck a smooth, slow joint trajectory for a survivor."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_compound_reaching import target_data
from search_translated_reaching import load,margin
from sock_dressing_simulation.arm_reaching import Kinematics,ARM
from sock_dressing_simulation.compound_reaching import CompoundScene
OUT=ROOT/'artifacts/compound-reaching'

def smooth_profile(q,velocity,acceleration):
    q=np.asarray(q);q=q[np.r_[True,np.linalg.norm(np.diff(q,axis=0),axis=1)>1e-10]]
    s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(q,axis=0),axis=1))];s/=s[-1];curve=PchipInterpolator(s,q,axis=0);c=curve.c;du=np.diff(s)[:,None]
    speeds=np.maximum(abs(c[2]),abs(3*c[0]*du**2+2*c[1]*du+c[2]));root=np.divide(-c[1],3*c[0],out=np.zeros_like(c[0]),where=abs(c[0])>1e-14);inside=(root>0)&(root<du);atroot=abs(3*c[0]*root**2+2*c[1]*root+c[2]);speeds=np.where(inside,np.maximum(speeds,atroot),speeds);vmax=speeds.max(0);amax=np.maximum(abs(2*c[1]),abs(6*c[0]*du+2*c[1])).max(0)
    duration=1.05*max(float(np.max(1.875*vmax/(velocity*.1))),float(np.sqrt(np.max((1.875**2*amax+5.773503*vmax)/(acceleration*.1)))),1.)
    grid=[]
    for i,width in enumerate(np.diff(s)):
        steps=max(2,int(np.ceil(speeds[i].max()*width/np.deg2rad(.25))));grid.extend(np.linspace(s[i],s[i+1],steps+1)[:-1])
    grid=np.r_[grid,1.];tau=np.array([0 if u==0 else 1 if u==1 else brentq(lambda t:10*t**3-15*t**4+6*t**5-u,0,1) for u in grid]);gp=30*tau**2-60*tau**3+30*tau**4;gpp=60*tau-180*tau**2+120*tau**3
    return dict(q=curve(grid),t=tau*duration,velocity=curve(grid,1)*gp[:,None]/duration,acceleration=(curve(grid,2)*gp[:,None]**2+curve(grid,1)*gpp[:,None])/duration**2,duration=duration,u=grid,curve=curve,velocity_bound=vmax*1.875/duration,acceleration_bound=(amax*1.875**2+vmax*5.773503)/duration**2)

def main():
    candidates=[json.loads(path.read_text()) for folder in ['rrt','optimized-route'] for path in (OUT/folder).glob('*.json')]
    candidates=[r for r in candidates if r.get('found') and r.get('dense_recheck_passed') and r.get('floor_constraint_included')]
    if not candidates:
        (OUT/'trajectory-status.json').write_text(json.dumps(dict(generated=False,reason='No floor-aware route passed collision checks',executed=False),indent=2));print('No floor-aware route; no trajectory exported');return
    r=candidates[0];a=dict(translation_m=r['translation_m'],gap_m=r['gap_m'],trajectory=dict(start_route_q=r['start_route_q'],approach_q=r['approach_q']));c,state,_=target_data(r);scene=load(r['number']);scene['particles_world']=state['particles_world'];k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',scene);checker=CompoundScene(k,scene,state['triangles'],OUT/'decomposition.json',ROOT);delta=np.array(a['translation_m']);gap=a['gap_m'];path=a['trajectory'];route=path['start_route_q']+path['approach_q'][1:];smooth=smooth_profile(route,k.velocity,k.acceleration);worst=None;minimum=float('inf');failures=[];joint_margin=180.;points=[];orientation_errors=[]
    for i,q in enumerate(smooth['q']):
        result=checker.slacks(q,gap,delta,details=True);joint_margin=min(joint_margin,margin(k,q));actual,rotation=k.pose(q);points.append(actual.tolist())
        if result['minimum_slack_m']<minimum:minimum=result['minimum_slack_m'];worst=dict(index=i,check=result)
        if not result['passed'] or margin(k,q)<5-1e-7:failures.append(dict(index=i,check=result,joint_margin_deg=margin(k,q)))
        if i%100==0:print('trajectory',i,len(smooth['q']),'minimum slack',minimum,flush=True)
    initial_gap=k.q0['right_gripper/finger_joint']+k.hand.gap_offset_m
    start_opening=[checker.slacks(k.start,float(g),delta,details=True) for g in np.linspace(initial_gap,gap,21)]
    closure=[checker.slacks(smooth['q'][-1],float(g),delta,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,61)]
    endpoint,rot=k.pose(smooth['q'][-1]);target=np.array(c['grasp_center_unity_world_m'])+delta;angle=float(np.rad2deg(Rotation.from_matrix(rot@np.array(c['rotation_matrix']).T).magnitude()))
    native=json.loads((OUT/'native-validation.json').read_text());passed=not failures and all(v['passed'] for v in start_opening+closure) and np.linalg.norm(endpoint-target)<=1e-4 and angle<=.5
    plan=dict(source=r['source'],number=r['number'],translation_m=delta.tolist(),moving_objects=['full human','sock','foot support'],fixed_objects=['robot base','torso','left arm','floor'],joint_names=ARM,opening_gap_m=gap,initial_gap_m=initial_gap,virtual_closing_gap_m=.001,gripper_speed_m_s=.005,times_s=smooth['t'].tolist(),q_rad=smooth['q'].tolist(),velocity_rad_s=smooth['velocity'].tolist(),acceleration_rad_s2=smooth['acceleration'].tolist(),tcp_world_m=points,interpolation='PCHIP joint curve with global quintic time map; initial and terminal velocity zero',duration_s=smooth['duration'],velocity_fraction_limit=.1,acceleration_fraction_limit=.1,analytic_velocity_bound_rad_s=smooth['velocity_bound'].tolist(),analytic_acceleration_bound_rad_s2=smooth['acceleration_bound'].tolist(),sampled_geometry_passed=passed,native_geometry_passed=native['passed'],executed=False,continuous_collision_certified=False)
    (OUT/'selected-trajectory.json').write_text(json.dumps(plan,indent=2));result=dict(passed=passed and native['passed'],offline_passed=passed,native_geometry_passed=native['passed'],sample_count=len(smooth['q']),minimum_margin_slack_m=minimum,minimum_joint_margin_deg=joint_margin,endpoint_position_error_m=float(np.linalg.norm(endpoint-target)),endpoint_orientation_error_deg=angle,maximum_velocity_fraction=float(np.max(smooth['velocity_bound']/k.velocity)),maximum_acceleration_fraction=float(np.max(smooth['acceleration_bound']/k.acceleration)),failures=failures,worst_sample=worst,start_opening_checks=start_opening,virtual_closure_checks=closure,executed=False,continuous_collision_certified=False)
    (OUT/'trajectory-validation.json').write_text(json.dumps(result,indent=2));print('offline pass',passed,'native pass',native['passed'],'samples',len(smooth['q']),'duration',smooth['duration'])
if __name__=='__main__':main()
