#!/usr/bin/env python3
"""Fit the supplied front-facing, elbows-out sock-reaching photograph.

The photograph has no camera calibration or joint encoders. Its constraints
are explicit spatial/angle preferences, not a claim of exact image recovery.
"""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import state_for,write,FAST_MANIFEST,REGISTERED_MANIFEST
from search_fixed_torso_starts import context
from search_compound_reaching import pose_constraint
from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM
from sock_dressing_simulation.grasp_geometry import HANDEDNESS

PREFERRED=np.deg2rad([70,-10,0,110,90,-60,30])
PHOTO_LO=np.deg2rad([-60,-120,-110,0,-160,-80,0])
PHOTO_HI=np.deg2rad([230,10,110,125,160,80,80])


class LeftChain:
    def __init__(self,k):
        self.k=k;f=k.fk(k.start);self.parent=f[k.joints[LEFT_ARM[0]].find('parent').get('link')]
        last=k.joints[LEFT_ARM[-1]].find('child').get('link')
        self.tail=np.linalg.inv(f[last])@f['left_gripper/gripper_base']

    def pose(self,q):
        k=self.k;T=self.parent.copy();axes=[];points=[]
        for a,n in zip(q,LEFT_ARM):
            T=T@k.origins[n];points.append(T[:3,3].copy());axes.append(T[:3,:3]@k.axes[n]);T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(k.axes[n]*a).as_matrix()
        T=T@self.tail;raw=T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf
        return k.A@raw+k.b,k.A@T[:3,:3]@HANDEDNESS,k.A@np.cross(np.asarray(axes),raw-np.asarray(points)).T,-(np.asarray(axes)@k.A.T).T


def photo_context(number,state,allowance,manifest=FAST_MANIFEST):
    k,c,dl,du,contract,scene=context(number,state,manifest,left_foot_floor_allowance=allowance)
    ankle=np.array(state['source']['ankle']);center=-.144-ankle[0]
    dl=np.maximum(dl,[center-.035,-.25,-.6]);du=np.minimum(du,[center+.035,.05,0.])
    contract.update(reference_photo='artifacts/sock-pose-pattern/screen_shot 2026-11-05 121319.png',
        photo_ankle_lateral_tolerance_m=.035,translation_lower_m=dl.tolist(),translation_upper_m=du.tolist(),
        minimum_start_tcp_distance_m=.04,minimum_start_skin_cloth_clearance_m=.02)
    return k,c,dl,du,contract,scene


def screen(a):
    refs=json.loads((ROOT/'artifacts/centered-reaching/full-grid/geometry-summary.json').read_text())['accepted'];contexts={};counts=Counter()
    refs=sorted(enumerate(refs),key=lambda ir:(ir[1]['number']!=10,ir[1]['number'] not in (38,50),ir[1]['number'],ir[0]))
    for index,ref in refs:
        path=a.output/'screen'/f'target-{index:04d}.json'
        if path.exists():continue
        gr=json.loads((ROOT/ref['file']).read_text())['accepted'][ref['candidate_index']];state=state_for(gr);n=gr['number']
        if n not in contexts:contexts[n]=photo_context(n,state,a.floor_allowance)
        k,c,dl,du,contract,scene=contexts[n];c.park_left(PREFERRED,.04)
        lo=np.r_[np.maximum(k.lower+np.deg2rad(10),PHOTO_LO),dl];hi=np.r_[np.minimum(k.upper-np.deg2rad(10),PHOTO_HI),du]
        p=np.array(gr['candidate']['grasp_center_unity_world_m']);R=np.array(gr['candidate']['rotation_matrix']);rows=[];t0=time.monotonic()
        seeds=[PREFERRED,np.deg2rad([47,-30,0,100,0,45,25]),np.deg2rad([70,-40,-50,80,-50,-50,35]),np.deg2rad([20,-20,40,100,50,50,35])]
        for si,q in enumerate(seeds[:a.seeds]):
            delta=np.clip(k.pose(q)[0]-p,dl+1e-9,du-1e-9);x0=np.clip(np.r_[q,delta],lo+1e-9,hi-1e-9)
            fit=least_squares(lambda x:pose_constraint(k,p,R,x)*np.r_[np.full(3,8.),np.ones(3)],x0,
                jac=lambda x:pose_constraint(k,p,R,x,True)[1]*np.r_[np.full(3,8.),np.ones(3)][:,None],bounds=(lo,hi),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            x=fit.x;e=pose_constraint(k,p,R,x);entry=dict(seed=si,q=x[:7].tolist(),translation_m=x[7:].tolist(),position_error_m=float(np.linalg.norm(e[:3])),orientation_error_deg=float(np.rad2deg(np.linalg.norm(e[3:]))),stage='ik_failed')
            if entry['position_error_m']<=1e-4 and entry['orientation_error_deg']<=.5:
                gap=max(g['gap_m'] for g in gr['valid_openings']);ch=c.slacks(x[:7],gap,x[7:],details=True);issues=[i for i in ch['issues'] if not any(s.startswith(('left_arm/','left_gripper/')) for s in i['pair'])]
                entry.update(stage='right_endpoint_passed' if not issues else 'right_endpoint_collision',gap_m=gap,check=ch,right_issues=issues,right_minimum_slack_m=min([i['slack_m'] for i in issues]+[.02]))
            rows.append(entry);counts[entry['stage']]+=1
        write(path,dict(index=index,number=n,geometry_reference=ref,geometry_record=gr,central_contract=contract,left_foot_floor_allowance_m=a.floor_allowance,attempts=rows,elapsed_s=time.monotonic()-t0,photo_joint_lower_deg=np.rad2deg(lo[:7]).tolist(),photo_joint_upper_deg=np.rad2deg(hi[:7]).tolist()))
        print('PHOTO SCREEN',index,n,dict(Counter(r['stage'] for r in rows)),round(time.monotonic()-t0,1),flush=True)
    write(a.output/'screen-summary.json',dict(counts=dict(counts),left_foot_floor_allowance_m=a.floor_allowance))


def initials(a):
    state=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{a.number:02d}/state.json').read_text())
    k,c,dl,du,contract,scene=photo_context(a.number,state,a.floor_allowance)
    lc=LeftChain(k);lo=np.tile(np.maximum(k.lower+np.deg2rad(10),PHOTO_LO),2);hi=np.tile(np.minimum(k.upper-np.deg2rad(10),PHOTO_HI),2)
    preferred=np.tile(np.deg2rad([140,-50,-100,95,120,-60,45] if a.forearms_down else [145,-30,-90,95,110,-60,45]),2);candidates=[];attempts=[]
    if a.seed_configuration:
        source=json.loads(a.seed_configuration.read_text())['candidates'][0];preferred=np.r_[source['right_q_rad'],source['left_q_rad']]
    samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{a.number:02d}/samples.json').read_text())
    cuff=np.mean([s['position'] for s in samples if s['depth_m']==min(v['depth_m'] for v in samples)],axis=0)
    for dy in a.heights:
      for dz in a.depths:
        d=np.clip([-.144-state['source']['ankle'][0],dy,dz],dl+1e-6,du-1e-6)
        target=cuff+d;right=target+[-.075,.01,-.035];left=target+[.075,.01,-.035]
        def equality(x):return np.r_[k.pose(x[:7])[0]-right,lc.pose(x[7:])[0]-left]
        def eqjac(x):
            J=np.zeros((6,14));J[:3,:7]=k.fast_pose(x[:7])[2];J[3:,7:]=lc.pose(x[7:])[2];return J
        def elbow_constraint(x):
            c.park_left(x[7:],.04);f=k.fk(x[:7],only=[side+'_arm/link_'+str(i) for side in ('right','left') for i in (3,5,7)])
            r=k.A@f['right_arm/link_3'][:3,3]+k.b;l=k.A@f['left_arm/link_3'][:3,3]+k.b
            values=[r[1]-right[1]-.05,l[1]-left[1]-.05,.91-r[1],.91-l[1],-.41-r[0],l[0]-.12]
            if a.face_cuff:
                rd=(target-right)/np.linalg.norm(target-right);ld=(target-left)/np.linalg.norm(target-left)
                values.extend([k.pose(x[:7])[1][:,0]@rd-np.cos(np.deg2rad(40)),lc.pose(x[7:])[1][:,0]@ld-np.cos(np.deg2rad(40))])
            if a.forearms_down:
                for side,elbow in [('right',r),('left',l)]:
                    for i in (5,7):
                        wrist=k.A@f[side+'_arm/link_'+str(i)][:3,3]+k.b
                        values.append(elbow[1]-wrist[1]-.02)
            return np.array(values)
        fit=least_squares(lambda x:np.r_[10*equality(x),3*np.minimum(elbow_constraint(x),0),.005*(x-preferred)],preferred,bounds=(lo,hi),max_nfev=140)
        print('PHOTO INITIAL SEED',dy,dz,'position',np.linalg.norm(equality(fit.x)),'elbow',elbow_constraint(fit.x).min(),flush=True)
        cache={};calls=0
        def cons(x):
            nonlocal calls
            if 'x' not in cache or not np.array_equal(cache['x'],x):
                calls+=1;c.park_left(x[7:],.04);g,J=c.slacks(x[:7],.04,d,with_jacobian=True);G=np.zeros((len(g),14));G[:,:7]=J[:,:7];h=1e-5
                for j in range(7):
                    plus=x[7:].copy();minus=plus.copy();plus[j]+=h;minus[j]-=h
                    c.park_left(plus,.04);gp=c.slacks(x[:7],.04,d);c.park_left(minus,.04);gm=c.slacks(x[:7],.04,d);G[:,7+j]=(gp-gm)/(2*h)
                c.park_left(x[7:],.04);cache.update(x=x.copy(),g=g,J=G)
                if calls%20==0:print('PHOTO INITIAL OPT',dy,dz,calls,'slack',g.min(),flush=True)
            return cache['g'],cache['J']
        class Feasible(Exception):pass
        def stop_when_feasible(x):
            if np.linalg.norm(equality(x))<1e-5 and elbow_constraint(x).min()>=-1e-7:
                c.park_left(x[7:],.04)
                if c.slacks(x[:7],.04,d).min()>=.0005-1e-8:raise Feasible(x.copy())
        try:
            opt=minimize(lambda x:np.sum((x-preferred)**2),fit.x,jac=lambda x:2*(x-preferred),method='SLSQP',bounds=list(zip(lo,hi)),callback=stop_when_feasible,
                constraints=[dict(type='eq',fun=equality,jac=eqjac),dict(type='ineq',fun=elbow_constraint),dict(type='ineq',fun=lambda x:cons(x)[0]-.0005,jac=lambda x:cons(x)[1])],options=dict(maxiter=80,ftol=1e-10))
        except Feasible as solution:opt=SimpleNamespace(x=solution.args[0],message='Stopped at a feasible photo-constrained posture')
        x=opt.x;c.park_left(x[7:],.04);ch=c.slacks(x[:7],.04,d,details=True);err=float(np.linalg.norm(equality(x)));passed=ch['passed'] and err<1e-4 and elbow_constraint(x).min()>=-1e-7
        r=dict(number=a.number,right_q_rad=x[:7].tolist(),left_q_rad=x[7:].tolist(),right_q_deg=np.rad2deg(x[:7]).tolist(),left_q_deg=np.rad2deg(x[7:]).tolist(),translation_m=d.tolist(),
            right_gap_m=.04,left_gap_m=.04,left_foot_floor_allowance_m=a.floor_allowance,central_contract=contract,passed=bool(passed),check=ch,position_error_m=err,
            right_tcp_world_m=k.pose(x[:7])[0].tolist(),left_tcp_world_m=lc.pose(x[7:])[0].tolist(),cuff_center_world_m=target.tolist(),
            torso_joints_rad=c.reference_torso,minimum_joint_margin_deg=float(np.rad2deg(np.minimum(x-np.tile(k.lower,2),np.tile(k.upper,2)-x)).min()),
            grounded_left_foot_penetration_m=max(0.,c.floor_top-c.original_human_minimum_y-d[1]),elbow_constraints_m=elbow_constraint(x)[:6].tolist(),facing_cuff_required=a.face_cuff,forearms_down_required=a.forearms_down,photo_constraint_slacks=elbow_constraint(x).tolist(),solver_message=str(opt.message),constraint_evaluations=calls)
        attempts.append(r);write(a.output/'initial-search.json',dict(attempts=attempts))
        print('PHOTO INITIAL',len(attempts),passed,'error',err,'slack',ch['minimum_slack_m'],'d',d,'qR',np.rad2deg(x[:7]).round(1),'qL',np.rad2deg(x[7:]).round(1),flush=True)
        if passed:
            registered_k,registered,*_=photo_context(a.number,state,a.floor_allowance,REGISTERED_MANIFEST);registered.park_left(x[7:],.04)
            r['registered_check']=registered.slacks(x[:7],.04,d,details=True)
            if not r['registered_check']['passed']:continue
            candidates.append(r);write(a.output/'initial-configurations.json',dict(candidates=candidates,photo_reference=contract['reference_photo'],reachability_verified=False));return


def continuous(a):
    from search_centered_reaching import geometric_check
    from sock_dressing_simulation.grasp_geometry import SkinBody
    from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
    from sock_dressing_simulation.geometric_closure import material_distances
    import yaml
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());contexts={};counts=Counter()
    paths=sorted((a.output/'comfortable/screen').glob('*.json'))
    paths.sort(key=lambda path:min(v['position_error_m']+.15*np.deg2rad(v['orientation_error_deg']) for v in json.loads(path.read_text())['attempts']))
    for path in paths:
        row=json.loads(path.read_text());n=row['number'];gr=row['geometry_record'];state=state_for(gr);out=a.output/'continuous'/path.name
        if out.exists():continue
        if n not in contexts:
            k,c,dl,du,contract,scene=photo_context(n,state,a.floor_allowance);c.park_left(PREFERRED,.04)
            samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/samples.json').read_text());contexts[n]=(k,c,dl,du,contract,samples,SkinBody(state['source']['skin_surface']),ClothScreen(state['particles_world'],state['triangles']))
        k,c,dl,du,contract,samples,skin,cloth=contexts[n];sample=samples[gr['candidate']['sample_index']];p=np.array(sample['position']);R=np.array(gr['candidate']['rotation_matrix'])
        faces=np.array(state['triangles']).reshape(-1,3);dist=material_distances(np.array(state['source']['rest_positions']),faces,faces[sample['face']],sample['weights'])
        lo=np.r_[np.maximum(k.lower+np.deg2rad(10),PHOTO_LO),dl];hi=np.r_[np.minimum(k.upper-np.deg2rad(10),PHOTO_HI),du];attempts=[]
        for si,src in enumerate(sorted(row['attempts'],key=lambda v:v['position_error_m']+.15*np.deg2rad(v['orientation_error_deg']))[:a.seeds]):
            x0=np.clip(np.r_[src['q'],src['translation_m']],lo+1e-10,hi-1e-10)
            def objective(x):
                e,J=pose_constraint(k,p,R,x,True);return np.sum(e[3:]**2)+1e-5*np.sum((x[:7]-PREFERRED)**2),2*J[3:].T@e[3:]+np.r_[2e-5*(x[:7]-PREFERRED),np.zeros(3)]
            fit=minimize(objective,x0,jac=True,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=lambda x:pose_constraint(k,p,R,x)[:3],jac=lambda x:pose_constraint(k,p,R,x,True)[1][:3])],options=dict(maxiter=120,ftol=1e-11))
            x=fit.x;actual,rotation=k.pose(x[:7]);error=float(np.linalg.norm(actual-p-x[7:]));a0=dict(q=x[:7].tolist(),translation_m=x[7:].tolist(),position_error_m=error,rotation_matrix=rotation.tolist(),stage='position_failed');attempts.append(a0)
            if error<1e-4:
                geo,reason=geometric_check(k,state,sample,rotation,skin,cloth,settings,dist);a0['stage']=reason
                if geo:
                    new=dict(gr,**geo,generator='photo_reference_continuous_10deg_margin');gap=max(v['gap_m'] for v in geo['valid_openings']);a0.update(gap_m=gap,endpoint=c.slacks(x[:7],gap,x[7:],details=True),left_q_rad=PREFERRED.tolist());write(a.output/'endpoints'/path.name,dict(row,geometry_record=new,attempts=[a0],left_gap_m=.04));print('PHOTO CONTINUOUS ACCEPT',n,row['index'],flush=True)
            counts[a0['stage']]+=1
        write(out,dict(number=n,index=row['index'],attempts=attempts));print('PHOTO CONTINUOUS',n,row['index'],[v['stage'] for v in attempts],flush=True)
    write(a.output/'continuous-summary.json',dict(counts=dict(counts)))


def photo_geometry(a):
    from search_centered_reaching import geometric_check
    from sock_dressing_simulation.grasp_geometry import SkinBody
    from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
    from sock_dressing_simulation.geometric_closure import material_distances
    import yaml
    initial=json.loads((a.output/'initial-configurations.json').read_text())['candidates'][0]
    state=json.loads((ROOT/f"artifacts/bulged-cuff/pose-{initial['number']:02d}/state.json").read_text());k,*_=photo_context(initial['number'],state,a.floor_allowance);R0=k.pose(initial['right_q_rad'])[1]
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());refs=json.loads((ROOT/'artifacts/centered-reaching/full-grid/geometry-summary.json').read_text())['accepted'];numbers=sorted({r['number'] for r in refs},key=lambda n:(n!=a.number,n))
    rotations=[R0]
    # Include rotations with the hand aimed inward from the robot's right.
    for magnitude in (15,30,45,60):
        for axis in np.r_[np.eye(3),-np.eye(3)]:rotations.append(Rotation.from_rotvec(axis*np.deg2rad(magnitude)).as_matrix()@R0)
    for n in numbers:
        output=a.output/'photo-geometry'/f'pose-{n:02d}.json'
        if output.exists():continue
        state_file=f'artifacts/bulged-cuff/pose-{n:02d}/state.json';state=json.loads((ROOT/state_file).read_text());samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/samples.json').read_text());skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);faces=np.array(state['triangles']).reshape(-1,3);accepted=[];counts=Counter()
        for sample in samples:
            dist=None
            for ri,R in enumerate(rotations):
                if dist is None:dist=material_distances(np.array(state['source']['rest_positions']),faces,faces[sample['face']],sample['weights'])
                geo,reason=geometric_check(k,state,sample,R,skin,cloth,settings,dist,approach_distance=a.geometry_approach_distance);counts['tested']+=1;counts[reason]+=1
                if geo:accepted.append(dict(geo,number=n,state_file=state_file,source=state_file,source_kind='bulged',generator='photo_initial_orientation_neighborhood',rotation_index=ri,approach_distance_m=a.geometry_approach_distance,actual_robot_route_required=True))
        write(output,dict(number=n,counts=dict(counts),accepted=accepted));print('PHOTO GEOMETRY',n,dict(counts),'accepted',len(accepted),flush=True)


def offsets(a):
    from search_centered_reaching import geometric_check
    from sock_dressing_simulation.grasp_geometry import SkinBody
    from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
    from sock_dressing_simulation.geometric_closure import material_distances
    import yaml
    initial=json.loads((a.output/'initial-configurations.json').read_text())['candidates'][0];n=a.number
    state_file=f'artifacts/bulged-cuff/pose-{n:02d}/state.json';state=json.loads((ROOT/state_file).read_text());k,*_=photo_context(n,state,a.floor_allowance);R0=k.pose(initial['right_q_rad'])[1]
    samples=json.loads((ROOT/f'artifacts/bulged-cuff/pose-{n:02d}/samples.json').read_text());skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);faces=np.array(state['triangles']).reshape(-1,3);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());accepted=[];counts=Counter();rotations=[R0]
    for mag in (15,30,45):
        for axis in np.r_[np.eye(3),-np.eye(3)]:rotations.append(Rotation.from_rotvec(axis*np.deg2rad(mag)).as_matrix()@R0)
    for sample in samples:
        if sample['depth_m']>.01000001:continue
        dist=material_distances(np.array(state['source']['rest_positions']),faces,faces[sample['face']],sample['weights'])
        for radial in (0.,.003,.006,.009):
          for axial in (0.,.003,.006):
            offset=radial*np.array(sample['normal'])+axial*np.array(state['opening_normal']);shifted=dict(sample,position=(np.array(sample['position'])+offset).tolist())
            for ri,R in enumerate(rotations):
                geo,reason=geometric_check(k,state,shifted,R,skin,cloth,settings,dist);counts['tested']+=1;counts[reason]+=1
                if geo:
                    accepted.append(dict(geo,number=n,state_file=state_file,source=state_file,source_kind='bulged',generator='photo_grasp_center_offset',target_offset_m=offset.tolist(),rotation_index=ri));print('OFFSET ACCEPT',n,sample['index'],radial,axial,ri,flush=True)
        if sample['theta_index']%8==0:
            write(a.output/'offset-geometry'/f'pose-{n:02d}.json',dict(number=n,counts=dict(counts),accepted=accepted));print('OFFSET',n,sample['index'],dict(counts),flush=True)
        if len(accepted)>=8:break
    write(a.output/'offset-geometry'/f'pose-{n:02d}.json',dict(number=n,counts=dict(counts),accepted=accepted))



def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['screen','initials','continuous','geometry','offsets']);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/photo-reference-reaching');ap.add_argument('--floor-allowance',type=float,default=.15);ap.add_argument('--seeds',type=int,default=4);ap.add_argument('--number',type=int,default=10);ap.add_argument('--face-cuff',action='store_true');ap.add_argument('--forearms-down',action='store_true');ap.add_argument('--geometry-approach-distance',type=float,default=.05);ap.add_argument('--seed-configuration',type=Path);ap.add_argument('--heights',type=float,nargs='+',default=[-.06,-.10,-.14]);ap.add_argument('--depths',type=float,nargs='+',default=[-.3,-.4,-.5]);a=ap.parse_args();a.output=a.output.resolve();{'screen':screen,'initials':initials,'continuous':continuous,'geometry':photo_geometry,'offsets':offsets}[a.stage](a)


if __name__=='__main__':main()
