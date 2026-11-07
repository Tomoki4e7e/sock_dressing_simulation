#!/usr/bin/env python3
"""Validate an either-arm photo-start approach, opening and complete closure."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import state_for,write,REGISTERED_MANIFEST,geometric_check
from search_fixed_torso_starts import context
from validate_compound_trajectory import smooth_profile
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene,shared_gripper_geometry_audit
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances
import yaml
from sock_dressing_simulation.reaching_dataset import load_grasp_samples, verify_dataset_inputs


def validate(source,out,time_scale):
    row=json.loads(source.read_text());side=row['moving_side'];attempt=row['attempts'][0];state=state_for(row['geometry_record']);d=np.array(attempt['translation_m']);gap=attempt['gap_m'];initial_gap=row['initial_opening_gap_m']
    k,base,dl,du,contract,scene=context(row['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=row['left_foot_floor_allowance_m']);base.park_left(row['left_q_rad'],row['left_gap_m'])
    registration=REGISTERED_MANIFEST.parent;native=json.loads((registration/'native-validation/summary.json').read_text())
    if not native['passed'] or native['payload_sha256']!=hashlib.sha256((registration/'payload.json').read_bytes()).hexdigest():raise ValueError('Registered native geometry gate is missing or stale')
    equivalence=shared_gripper_geometry_audit(k)
    if side=='left' and not equivalence['passed']:raise ValueError('Right-hand capture geometry cannot be reused for this left hand')
    model=ControlledArmKinematics(k,side,row['right_q_rad'],row['right_gap_m']);checker=ControlledCollisionScene(base,side,row['right_q_rad'],row['right_gap_m']);raw=np.array(attempt['trajectory']['raw_q']);profile=smooth_profile(raw,model.velocity,model.acceleration)
    for key in ('t','duration'):profile[key]*=time_scale
    for key in ('velocity','velocity_bound'):profile[key]/=time_scale
    for key in ('acceleration','acceleration_bound'):profile[key]/=time_scale**2
    grid=np.unique(np.r_[profile['u'],(profile['u'][:-1]+profile['u'][1:])/2]);qs=profile['curve'](grid);minimum=float('inf');failures=[]
    fixed=['base_link','torso/link_1','torso/link_2','torso/link_3','head/link_1','head/link_2','head/link_3']+[(('right' if side=='left' else 'left')+'_arm/link_'+str(i)) for i in (1,3,5,7)]
    reference=model.fk(raw[0],initial_gap,only=fixed);position_drift=0.;rotation_drift=0.
    for i,q in enumerate(qs):
        ch=checker.slacks(q,gap,d,details=True);minimum=min(minimum,ch['minimum_slack_m'])
        if not ch['passed']:failures.append(dict(index=i,check=ch))
        frames=model.fk(q,gap,only=fixed)
        for name in fixed:
            position_drift=max(position_drift,float(np.linalg.norm(frames[name][:3,3]-reference[name][:3,3])));rotation_drift=max(rotation_drift,float(Rotation.from_matrix(frames[name][:3,:3]@reference[name][:3,:3].T).magnitude()))
        if i%100==0:print('CONTROLLED VALIDATION',side,i,len(qs),'slack',minimum,'failures',len(failures),flush=True)
    opening=[checker.slacks(raw[0],float(g),d,details=True) for g in np.linspace(initial_gap,gap,41)]
    closure=[checker.slacks(qs[-1],float(g),d,allow_cloth_overlap=True,details=True) for g in np.linspace(gap,.001,101)]
    c=row['geometry_record']['candidate'];p=np.array(c['grasp_center_unity_world_m'])+d;actual,R=model.pose(qs[-1]);pe=float(np.linalg.norm(actual-p));ae=float(np.rad2deg(Rotation.from_matrix(R@np.array(c['rotation_matrix']).T).magnitude()))
    # Recompute capture for the actual IK endpoint instead of accepting the
    # desired material sample/rotation merely because pose tolerance passed.
    samples=load_grasp_samples(row['geometry_record'],ROOT);sample=dict(samples[c['sample_index']],position=(actual-d).tolist());faces=np.array(state['triangles']).reshape(-1,3);dist=material_distances(np.array(state['source']['rest_positions']),faces,faces[sample['face']],sample['weights'])
    geo,reason=geometric_check(model,state,sample,R,SkinBody(state['source']['skin_surface']),ClothScreen(state['particles_world'],state['triangles']),yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text()),dist,approach_distance=0.)
    margin=float(np.rad2deg(np.minimum(qs-model.lower,model.upper-qs)).min());vf=float(np.max(profile['velocity_bound']/model.velocity));af=float(np.max(profile['acceleration_bound']/model.acceleration))
    passed=bool(not failures and all(g['passed'] for g in opening+closure) and geo and pe<=1e-4 and ae<=.5 and margin>=5-1e-6 and position_drift<1e-12 and rotation_drift<1e-12 and vf<=.1+1e-9 and af<=.1+1e-9 and np.all(d>=dl-1e-9) and np.all(d<=du+1e-9))
    record=dict(row,source_record=str(source.relative_to(ROOT)),translation_m=d.tolist(),joint_names=model.arm_names,start_q_rad=raw[0].tolist(),raw_q_rad=raw.tolist(),times_s=profile['t'].tolist(),q_rad=profile['q'].tolist(),velocity_rad_s=profile['velocity'].tolist(),acceleration_rad_s2=profile['acceleration'].tolist(),
        tcp_world_m=[model.pose(q)[0].tolist() for q in profile['q']],duration_s=profile['duration'],time_scale=time_scale,opening_gap_m=gap,close_after_reach=True,closing_gap_m=.001,
        preopen_duration_s=2.,closure_duration_s=10.,torso_joints_rad=base.reference_torso,fixed_joint_values_rad=base.reference_fixed_joints,geometry_manifest_sha256=hashlib.sha256(REGISTERED_MANIFEST.read_bytes()).hexdigest(),offline_passed=passed,executed=False,virtual_closure_passed=all(g['passed'] for g in closure),continuous_collision_certified=False)
    verify_dataset_inputs(record,ROOT)
    trajectory=out/'routes'/f'{source.stem}-trajectory.json';write(trajectory,record)
    result=dict(offline_passed=passed,moving_side=side,sample_count=len(qs),minimum_slack_m=minimum,minimum_joint_margin_deg=margin,maximum_velocity_fraction=vf,maximum_acceleration_fraction=af,failures=failures,
        start_opening_checks=opening,virtual_closure_checks=closure,virtual_closure_passed=record['virtual_closure_passed'],endpoint_position_error_m=pe,endpoint_orientation_error_deg=ae,actual_grasp_geometry_passed=bool(geo),actual_grasp_geometry_reason=reason,
        actual_grasp_geometry=geo,shared_gripper_geometry_audit=equivalence,maximum_fixed_frame_position_error_m=position_drift,maximum_fixed_frame_orientation_error_rad=rotation_drift,trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),geometry_manifest_sha256=record['geometry_manifest_sha256'],executed=False)
    write(trajectory.with_name(trajectory.name.replace('trajectory','validation')),result)
    with trajectory.with_suffix('.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['time_s']+[n+'_deg' for n in model.arm_names]);w.writerows(np.c_[profile['t'],np.rad2deg(profile['q'])])
    write(out/'trajectory-status.json',dict(generated=passed,moving_side=side,trajectory=str(trajectory.relative_to(ROOT)),executed=False));print('CONTROLLED VALIDATED',passed,'seconds',record['duration_s'],'closure',record['virtual_closure_passed'],'geometry',reason,flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/'artifacts/photo-grasp-adjustment/planned/target-left-0004.json');ap.add_argument('--output',type=Path,default=ROOT/'artifacts/photo-grasp-adjustment');ap.add_argument('--time-scale',type=float,default=1.);a=ap.parse_args()
    if not np.isfinite(a.time_scale) or a.time_scale<1:ap.error('--time-scale must be finite and at least 1')
    validate(a.source.resolve(),a.output.resolve(),a.time_scale)


if __name__=='__main__':main()
