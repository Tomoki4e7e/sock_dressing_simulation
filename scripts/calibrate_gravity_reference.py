#!/usr/bin/env python3
"""Read ShareSet teaching CSVs and identify a static, gravity-only reference."""
from __future__ import annotations
import os
for variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[variable]='1'
import argparse,csv,hashlib,io,json,sys,tarfile
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.signal import savgol_filter
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.gravity_reference import GravityTree,ARM_COLUMNS,ARM_NAMES,JOINT_NAMES,SOCK_TORSO


def digest(data):return hashlib.sha256(data).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def extract(archive,out):
    prefix='ShareSet/share_folder/data/data_center_general_depth_n5_sock/';rows=[]
    with tarfile.open(archive,'r:') as t:
        for name in sorted(t.getnames()):
            if not name.startswith(prefix) or not name.endswith('/angle.csv'):continue
            split,episode=name[len(prefix):].rsplit('/',1)[0].split('/',1);dest=out/'reference'/split/episode;dest.mkdir(parents=True,exist_ok=True);hashes={}
            for kind in ('angle','torque','external_torque'):
                data=t.extractfile(name.rsplit('/',1)[0]+'/'+kind+'.csv').read();(dest/(kind+'.csv')).write_bytes(data);hashes[kind]=digest(data)
            q=np.loadtxt(dest/'angle.csv',delimiter=',',ndmin=2)
            rows.append(dict(id=episode,split=split,path=str(dest.resolve()),frames=len(q),source_prefix=name.rsplit('/',1)[0],sha256=hashes))
        for member in ('ShareSet/share_folder/src/data/datarecord_gripper.py','ShareSet/share_folder/configs/joint_states.yaml','ShareSet/airec_joy_controller/scripts/change_pose.py','ShareSet/README.md'):
            data=t.extractfile(member).read();dest=out/'reference-source'/Path(member).name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
    manifest=dict(archive=str(archive),dataset=prefix,rows=rows,nominal_dt_s=.1,timestamps_available=False)
    write(out/'reference-manifest.json',manifest);return manifest


def terminal_interval(q,dt=.1):
    if q.ndim!=2 or q.shape[1]!=18 or len(q)<9 or not np.isfinite(q).all():raise ValueError('Invalid 18-column teaching angles')
    v=savgol_filter(q,9,2,deriv=1,delta=dt,axis=0);a=savgol_filter(q,9,2,deriv=2,delta=dt,axis=0)
    good=(q[:,7]>=.0095)&(q[:,16]>=.0095)&(np.max(abs(v[:,ARM_COLUMNS]),axis=1)<=.03)&(np.max(abs(a[:,ARM_COLUMNS]),axis=1)<=.1)
    start=len(q)
    while start and good[start-1]:start-=1
    if len(q)-start<6:return np.array([],dtype=int),v,a
    return np.arange(start,len(q)),v,a


def metrics(target,prediction):
    error=prediction-target
    return dict(rmse_Nm=float(np.sqrt(np.mean(error**2))),mae_Nm=float(np.mean(abs(error))),
                maximum_absolute_error_Nm=float(abs(error).max()),
                per_joint_rmse_Nm=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                per_joint_bias_Nm=error.mean(axis=0).tolist(),frames=len(error))


def fit(matrix,target,prior,regularization,strength,bounds):
    a=matrix.reshape(-1,matrix.shape[-1]);y=target.ravel()
    # Give each episode equal weight in the caller; regularize unobservable
    # mass/COM combinations towards the existing, independently specified URDF.
    reg=np.diag(regularization)*np.sqrt(strength)
    result=lsq_linear(np.vstack((a,reg)),np.r_[y,reg@prior],bounds=bounds,tol=1e-10,lsmr_tol=1e-10,max_iter=300)
    if not result.success:raise RuntimeError('Constrained gravity fit did not converge')
    return result.x


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,default=Path('/mnt/robot_ssd/ShareSet-from-Trash-20261106-151926/ShareSet.tar'))
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/gravity-calibration-v1')
    ap.add_argument('--urdf',type=Path,default=ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=extract(args.archive,out);tree=GravityTree(args.urdf);ng=len(tree.groups)
    prior=np.r_[np.ones(ng),np.zeros(ng*3)];mass=tree.group_masses
    regularization=np.r_[np.full(ng,1/.02),np.repeat(1/(mass*.002),3)]
    bounds=(np.r_[np.full(ng,.8),np.repeat(-mass*.01,3)],np.r_[np.full(ng,1.2),np.repeat(mass*.01,3)])
    episodes=[];segment_rows=[]
    for row in manifest['rows']:
        p=Path(row['path']);q=np.loadtxt(p/'angle.csv',delimiter=',');tau=np.loadtxt(p/'torque.csv',delimiter=',');ext=np.loadtxt(p/'external_torque.csv',delimiter=',')
        if tau.shape!=q.shape or ext.shape!=q.shape or not np.isfinite(tau).all() or not np.isfinite(ext).all():raise ValueError('CSV alignment or finite-value gate failed')
        mimic=float(max(np.max(abs(q[:,8]-.5*q[:,7])),np.max(abs(q[:,17]-.5*q[:,16]))))
        # Hardware publishes both finger coordinates: small transient tracking
        # differences are real observations, not the exact algebraic mimic
        # used by Unity. The gravity fit excludes these force channels.
        if mimic>5e-5:raise ValueError('Gripper column mapping differs from the documented 18-column contract')
        index,v,a=terminal_interval(q)
        if not len(index):raise ValueError(f'No released low-speed terminal interval: {row["id"]}')
        matrix=tree.regressor(q[index]);y=(tau-ext)[index][:,ARM_COLUMNS]
        episodes.append(dict(**row,q=q,tau=tau[:,ARM_COLUMNS],external=ext[:,ARM_COLUMNS],indices=index,matrix=matrix,target=y))
        record=dict(id=row['id'],split=row['split'],start=int(index[0]),end_exclusive=int(index[-1]+1),frames=len(index),
            duration_nominal_s=float((len(index)-1)*.1),maximum_arm_speed_rad_s=float(abs(v[index][:,ARM_COLUMNS]).max()),
            maximum_arm_acceleration_rad_s2=float(abs(a[index][:,ARM_COLUMNS]).max()),
            maximum_abs_external_effort_Nm=float(abs(ext[index][:,ARM_COLUMNS]).max()),
            contact_status='released-gripper and low-speed candidate; requires saved-image separation review',
            mimic_position_error_m=mimic)
        segment_rows.append(record);write(out/'segments'/row['id']/'selection.json',record)
        np.savetxt(out/'segments'/row['id']/'angle.csv',q[index],delimiter=',')
        np.savetxt(out/'segments'/row['id']/'link_effort_arm.csv',tau[index][:,ARM_COLUMNS],delimiter=',')
        np.savetxt(out/'segments'/row['id']/'external_effort_arm.csv',ext[index][:,ARM_COLUMNS],delimiter=',')
        np.savetxt(out/'segments'/row['id']/'gravity_proxy_arm.csv',y,delimiter=',')
    train=[e for e in episodes if e['split']=='train'];test=[e for e in episodes if e['split']=='test']
    if len(train)!=9 or len(test)!=2:raise ValueError('Expected the original nine training and two test sock episodes')
    # Model/regularization selection uses training episodes only. Test episodes
    # are evaluated once after selection, with no test-specific offsets.
    def stack(items):
        return np.concatenate([e['matrix']/np.sqrt(len(e['indices'])) for e in items]),np.concatenate([e['target']/np.sqrt(len(e['indices'])) for e in items])
    cv=[]
    for strength in (0.01,.1,1.,10.,100.):
        errors=[]
        for held in train:
            x,y=stack([e for e in train if e is not held]);p=fit(x,y,prior,regularization,strength,bounds)
            errors.append(metrics(held['target'],held['matrix']@p)['rmse_Nm'])
        cv.append(dict(strength=strength,episode_rmse_Nm=errors,mean_episode_rmse_Nm=float(np.mean(errors))))
    best=min(cv,key=lambda r:r['mean_episode_rmse_Nm']);x,y=stack(train)
    candidate=fit(x,y,prior,regularization,best['strength'],bounds)
    nominal_cv=float(np.mean([metrics(e['target'],e['matrix']@prior)['rmse_Nm'] for e in train]))
    # A small improvement below the 0.005 Nm threshold is insufficient evidence
    # to alter physical mass/COM parameters inferred from one held posture.
    selected=candidate if best['mean_episode_rmse_Nm']<nominal_cv-.005 else prior
    selection='regularized mass/first-moment fit' if selected is candidate else 'nominal URDF retained; CV improvement below 0.005 Nm'
    singular=np.linalg.svd(x.reshape(-1,len(prior)),compute_uv=False)
    static_offset=np.mean([np.median(e['tau'][e['indices']]-e['matrix']@selected,axis=0) for e in train],axis=0)
    proxy_test=np.concatenate([e['target'] for e in test]);test_matrix=np.concatenate([e['matrix'] for e in test]);raw_test=np.concatenate([e['tau'][e['indices']] for e in test])
    calibrated=out/'calibrated_gravity.urdf';tree.write_calibrated_urdf(calibrated,selected)
    copy_model=GravityTree(calibrated);all_q=np.concatenate([e['q'][e['indices']] for e in episodes]);all_matrix=np.concatenate([e['matrix'] for e in episodes])
    error=float(abs(copy_model.predict(all_q)-all_matrix@selected).max())
    if error>1e-8:raise RuntimeError(f'Calibrated URDF does not reproduce fitted gravity: {error}')
    for e in episodes:
        p=out/'predictions'/e['split']/e['id'];p.mkdir(parents=True,exist_ok=True)
        g=tree.regressor(e['q'])@selected
        np.savetxt(p/'gravity_arm.csv',g,delimiter=',')
        np.savetxt(p/'static_link_effort_estimate_arm.csv',g+static_offset,delimiter=',')
        write(p/'metadata.json',dict(joint_names=ARM_NAMES,units='N*m',nominal_dt_s=.1,source_prefix=e['source_prefix'],
            gravity_scope='URDF static gravity only; no drive, friction, inertia or contact terms',
            static_link_effort_scope='gravity plus training-derived static residual; this offset is not physical gravity',
            source_csv_sha256=e['sha256'],calibration_indices=e['indices'].tolist(),split=e['split']))
    model=dict(schema='shareset-static-gravity-reference-v1',joint_names=ARM_NAMES,input_joint_names=JOINT_NAMES,
        gravity_root_m_s2=[0.,0.,-9.81],fixed_joint_positions_rad=SOCK_TORSO,
        torso_source='Archive change_pose.py kb preset [-45,100,0] deg; not timestamped per episode',
        source_urdf=str(args.urdf.resolve()),source_urdf_sha256=digest(args.urdf.read_bytes()),
        calibrated_urdf=str(calibrated),calibrated_urdf_sha256=digest(calibrated.read_bytes()),
        groups=tree.groups,parameters=selected.tolist(),mass_scales=selected[:ng].tolist(),
        first_moment_delta_kg_m=selected[ng:].reshape(ng,3).tolist(),nominal_group_masses_kg=mass.tolist(),
        static_observation_offset_Nm=static_offset.tolist(),selection=selection,
        observation_target='link_effort minus external_effort, used as a quasi-static controller gravity proxy; not an independent gravity measurement',
        raw_link_effort_is_not_pure_gravity=True,contact_force_ground_truth_available=False,
        gripper_calibrated=False,full_dynamic_inertias_identified=False,
        mass_com_parameters_uniquely_identified=False,validated_for_robot_control=False)
    write(out/'gravity-model.json',model)
    report=dict(completed=True,archive=str(args.archive),episodes=len(episodes),segments=segment_rows,
        train_episodes=len(train),test_episodes=len(test),selected_frames=sum(len(e['indices']) for e in episodes),
        train_selected_frames=sum(len(e['indices']) for e in train),test_selected_frames=len(raw_test),
        selection_policy=dict(nominal_dt_s=.1,smoothing_window=9,minimum_samples=6,minimum_gripper_position_m=.0095,
            maximum_speed_rad_s=.03,maximum_acceleration_rad_s2=.1,terminal_contiguous_only=True,
            absolute_external_effort_below_1Nm_frames=sum(int(np.sum(np.max(abs(e['external']),axis=1)<1)) for e in episodes)),
        observation_definition=model['observation_target'],cross_validation=cv,nominal_cv_rmse_Nm=nominal_cv,
        selection=selection,test_nominal_gravity_vs_proxy=metrics(proxy_test,test_matrix@prior),
        test_calibrated_gravity_vs_proxy=metrics(proxy_test,test_matrix@selected),
        test_gravity_vs_raw_link_effort=metrics(raw_test,test_matrix@selected),
        test_gravity_plus_static_offset_vs_raw_link_effort=metrics(raw_test,test_matrix@selected+static_offset),
        zero_gravity_baseline_vs_proxy=metrics(proxy_test,np.zeros_like(proxy_test)),
        regressor_rank=int(np.sum(singular>singular[0]*1e-8)),regressor_columns=len(prior),singular_values=singular.tolist(),
        calibrated_urdf_equivalence_maximum_error_Nm=error,source_sha256=digest(Path(__file__).read_bytes()))
    write(out/'report.json',report)
    # Select review frames using q only, before looking at their torque errors.
    with tarfile.open(args.archive,'r:') as archive:
        sheet=Image.new('RGB',(400*3,330*len(episodes)),'white');draw=ImageDraw.Draw(sheet)
        for row,e in enumerate(episodes):
            selected_indices=sorted(set([int(e['indices'][0]),int(e['indices'][len(e['indices'])//2]),int(e['indices'][-1])]))
            for col,index in enumerate(selected_indices):
                member=e['source_prefix']+f'/camera_right/{index}.png';data=archive.extractfile(member).read()
                p=out/'segments'/e['id']/f'{index}.png';p.write_bytes(data)
                with Image.open(io.BytesIO(data)) as image:sheet.paste(image.convert('RGB').resize((400,300)),(col*400,row*330+30))
                draw.text((col*400+5,row*330+5),e['id']+f' / {index}',fill='black')
        sheet.save(out/'selected-contact-review.jpg',quality=92)
    print('GRAVITY REFERENCE FIT COMPLETE',json.dumps(dict(selection=selection,test_proxy_rmse=report['test_calibrated_gravity_vs_proxy']['rmse_Nm'],test_raw_rmse=report['test_gravity_vs_raw_link_effort']['rmse_Nm'],segments=len(episodes),frames=report['selected_frames'])),flush=True)


if __name__=='__main__':main()
