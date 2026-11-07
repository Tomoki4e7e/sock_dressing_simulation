#!/usr/bin/env python3
"""Search immutable B shapes with dense samples and finite-pad TCP offsets.

The physical hand, 1 mm skin margin, 1 mm closing aperture and 4 mm2 capture
rule are unchanged. Offsets change hand placement relative to its existing 8 mm
capture window, not its dimensions or the cloth. Every candidate is rechecked
at the actual bounded-IK pose and throughout jaw closure.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,copy,json,sys,time
from pathlib import Path
from collections import Counter
import numpy as np,yaml,trimesh
from scipy.spatial.transform import Rotation
from shapely.errors import GEOSException
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import geometric_check,write,REGISTERED_MANIFEST
from search_fixed_torso_starts import context
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics,ControlledCollisionScene
from sock_dressing_simulation.grasp_geometry import candidate_samples,SkinBody,orientation
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances
from sock_dressing_simulation.reaching_dataset import file_sha256
from sock_dressing_simulation.fitted_b_cuff import PROFILE


def search(number,out,seconds,seed,limit,normal_offset_m=.002,tangent_offset_m=.006):
    state_path=ROOT/f'artifacts/diverse-left-reaching/b-natural-toe/final-shapes/pose-{number:02d}/state.json'
    state=json.loads(state_path.read_text());shape_sha=file_sha256(state_path)
    if state['shape_profile']!=PROFILE or not state['cuff_fit']['passed']:raise ValueError('Require qualified current B shape')
    initial_path=ROOT/'config/diverse_left_reaching_shared.yaml';initial_sha=file_sha256(initial_path);initial=yaml.safe_load(initial_path.read_text())
    start=np.array(initial['left_q_rad']);delta=np.array(initial['translation_m']);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text())
    samples=candidate_samples(state,circumferential_samples=64,depths_m=np.linspace(0,.030,13))
    sample_path=out/'samples'/f'pose-{number:02d}-B.json';write(sample_path,samples)
    # Keep a private immutable geometry input for downstream route hashes.
    frozen_path=out/'states'/f'pose-{number:02d}-B.json';write(frozen_path,state)
    base,scene,*_=context(number,state,REGISTERED_MANIFEST,left_foot_floor_allowance=initial['left_foot_floor_allowance_m'])
    scene.park_left(start,initial['left_gap_m']);model=ControlledArmKinematics(base,'left',initial['right_q_rad'],initial['right_gap_m']);checker=ControlledCollisionScene(scene,'left',initial['right_q_rad'],initial['right_gap_m'])
    initial_check=checker.slacks(start,initial['left_gap_m'],delta,details=True)
    comfortable=copy.copy(model);comfortable.lower=model.lower+np.deg2rad(5);comfortable.upper=model.upper-np.deg2rad(5)
    skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);closed=model.hand.bank(.001)
    points=np.array(state['particles_world']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);distances={}
    _,skin_distance,_=trimesh.proximity.closest_point_naive(skin.mesh,np.array([s['position'] for s in samples]))
    order=sorted(range(len(samples)),key=lambda i:(-skin_distance[i],abs(samples[i]['depth_m']-.015)))
    old=json.loads((ROOT/'artifacts/photo-grasp-adjustment/planned/target-left-0004.json').read_text());qref=np.array(old['attempts'][0]['q'])
    rng=np.random.default_rng(seed+number);counts=Counter();accepted=[];near=[];promising=[];best_area=0.;began=time.monotonic();total=0
    rotations=[(yaw,tilt,roll) for yaw in (330,0,180,90,270,300,30,150,210,60,120,240) for tilt in (-45,0,45,-75,75) for roll in (135,90,180,0,45,225,270,315)]
    offsets=[(0.,0.,0.)]+[(x,0.,z) for x,z in ((-.004,0),(.004,0),(0,-.004),(0,.004),(-.003,-.003),(-.003,.003),(.003,-.003),(.003,.003))]
    def consider(index,R,offset):
        nonlocal total,best_area
        if time.monotonic()-began>=seconds:return False
        total+=1;s=samples[index];p=np.array(s['position'])+R@offset
        clear,clearance,part,reason=skin.inspect(closed,R,p,.001)
        if not clear:
            counts['closed_foot']+=1;return False
        counts['closed_skin_passed']+=1
        if index not in distances:distances[index]=material_distances(rest,faces,faces[s['face']],s['weights'])
        shifted=dict(s,position=p.tolist());diagnostics={}
        try:geo,reason=geometric_check(model,state,shifted,R,skin,cloth,settings,distances[index],approach_distance=0.,diagnostics=diagnostics)
        except GEOSException:
            counts['numerical_geometry_rejected']+=1;return False
        counts[reason]+=1
        for failure,value in diagnostics.get('opening_rejections',{}).items():counts['opening_'+failure]+=value
        projection=diagnostics.get('projection',{});score=min(projection.get('area_m2',0.),projection.get('largest_component_area_m2',0.))
        best_area=max(best_area,score)
        if score>=1e-6:
            promising.append((score,index,R.copy(),np.array(offset).copy()));promising.sort(key=lambda item:-item[0]);del promising[32:]
        if not geo:
            if len(near)<32:near.append(dict(sample_index=index,reason=reason,rotation=R.tolist(),offset_local_m=list(offset),skin_clearance_m=clearance,material_depth_m=s['depth_m'],theta_deg=s['theta_deg'],projected_area_mm2=projection.get('area_m2',0.)*1e6,connected_area_mm2=score*1e6,opening_rejections=diagnostics.get('opening_rejections',{})))
            return False
        # Include multiple arm branches; no distance-to-start search restriction.
        seeds=[start,qref]+list(rng.uniform(comfortable.lower,comfortable.upper,(4,7)))
        for qseed in seeds:
            if time.monotonic()-began>=seconds:break
            sol=comfortable.solve(p+delta,R,np.clip(qseed,comfortable.lower+1e-9,comfortable.upper-1e-9))
            if sol['position_error_m']>1e-4 or sol['orientation_error_deg']>.5:counts['ik_failed']+=1;continue
            q=np.array(sol['q']);actual,actual_R=model.pose(q)
            actual_sample=dict(s,position=(actual-delta).tolist())
            try:actual_geo,actual_reason=geometric_check(model,state,actual_sample,actual_R,skin,cloth,settings,distances[index],approach_distance=0.)
            except GEOSException:counts['actual_ik_numerical_rejected']+=1;continue
            if not actual_geo:counts['actual_ik_'+actual_reason]+=1;continue
            for opening in sorted(actual_geo['valid_openings'],key=lambda r:-r['gap_m']):
                gap=opening['gap_m']
                if not checker.slacks(q,gap,delta,details=True)['passed']:counts['endpoint_collision']+=1;continue
                if not all(checker.slacks(q,float(g),delta,allow_cloth_overlap=True,details=True)['passed'] for g in np.linspace(gap,.001,21)):
                    counts['closure_collision']+=1;continue
                if any(np.linalg.norm(q-np.array(a['q']))<.03 for a in accepted):counts['duplicate_endpoint']+=1;continue
                accepted.append(dict(q=q.tolist(),gap_m=gap,geometry=actual_geo,sample=s,tool_center_offset_local_m=list(offset),material_point_world=s['position'],actual_ik_pose_rechecked=True))
                counts['accepted']+=1;print('DENSE B ENDPOINT',number,len(accepted),'seconds',round(time.monotonic()-began,1),flush=True);return True
        return False
    if initial_check['passed']:
        # Concentrate the deterministic sweep on the largest existing cuff/skin
        # gaps, then broaden over all band positions with continuous proposals.
        for index in order[:32]:
            for yaw,tilt,roll in rotations:
                R=orientation(samples[index],yaw,tilt)@Rotation.from_rotvec([np.deg2rad(roll),0,0]).as_matrix()
                for offset in offsets:
                    consider(index,R,np.array(offset))
                    if len(accepted)>=limit or time.monotonic()-began>=seconds:break
                if len(accepted)>=limit or time.monotonic()-began>=seconds:break
            if len(accepted)>=limit or time.monotonic()-began>=seconds:break
        while len(accepted)<limit and time.monotonic()-began<seconds:
            if promising and rng.random()<.5:
                _,index,R0,offset0=promising[int(rng.integers(min(8,len(promising))))]
                R=Rotation.from_rotvec(rng.normal(0,np.deg2rad(3),3)).as_matrix()@R0
                offset=np.clip(offset0+rng.normal(0,[.001,.0005,.001]),[-tangent_offset_m,-normal_offset_m,-tangent_offset_m],[tangent_offset_m,normal_offset_m,tangent_offset_m])
                consider(index,R,offset);continue
            # Half are unbiased over the band, half favour maximum skin gaps.
            index=int(rng.choice(order[:128] if rng.random()<.5 else order))
            yaw=float(rng.uniform(0,360));tilt=float(rng.uniform(-85,85));roll=float(rng.uniform(0,360))
            R=orientation(samples[index],yaw,tilt)@Rotation.from_rotvec([np.deg2rad(roll),0,0]).as_matrix()
            offset=np.array([rng.uniform(-tangent_offset_m,tangent_offset_m),rng.uniform(-normal_offset_m,normal_offset_m),rng.uniform(-tangent_offset_m,tangent_offset_m)])
            consider(index,R,offset)
    assert file_sha256(state_path)==shape_sha and file_sha256(initial_path)==initial_sha
    result=dict(number=number,stage='B',shape_profile=PROFILE,initial_configuration=str(initial_path),initial_sha256=initial_sha,workbook_sha256=file_sha256(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx'),initial_check=initial_check,accepted=accepted,counts=dict(counts),elapsed_s=time.monotonic()-began,search_mode='dense_band_samples_finite_pad_TCP_offsets',seed=seed+number,pose_trials=total,best_connected_capture_area_mm2=best_area*1e6,samples_file=str(sample_path.relative_to(ROOT)),samples_sha256=file_sha256(sample_path),frozen_source_file=str(state_path.relative_to(ROOT)),frozen_source_sha256=shape_sha,geometric_policy_sha256=file_sha256(ROOT/'config/geometric_closure.yaml'),promising_poses=[dict(connected_area_mm2=a*1e6,sample_index=i,rotation=R.tolist(),offset_local_m=o.tolist()) for a,i,R,o in promising[:8]],near_candidates=near,coverage=dict(circumferential_samples=64,depths_m=np.linspace(0,.030,13).tolist(),local_TCP_offset_limit_m=[tangent_offset_m,normal_offset_m,tangent_offset_m],full_band_exhausted=False))
    write(out/f'pose-{number:02d}-B.json',result);print('DENSE B RESULT',number,'found',len(accepted),'trials',total,dict(counts),'seconds',round(result['elapsed_s'],1),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--numbers',type=int,nargs='+',default=[8,10,3,15,19,22,23,31,38,41]);ap.add_argument('--output',type=Path,default=ROOT/'artifacts/diverse-left-reaching/b-natural-toe/grasp-search/dense-probe');ap.add_argument('--seconds',type=float,default=120);ap.add_argument('--seed',type=int,default=20261105);ap.add_argument('--endpoints',type=int,default=2);ap.add_argument('--normal-offset-mm',type=float,default=2.);ap.add_argument('--tangent-offset-mm',type=float,default=6.);a=ap.parse_args()
    if not np.isfinite(a.seconds) or a.seconds<=0 or a.endpoints<1:ap.error('Require a positive search budget and endpoint count')
    if not all(np.isfinite(v) and v>=0 for v in (a.normal_offset_mm,a.tangent_offset_mm)):ap.error('TCP offset bounds must be finite and nonnegative')
    for n in a.numbers:search(n,a.output.resolve(),a.seconds,a.seed,a.endpoints,a.normal_offset_mm*.001,a.tangent_offset_mm*.001)
