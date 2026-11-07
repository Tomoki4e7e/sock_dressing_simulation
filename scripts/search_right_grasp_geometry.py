#!/usr/bin/env python3
"""Search collision-free right-hand poses on all prepared ankle states.

No robot IK, cloth collision/force simulation or grasp-success assertion.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,candidate_samples,orientation,HANDEDNESS

ROS_UNITY=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def hand_info(hand):
    return dict(grasp_center_definition='Physical collision-mesh tip minus 10 mm along the approach axis; jaw/width mechanical centerline.',
        grasp_center_in_gripper_base_urdf_m=hand.grasp_center_urdf.tolist(),
        grasp_center_in_gripper_base_unity_m=(ROS_UNITY@hand.grasp_center_urdf).tolist(),
        physical_tip_in_gripper_base_urdf_x_m=hand.physical_tip_x_m,
        inner_face_local_z_m=hand.inner_face_local_z_m,
        maximum_gap_m=hand.maximum_gap_m,joint_for_maximum_gap_m=hand.joint_for_gap(hand.maximum_gap_m),
        mimic_joint_for_maximum_gap_m=hand.joint_for_gap(hand.maximum_gap_m)*.5,
        finger_joint_limits_m=hand.joint_limits,
        representation='Per-link convex hulls of the URDF collision STL meshes, matching runtime URDF MeshCollider.convex=true and Obi convex-solid routing.',
        urdf_sha256=sha(hand.urdf),meshes=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in hand.mesh_paths],
        ros_to_unity=ROS_UNITY.tolist(),canonical_handedness_reflection=HANDEDNESS.tolist())


def trial(arguments):
    settings,record,output=arguments;output=Path(output);output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    import fcl,trimesh
    state_path=(ROOT/record['report']).with_name('visual-state.json')
    state=json.loads(state_path.read_text())
    hand=RightHand(ROOT/settings['urdf'],settings['tip_inset_m'],settings['maximum_gap_m'])
    body=SkinBody(state['source']['skin_surface'])
    samples=candidate_samples(state,settings['circumferential_samples'],settings['material_depths_m'],settings['band_width_m'])
    pinch=settings['nominal_pinch_gap_m'];margin=settings['body_clearance_m'];approach=settings['approach_distance_m']
    gaps=sorted(settings['opening_gaps_m'],reverse=True)
    if any(g<pinch or g>hand.maximum_gap_m for g in gaps):raise ValueError('Opening grid exceeds physical gap limits')
    endpoint=hand.bank(pinch)
    for gap in gaps:
        hand.bank(gap,'closure',pinch,approach);hand.bank(gap,'approach',pinch,approach)
    # Artificial caps close a remote skin cut, and must never be an obstacle
    # near any possible hand location in this search.
    cap_bound=None
    if len(body.caps):
        caps=body.mesh.vertices[body.caps].reshape(-1,3)
        points=np.asarray([s['position'] for s in samples])
        bound=np.maximum(caps.min(0),np.minimum(caps.max(0),points))
        cap_distance=float(np.linalg.norm(points-bound,axis=1).min())
        reach=max(np.linalg.norm(p.vertices,axis=1).max() for p in hand.bank(hand.maximum_gap_m))+approach
        cap_bound=cap_distance-float(reach)
        if cap_bound<=margin:raise ValueError('Remote skin closure can affect hand search; cannot certify this state')
    rejected=Counter();counts=Counter();rows=[];accepted=[];tilt_counts={str(t):Counter() for t in settings['tilt_degrees']}
    total=len(samples)*len(settings['yaw_degrees'])*len(settings['tilt_degrees'])*len(settings['closing_signs'])
    for sample in samples:
        local=Counter();winner=None;static_winner=None
        position=np.asarray(sample['position'])
        for yaw in settings['yaw_degrees']:
            for tilt in settings['tilt_degrees']:
                for sign in settings['closing_signs']:
                    rotation=orientation(sample,yaw,tilt,sign)
                    valid,clearance,part,reason=body.inspect(endpoint,rotation,position,margin)
                    counts['tested']+=1
                    if not valid:
                        rejected[part+':'+reason]+=1;continue
                    counts['endpoint_clear']+=1;local['endpoint_clear']+=1
                    if sample['native_contact_face']:counts['native_endpoint_clear']+=1
                    basic=dict(yaw_deg=yaw,tilt_deg=tilt,closing_sign=sign,endpoint_clearance_m=clearance,
                        limiting_endpoint_part=part,rotation_matrix=rotation.tolist())
                    if static_winner is None or clearance>static_winner['endpoint_clearance_m']:static_winner=basic
                    best_closure=None;best_approach=None
                    for gap in gaps:
                        clear,cd,_,_=body.inspect(hand.bank(gap,'closure',pinch,approach),rotation,position,margin)
                        if not clear:continue
                        if best_closure is None:best_closure=dict(gap_m=gap,clearance_m=cd)
                        clear,ad,_,_=body.inspect(hand.bank(gap,'approach',pinch,approach),rotation,position,margin)
                        if clear:
                            best_approach=dict(gap_m=gap,closure_clearance_m=cd,approach_clearance_m=ad)
                            break
                    if best_closure is not None:counts['closure_clear']+=1;local['closure_clear']+=1
                    if best_approach is not None:
                        counts['approach_and_closure_clear']+=1;local['approach_and_closure_clear']+=1
                        if sample['native_contact_face']:counts['native_approach_and_closure_clear']+=1
                    candidate=dict(basic,maximum_checked_closure_opening=best_closure,maximum_checked_approach_opening=best_approach)
                    candidate['sample_index']=sample['index']
                    candidate['native_contact_face']=sample['native_contact_face']
                    accepted.append(candidate)
                    if sample['native_contact_face']:
                        tilt_counts[str(tilt)]['endpoint_clear']+=1
                        if best_approach:
                            tilt_counts[str(tilt)]['approach_clear']+=1
                            if best_approach['gap_m']>=.010:tilt_counts[str(tilt)]['approach_with_at_least_10mm_opening']+=1
                            if best_approach['gap_m']>=.050:tilt_counts[str(tilt)]['approach_with_50mm_opening']+=1
                    score=(best_approach['gap_m'] if best_approach else -1.,best_closure['gap_m'] if best_closure else -1.,clearance)
                    if winner is None or score>winner[0]:winner=(score,candidate)
        row=dict(sample,counts=dict(local),best_static_orientation=static_winner)
        if winner is not None:
            best=winner[1];rotation=np.asarray(best['rotation_matrix']);linear=rotation@HANDEDNESS
            # Certify a nested envelope of ALL openings between pinch and g
            # at ALL positions along the straight approach. Its convex hull
            # is exact for the rectangular translation parameter domain.
            low=pinch;high=hand.maximum_gap_m
            clear,cd,_,_=body.inspect(hand.bank(low,'combined',pinch,approach),rotation,position,margin)
            if clear:
                clear,max_cd,_,_=body.inspect(hand.bank(high,'combined',pinch,approach),rotation,position,margin)
                if clear:low=high;cd=max_cd
                else:
                    while high-low>.00025:
                        middle=(low+high)*.5
                        clear,mid_cd,_,_=body.inspect(hand.bank(middle,'combined',pinch,approach),rotation,position,margin)
                        if clear:low=middle;cd=mid_cd
                        else:high=middle
                best['certified_opening_interval_m']=[pinch,float(low)]
                best['certified_interval_clearance_m']=float(cd)
                best['opening_upper_bound_m']=float(high)
            else:
                best['certified_opening_interval_m']=None
            base_rotation=linear@ROS_UNITY.T
            best['gripper_base_position_unity_world_m']=(position-linear@hand.grasp_center_urdf).tolist()
            best['gripper_base_rotation_unity_world_xyzw']=Rotation.from_matrix(base_rotation).as_quat().tolist()
            best['gripper_base_rotation_unity_world_matrix']=base_rotation.tolist()
            best['grasp_center_unity_world_m']=position.tolist()
            row['best_orientation']=best
        rows.append(row)
        if len(rows)%32==0:
            print(f'pose {record["number"]:02d}: {counts["tested"]}/{total}, endpoint={counts["endpoint_clear"]}, approach+closure={counts["approach_and_closure_clear"]}',flush=True)
    for candidate in accepted:
        sample=samples[candidate['sample_index']];position=np.asarray(sample['position'])
        rotation=np.asarray(candidate['rotation_matrix']);linear=rotation@HANDEDNESS
        candidate['grasp_center_unity_world_m']=position.tolist()
        candidate['gripper_base_position_unity_world_m']=(position-linear@hand.grasp_center_urdf).tolist()
        candidate['gripper_base_rotation_unity_world_xyzw']=Rotation.from_matrix(linear@ROS_UNITY.T).as_quat().tolist()
        candidate['pinch_joint_m']=hand.joint_for_gap(pinch)
        candidate['pinch_mimic_joint_m']=hand.joint_for_gap(pinch)*.5
        if candidate['maximum_checked_approach_opening']:
            gap=candidate['maximum_checked_approach_opening']['gap_m']
            candidate['pregrasp_joint_m']=hand.joint_for_gap(gap)
            candidate['pregrasp_mimic_joint_m']=hand.joint_for_gap(gap)*.5
    active=[r for r in rows if r['native_contact_face']]
    result=dict(number=record['number'],pitch_deg=record['pitch_deg'],roll_deg=record['roll_deg'],
        status='sampled_geometry_search_completed',input_state=str(state_path.relative_to(ROOT)),input_sha256=sha(state_path),
        settings=settings,hand=hand_info(hand),versions=dict(python=sys.version,fcl=fcl.__version__,trimesh=trimesh.__version__),
        body=dict(source='Actual baked skin triangles; only remote boundary loops capped for solid containment.',
            actual_faces=len(np.asarray(state['source']['skin_surface']['triangles']).reshape(-1,3)),
            capped_faces=len(body.caps),remote_cap_separation_lower_bound_m=cap_bound,
            watertight=bool(body.mesh.is_watertight)),
        sampling=dict(surface_points=len(samples),native_contact_surface_points=len(active),orientations_per_point=total//len(samples),total_poses=total),
        counts=dict(counts),rejected_endpoint_parts=dict(rejected),counts_by_tilt_deg={k:dict(v) for k,v in tilt_counts.items()},
        native_contact_points_with_clear_endpoint=sum(r['counts'].get('endpoint_clear',0)>0 for r in active),
        native_contact_points_with_clear_approach=sum(r['counts'].get('approach_and_closure_clear',0)>0 for r in active),
        samples=rows,candidates=accepted,wall_time_s=time.monotonic()-started,
        limitations=['Finite orientation and material-position grid; absence of a candidate is not a proof of geometric impossibility.',
            'Nominal pinch gap is 5 mm, from the simulator external contact envelope, not a measured physical cloth thickness.',
            'Only hand versus selected human skin is checked; cloth outside the target, other robot links and arm IK are not checked.',
            'The 50 mm approach and closure sweeps are continuous translation sweeps at a fixed hand orientation.',
            'No contact rollout, grasp success, slip or holding verification.'])
    write_json(output/'report.json',result)
    columns=['sample_index','theta_deg','depth_mm','native_contact_face','yaw_deg','tilt_deg','closing_sign',
             'endpoint_clearance_mm','pregrasp_gap_mm','grasp_x_m','grasp_y_m','grasp_z_m',
             'base_x_m','base_y_m','base_z_m','qx','qy','qz','qw','pregrasp_joint_m','pregrasp_mimic_joint_m','pinch_joint_m']
    with (output/'candidates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for c in accepted:
            s=samples[c['sample_index']];opening=c['maximum_checked_approach_opening']
            row={k:c[k] for k in ['sample_index','native_contact_face','yaw_deg','tilt_deg','closing_sign','pinch_joint_m']}
            row.update(theta_deg=s['theta_deg'],depth_mm=s['depth_m']*1000,endpoint_clearance_mm=c['endpoint_clearance_m']*1000,
                pregrasp_gap_mm=opening['gap_m']*1000 if opening else None,
                pregrasp_joint_m=c.get('pregrasp_joint_m'),pregrasp_mimic_joint_m=c.get('pregrasp_mimic_joint_m'))
            row.update(zip(['grasp_x_m','grasp_y_m','grasp_z_m'],c['grasp_center_unity_world_m']))
            row.update(zip(['base_x_m','base_y_m','base_z_m'],c['gripper_base_position_unity_world_m']))
            row.update(zip(['qx','qy','qz','qw'],c['gripper_base_rotation_unity_world_xyzw']))
            writer.writerow(row)
    return {k:result[k] for k in ['number','pitch_deg','roll_deg','counts','sampling','native_contact_points_with_clear_endpoint','native_contact_points_with_clear_approach','wall_time_s']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/right_grasp_geometry.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/right-grasp-geometry')
    parser.add_argument('--numbers',type=int,nargs='+')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--quick',action='store_true',help='Small grid for code/geometry verification only')
    args=parser.parse_args();settings=yaml.safe_load(args.config.read_text());settings['configuration_sha256']=sha(args.config)
    if args.quick:
        settings.update(circumferential_samples=8,material_depths_m=[0.,.015,.030],yaw_degrees=[0,90,180,270],tilt_degrees=[-45,0,45])
    source=json.loads((ROOT/settings['summary']).read_text())
    if not source.get('passed'):raise ValueError('Prepared states have not passed stability and surface checks')
    records=source['trials']
    if args.numbers:
        records=[r for r in records if r['number'] in args.numbers]
        if len(records)!=len(set(args.numbers)):raise ValueError('Numbers must identify existing unique-angle trial representatives')
    args.output.mkdir(parents=True,exist_ok=True)
    hand=RightHand(ROOT/settings['urdf'],settings['tip_inset_m'],settings['maximum_gap_m'])
    write_json(args.output/'gripper-calibration.json',hand_info(hand))
    jobs=[(settings,r,str(args.output/f'pose-{r["number"]:02d}')) for r in records]
    completed=[]
    if args.workers==1:
        for job in jobs:completed.append(trial(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(trial,job) for job in jobs]):
                row=future.result();completed.append(row)
                print('COMPLETED '+json.dumps(row),flush=True)
                write_json(args.output/'progress.json',dict(completed=completed,expected=len(records)))
    completed.sort(key=lambda r:(r['pitch_deg'],r['roll_deg']))
    result=dict(completed=True,quick_grid=args.quick,settings=settings,hand=hand_info(hand),
        pose_count=len(completed),total_tested_poses=sum(r['counts'].get('tested',0) for r in completed),
        poses_with_native_clear_endpoint=sum(r['native_contact_points_with_clear_endpoint']>0 for r in completed),
        poses_with_native_clear_approach=sum(r['native_contact_points_with_clear_approach']>0 for r in completed),
        trials=completed,grasp_success_validated=False,arm_reachability_validated=False)
    write_json(args.output/'summary.json',result)
    from scripts.check_right_grasp_foot_proxy import check_outputs
    proxy=check_outputs(args.output)
    result['current_convex_proxy_poses_with_clear_endpoint']=proxy['poses_with_native_clear_endpoint']
    print(json.dumps({k:v for k,v in result.items() if k not in ['settings','hand','trials']},indent=2))


if __name__=='__main__':main()
