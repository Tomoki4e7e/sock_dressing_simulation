#!/usr/bin/env python3
"""Add a small approach/terminal clearance without relaxing contact criteria."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import fcl
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from verify_contact_pregrasp_continuous import make_scene
from sock_dressing_simulation.continuous_pregrasp import SweptPolicy,trajectory_intervals
from sock_dressing_simulation.common_opening_approach import linear_quintic_profile,sha256
from search_contact_pregrasp import export_route_csv
from search_centered_reaching import write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--id',required=True)
    parser.add_argument('--side',choices=['left','right'],required=True)
    parser.add_argument('--obstacle',choices=['remote_cloth','remote_skin'],default='remote_cloth')
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    original=json.loads((source/'routes'/f'{args.id}.json').read_text())
    scene,_,initial=make_scene(original['condition'],source,SweptPolicy())
    side=args.side;sl=slice(0,7) if side=='left' else slice(7,14)
    final=trajectory_intervals(original,initial)[-1]['end'];scene.update(final)
    skin=args.obstacle=='remote_skin'
    finger=side+'_gripper/'+original['final_contact_checks'][side]['foot_finger' if skin else 'cloth_finger']
    remote=scene.split['human_full_skin' if skin else 'frozen_cloth'][1];nearest=None
    for row in scene.base.groups[finger]:
        report=fcl.DistanceResult();distance=fcl.distance(scene.base.query_object(row),remote,fcl.DistanceRequest(enable_nearest_points=True),report)
        if nearest is None or distance<nearest[0]:nearest=(distance,np.array(report.nearest_points))
    if nearest[0]<=0:raise ValueError('Expected a clear nominal endpoint')
    normal=nearest[1][0]-nearest[1][1];normal/=np.linalg.norm(normal)
    model=scene.models[side];q=final[sl];p,R=model.pose(q);J=np.zeros((6,7));h=1e-6
    for j in range(7):
        trial=q.copy();trial[j]+=h;point,rotation=model.pose(trial)
        J[:,j]=np.r_[(point-p)/h,Rotation.from_matrix(rotation@R.T).as_rotvec()/h]
    trials=[]
    for distance in [.00005,-.00005,.000075,-.000075,.0001,-.0001,.000025,-.000025]:
        offset=np.linalg.pinv(J)@np.r_[normal*distance,np.zeros(3)]
        route=copy.deepcopy(original);new_final=final.copy();new_final[sl]+=offset
        scene.update(new_final);point,rotation=model.pose(new_final[sl])
        contact=scene.goal.inspect(point-scene.delta,rotation,float(new_final[14 if side=='left' else 15]),side,
                                   original['final_contact_checks'][side]['cloth_finger'])
        whole=scene.base.slacks(new_final[7:14],new_final[15],scene.delta,details=True)
        measured=scene._distance([scene.base.query_object(row) for row in scene.base.groups[finger]],remote)
        trials.append(dict(offset_m=distance,contact_passed=contact['passed'],whole_passed=whole['passed'],remote_clearance_m=measured,whole=whole,contacts=contact['contacts']))
        if not contact['passed'] or not whole['passed'] or measured<abs(distance)*.7:continue
        route['terminal'][side].update(q_rad=new_final[sl].tolist(),tcp_world_m=point.tolist(),rotation=rotation.tolist(),contact_check=contact)
        route['final_contact_checks'][side]=contact;route['simultaneous_endpoint_check']=whole
        for segment in route['segments']:
            if segment['side']==side:
                knots=np.asarray(segment['knots_q_rad']);knots[min(2,len(knots)-1):]+=offset
                profile=linear_quintic_profile(knots,model.velocity,model.acceleration)
                segment.update(raw_q_rad=knots.tolist(),knots_q_rad=profile['knots_q_rad'],segment_durations_s=profile['segment_durations_s'],
                               q_rad=profile['q'].tolist(),times_s=profile['t'].tolist(),velocity_rad_s=profile['velocity'].tolist(),
                               acceleration_rad_s2=profile['acceleration'].tolist(),duration_s=profile['duration'],continuous_collision_certified=False)
            elif np.max(abs(np.array(segment['fixed_other_q_rad'])-q))<1e-8:
                segment['fixed_other_q_rad']=new_final[sl].tolist()
        route['native_clearance_repair']=dict(original_route_sha256=sha256(source/'routes'/f'{args.id}.json'),side=side,
            obstacle=args.obstacle,
            intended_tcp_translation_m=(normal*distance).tolist(),joint_offset_rad=offset.tolist(),remote_clearance_m=measured,
            original_clearance_m=nearest[0],criteria_unchanged=True,continuous_reverification_required=True)
        route['native_clearance_repairs']=original.get('native_clearance_repairs',[])+[route['native_clearance_repair']]
        write(out/'routes'/f'{args.id}.json',route);export_route_csv(out/'routes'/f'{args.id}.csv',initial,route)
        write(out/f'repair-{args.id}.json',dict(passed=True,trials=trials,repair=route['native_clearance_repair']))
        print('REPAIRED',args.id,route['native_clearance_repair'],flush=True);return
    write(out/f'repair-{args.id}.json',dict(passed=False,trials=trials));raise RuntimeError('No contact-preserving clearance offset found')


if __name__=='__main__':main()
