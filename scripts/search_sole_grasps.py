#!/usr/bin/env python3
"""Search new cuff grasps whose preapproach lies in the plantar hemisphere."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,hashlib,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,orientation,unit
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area
from reevaluate_geometric_closure import insertion,placements

def search(args):
    number,dense,offset,alignment_min=args;out=ROOT/f'artifacts/sole-grasp-search/pose-{number:02d}';out.mkdir(parents=True,exist_ok=True)
    prior=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text());state_path=ROOT/prior['input_state'];state=json.loads(state_path.read_text())
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());hand=RightHand(ROOT/prior['settings']['urdf']);skin=SkinBody(state['source']['skin_surface']);points=np.asarray(state['particles_world']);faces=np.asarray(state['triangles']).reshape(-1,3);cloth=ClothScreen(points,faces)
    rest=np.asarray(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min()
    source=state['source'];forward=unit(np.asarray(source['toe'])-source['ankle']);heel_to_ankle=np.asarray(source['ankle'])-source['heel'];dorsal=unit(heel_to_ankle-forward*(heel_to_ankle@forward));sole=-dorsal
    base=json.loads((ROOT/'artifacts/arm-reaching/pose-03/manifest.json').read_text());k=Kinematics(ROOT/base['urdf'],json.loads((ROOT/'artifacts/arm-reaching/pose-03/initial-scene.json').read_text()));safe=np.deg2rad([0,-60,0,90,0,-60,55])
    counts=Counter();accepted=[];geometry_accepted=[];ik_rejected=[];closed_rejected=[]
    samples=[s for s in prior['samples'] if s['native_contact_face'] and s['theta_index']%(2 if dense else 4)==0 and (dense or s['depth_m'] in (0.,.01,.02))]
    gaps=[.005,.01,.02,.03,.04,.05];fingers=[v for name,v in hand.vertices(.001).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
    yaws=list(range(0,360,15 if dense else 30));tilts=list(range(-75,76,15)) if dense else [-60,-45,-30,-15,0,15,30,45,60]
    for sample in samples:
        p=np.asarray(sample['position']);dist=material_distances(rest,faces,faces[sample['face']],sample['weights'])
        for yaw in yaws:
            for tilt in tilts:
                for sign in [1,-1]:
                    R=orientation(sample,yaw,tilt,sign);alignment=float((-R[:,0])@sole)
                    if alignment<alignment_min:continue
                    p=np.asarray(sample['position'])+R[:,0]*offset
                    counts['orientations']+=1
                    local=(points-p)@R
                    projection=projected_cloth_area(local,faces,fingers,dist,depth,**{key:settings[key] for key in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
                    if projection['area_m2']<4e-6 or projection['largest_component_area_m2']<4e-6 or projection['area_fraction']<.05 or min(projection['span_x_m'],projection['span_z_m'])<.0005:
                        counts['area_rejected']+=1;continue
                    clear,distance,part,reason=skin.inspect(hand.bank(.001),R,p,.001)
                    if not clear:
                        counts['closed_foot_rejected']+=1
                        closed_rejected.append(dict(sample_index=sample['index'],yaw_deg=yaw,tilt_deg=tilt,closing_sign=sign,rotation_matrix=R.tolist(),grasp_center_unity_world_m=p.tolist(),foot_clearance_m=distance,limiting_part=part,reason=reason,area_m2=projection['area_m2'],sole_alignment=alignment))
                        if len(closed_rejected)>32:closed_rejected=sorted(closed_rejected,key=lambda r:r['foot_clearance_m'],reverse=True)[:16]
                        continue
                    c=dict(sample_index=sample['index'],yaw_deg=yaw,tilt_deg=tilt,closing_sign=sign,rotation_matrix=R.tolist(),grasp_center_unity_world_m=p.tolist())
                    valid_gaps=[]
                    for gap in gaps:
                        clear,_,_,_=skin.inspect(hand.bank(gap,'approach',approach=.05),R,p,.001)
                        if not clear:continue
                        if cloth.intersections(hand.bank(gap,'approach',approach=.05),R,p):continue
                        candidate=dict(c,gap_m=gap);insert=insertion(hand,skin,candidate,sample)
                        if not insert['passed']:continue
                        clear,clearance,_,_=skin.inspect(hand.bank(gap,'closure',pinch_gap=.001),R,p,.001)
                        if clear:valid_gaps.append(dict(gap_m=gap,skin_clearance_m=clearance,insertion=insert))
                    if not valid_gaps:counts['opening_or_approach_rejected']+=1;continue
                    counts['geometry_accepted']+=1
                    record=dict(number=number,candidate=c,projection=projection,sole_alignment=alignment,valid_openings=valid_gaps,source_state=str(state_path.relative_to(ROOT)))
                    geometry_accepted.append(record)
                    rng=np.random.default_rng(number*10000+sample['index']*10+yaw+sign);solutions=[]
                    for seed in [safe,k.start]+[rng.uniform(k.lower,k.upper) for _ in range(10)]:
                        sol=k.solve(p,R,seed)
                        if sol['position_error_m']>.0001 or sol['orientation_error_deg']>.5:continue
                        q=np.asarray(sol['q']);margin=float(np.rad2deg(np.minimum(q-k.lower,k.upper-q)).min())
                        pre=k.solve(p-R[:,0]*.05,R,q)
                        if pre['position_error_m']>.0001 or pre['orientation_error_deg']>.5:continue
                        premargin=float(np.rad2deg(np.minimum(np.asarray(pre['q'])-k.lower,k.upper-np.asarray(pre['q']))).min())
                        solutions.append(dict(endpoint=sol,preapproach=pre,minimum_joint_limit_margin_deg=min(margin,premargin)))
                    if not solutions:
                        counts['ik_rejected']+=1;ik_rejected.append(record);continue
                    solutions.sort(key=lambda r:r['minimum_joint_limit_margin_deg'],reverse=True)
                    record['ik_solutions']=solutions;record['comfortable']=solutions[0]['minimum_joint_limit_margin_deg']>=5
                    accepted.append(record);counts['ik_accepted']+=1
                    print('sole',number,'sample',sample['index'],'yaw',yaw,'tilt',tilt,'IK',len(solutions),'margin',solutions[0]['minimum_joint_limit_margin_deg'],flush=True)
        print('progress',number,'sample',sample['index'],dict(counts),flush=True)
    result=dict(number=number,pitch_deg=prior['pitch_deg'],roll_deg=prior['roll_deg'],dense=dense,sample_count=len(samples),sole_outward_world=sole.tolist(),sole_definition='negative heel-to-ankle component orthogonal to ankle-to-toe axis',
        required_approach_sole_alignment=alignment_min,tcp_approach_offset_m=offset,counts=dict(counts),accepted=accepted,geometry_accepted=geometry_accepted,closed_foot_rejected_examples=closed_rejected,source_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),full_arm_collision_checked=False)
    filename=('wide-' if alignment_min<.5 else '')+('dense-' if dense else '')+(f'offset-{offset*1000:g}mm-' if offset else '')+'report.json'
    (out/filename).write_text(json.dumps(result,indent=2)+'\n');return {k:v for k,v in result.items() if k not in ('accepted','geometry_accepted','closed_foot_rejected_examples')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--numbers',type=int,nargs='+',default=[3,12]);p.add_argument('--dense',action='store_true');p.add_argument('--workers',type=int,default=2);p.add_argument('--tcp-offset-mm',type=float,default=0);p.add_argument('--minimum-sole-alignment',type=float,default=.5);a=p.parse_args()
    if abs(a.tcp_offset_mm)>4:raise ValueError('TCP offset must stay within the finite pad window')
    if not 0<=a.minimum_sole_alignment<=1:raise ValueError('Sole alignment must be in [0, 1]')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(search,[(n,a.dense,a.tcp_offset_mm/1000,a.minimum_sole_alignment) for n in a.numbers]))
    out=ROOT/'artifacts/sole-grasp-search';name=('wide-' if a.minimum_sole_alignment<.5 else '')+('dense-' if a.dense else '')+(f'offset-{a.tcp_offset_mm:g}mm-' if a.tcp_offset_mm else '')+'summary.json';(out/name).write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
