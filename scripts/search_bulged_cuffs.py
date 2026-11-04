#!/usr/bin/env python3
"""Re-search expanded cuff states using unchanged geometry and right-arm gates."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,sys,hashlib,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,orientation,unit
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area
from reevaluate_geometric_closure import insertion,placements

def task(arguments):
    number,theta_step,ik_seeds,tilts=arguments;folder=ROOT/f'artifacts/bulged-cuff/pose-{number:02d}';out=folder/'search';out.mkdir(exist_ok=True)
    state_path=folder/'state.json';state=json.loads(state_path.read_text());samples=json.loads((folder/'samples.json').read_text());audit=json.loads((folder/'report.json').read_text())
    if not audit['passed']:raise RuntimeError(f'Bad deformed state {number}')
    native=json.loads((folder/'native-scene.json').read_text());scene=json.loads((ROOT/native.pop('shared_robot_scene')).read_text());scene.update(native)
    k=Kinematics(ROOT/audit['source_urdf'],scene);k.preferred_seed=k.start
    hand=k.hand;skin=SkinBody(state['source']['skin_surface']);points=np.array(state['particles_world']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();cloth=ClothScreen(points,faces);checker=CollisionScene(k,scene,faces)
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());fingers=[v for name,v in hand.vertices(.001).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
    source=state['source'];forward=unit(np.array(source['toe'])-source['ankle']);dorsal=np.array(source['ankle'])-source['heel'];sole=-unit(dorsal-forward*(dorsal@forward))
    counts=Counter();accepted=[];geometry=[];old_rows=[r for r in json.loads((ROOT/f'artifacts/geometric-closure/pose-{number:02d}/report.json').read_text())['candidates'] if r['accepted']]
    old_map={};old_results={r['candidate_index']:dict(candidate_index=r['candidate_index'],evaluated=False) for r in old_rows}
    for r in old_rows:
        c=r['candidate'];old_map.setdefault((c['sample_index'],c['yaw_deg'],c['tilt_deg'],c['closing_sign']),[]).append(r)
    for sample in samples:
      if sample['theta_index']%theta_step:continue
      p=np.array(sample['position']);dist=material_distances(rest,faces,faces[sample['face']],sample['weights'])
      for yaw in range(0,360,30):
       for tilt in tilts:
        for sign in [1,-1]:
            R=orientation(sample,yaw,tilt,sign);key=(sample['index'],yaw,tilt,sign);old=old_map.get(key,[])
            def mark(stage,**values):
                for r in old:old_results[r['candidate_index']]=dict(candidate_index=r['candidate_index'],evaluated=True,stage=stage,**values)
            counts['orientations_tested']+=1
            clear,clearance,_,_=skin.inspect(hand.bank(.001),R,p,.001)
            if not clear:counts['closed_foot_rejected']+=1;mark('closed_foot_rejected');continue
            counts['closed_foot_passed']+=1
            projection=projected_cloth_area((points-p)@R,faces,fingers,dist,depth,**{name:settings[name] for name in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
            if min(projection['area_m2'],projection['largest_component_area_m2'])<4e-6 or projection['area_fraction']<.05 or min(projection['span_x_m'],projection['span_z_m'])<.0005:counts['area_rejected']+=1;mark('area_rejected',area_mm2=projection['area_m2']*1e6);continue
            c=dict(sample_index=sample['index'],yaw_deg=yaw,tilt_deg=tilt,closing_sign=sign,rotation_matrix=R.tolist(),grasp_center_unity_world_m=p.tolist());gaps=[]
            for gap in [.0015,.002,.003,.004,.005,.0075,.01,.015,.02,.025,.03,.04,.05]:
                bank=hand.bank(gap,'approach',approach=.05);clear,reach_distance,_,_=skin.inspect(bank,R,p,.001)
                if not clear or cloth.intersections(bank,R,p):continue
                insert=insertion(hand,skin,dict(c,gap_m=gap),sample)
                if not insert['passed']:continue
                clear,close_distance,_,_=skin.inspect(hand.bank(gap,'closure',pinch_gap=.001),R,p,.001)
                if clear:gaps.append(dict(gap_m=gap,approach_skin_clearance_m=reach_distance,closure_skin_clearance_m=close_distance,insertion=insert))
            if not gaps:counts['opening_or_approach_rejected']+=1;mark('opening_or_approach_rejected');continue
            counts['geometry_accepted_poses']+=1;counts['geometry_accepted_configurations']+=len(gaps)
            record=dict(number=number,index=counts['geometry_accepted_poses']-1,candidate=c,projection=projection,valid_openings=gaps,
                sole_alignment=float(-R[:,0]@sole),source_state=str(state_path.relative_to(ROOT)),torso_ik_enabled=False)
            solutions,attempts=k.multistart(p,R,count=ik_seeds);record['ik_attempts']=attempts;record['endpoint_ik_solution_count']=len(solutions)
            pairs=[]
            for sol in sorted(solutions,key=lambda r:float(np.minimum(np.array(r['q'])-k.lower,k.upper-np.array(r['q'])).min()),reverse=True):
                q=np.array(sol['q']);margin=float(np.rad2deg(np.minimum(q-k.lower,k.upper-q)).min())
                if margin<5:continue
                pre=k.solve(p-R[:,0]*.05,R,q)
                if pre['position_error_m']>.0001 or pre['orientation_error_deg']>.5:continue
                premargin=float(np.rad2deg(np.minimum(np.array(pre['q'])-k.lower,k.upper-np.array(pre['q']))).min())
                if premargin<5:continue
                pairs.append(dict(endpoint=sol,preapproach=pre,minimum_joint_limit_margin_deg=min(margin,premargin)))
            record['comfortable_pairs']=pairs
            selected=[]
            for pair in pairs:
                q=np.array(pair['endpoint']['q']);pre_q=np.array(pair['preapproach']['q']);closed=checker.check(q,.001,allow_cloth_overlap=True)
                if not closed['passed']:continue
                for g in gaps:
                    gap=g['gap_m'];end=checker.check(q,gap);start=checker.check(pre_q,gap)
                    if not end['passed'] or not start['passed']:continue
                    selected.append(dict(pair=pair,opening_gap_m=gap,closed_collision=closed,open_collision=end,preapproach_collision=start))
                if selected:break
            record['arm_valid_configurations']=selected
            if not solutions:counts['ik_rejected_poses']+=1;mark('ik_rejected')
            elif not pairs:counts['preapproach_or_comfort_rejected_poses']+=1;mark('preapproach_or_comfort_rejected')
            elif not selected:counts['whole_arm_collision_rejected_poses']+=1;mark('whole_arm_collision_rejected')
            else:
                counts['arm_accepted_poses']+=1;counts['arm_accepted_configurations']+=len(selected);accepted.append(record)
                mark('arm_accepted',original_opening_gap_still_valid=[r['candidate_index'] for r in old if any(abs(a['opening_gap_m']-r['opening_gap_m'])<1e-9 for a in selected)])
                print('ACCEPT',number,record['index'],'sample',sample['index'],'yaw',yaw,'tilt',tilt,'area',projection['area_m2']*1e6,flush=True)
            (out/f"geometry-{record['index']:04d}.json").write_text(json.dumps(record,indent=2)+'\n');geometry.append(dict(index=record['index'],candidate=c,area_mm2=projection['area_m2']*1e6,endpoint_ik_solutions=len(solutions),comfortable_pairs=len(pairs),arm_valid_configurations=len(selected),sole_alignment=record['sole_alignment']))
      if sample['theta_index']==0:print('progress',number,'depth',sample['depth_m'],dict(counts),flush=True)
    result=dict(number=number,pitch_deg=audit['pitch_deg'],roll_deg=audit['roll_deg'],counts=dict(counts),geometry=geometry,accepted=[dict(index=r['index'],path=f"geometry-{r['index']:04d}.json",candidate=r['candidate'],area_mm2=r['projection']['area_m2']*1e6,configurations=len(r['arm_valid_configurations']),sole_alignment=r['sole_alignment']) for r in accepted],old60_parameter_recheck=list(old_results.values()),
        settings=dict(theta_step=theta_step,yaw_step_deg=30,tilts_deg=tilts,ik_seeds=ik_seeds,opening_gaps_m=[.0015,.002,.003,.004,.005,.0075,.01,.015,.02,.025,.03,.04,.05],torso_ik_enabled=False,position_tolerance_m=.0001,orientation_tolerance_deg=.5,minimum_joint_margin_deg=5),
        source_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),global_reaching_path_verified=False)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');return {key:value for key,value in result.items() if key not in ('geometry','accepted','old60_parameter_recheck')}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--numbers',type=int,nargs='+');parser.add_argument('--theta-step',type=int,default=1);parser.add_argument('--ik-seeds',type=int,default=48);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--tilts',type=int,nargs='+',default=[-60,-30,0,30,60]);a=parser.parse_args()
    summary=json.loads((ROOT/'artifacts/bulged-cuff/summary.json').read_text());numbers=a.numbers or [r['number'] for r in summary['poses']]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(task,[(n,a.theta_step,a.ik_seeds,a.tilts) for n in numbers]))
    counts=Counter()
    for row in rows:counts.update(row['counts'])
    (ROOT/'artifacts/bulged-cuff/search-summary.json').write_text(json.dumps(dict(poses=rows,counts=dict(counts)),indent=2)+'\n');print(dict(counts),flush=True)
if __name__=='__main__':main()
