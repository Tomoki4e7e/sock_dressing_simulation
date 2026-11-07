#!/usr/bin/env python3
"""Refine near-feasible plantar approaches without relaxing geometric thresholds."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np,yaml
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,orientation
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area
from reevaluate_geometric_closure import insertion,placements

def run(number):
    out=ROOT/f'artifacts/sole-grasp-search/pose-{number:02d}'
    prior=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text());state=json.loads((ROOT/prior['input_state']).read_text());dense=json.loads((out/'dense-report.json').read_text())
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());hand=RightHand(ROOT/prior['settings']['urdf']);skin=SkinBody(state['source']['skin_surface']);points=np.array(state['particles_world']);faces=np.asarray(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();cloth=ClothScreen(points,faces)
    m=json.loads((ROOT/f'artifacts/arm-reaching-safe-start/pose-{number:02d}/manifest.json').read_text());scene=json.loads((ROOT/f'artifacts/arm-reaching-safe-start/pose-{number:02d}/initial-scene.json').read_text());k=Kinematics(ROOT/m['urdf'],scene);checker=CollisionScene(k,scene,faces)
    fingers=[v for name,v in hand.vertices(.001).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')];sole=np.array(dense['sole_outward_world']);counts=Counter();accepted=[];geometric=[];distances={};seen=set()
    for base in dense['closed_foot_rejected_examples'][:16]:
        sample=prior['samples'][base['sample_index']];idx=sample['index']
        if idx not in distances:distances[idx]=material_distances(rest,faces,faces[sample['face']],sample['weights'])
        for dyaw in [-5,0,5]:
          for dtilt in [-5,0,5]:
            original=orientation(sample,base['yaw_deg']+dyaw,base['tilt_deg']+dtilt,base['closing_sign'])
            if -original[:,0]@sole<.5:continue
            for roll in [-30,-15,0,15,30]:
              R=original@Rotation.from_euler('x',roll,degrees=True).as_matrix()
              for offset in [-.004,-.002,0,.002,.004]:
                key=(idx,base['yaw_deg']+dyaw,base['tilt_deg']+dtilt,base['closing_sign'],roll,offset)
                if key in seen:continue
                seen.add(key);counts['tested']+=1;p=np.array(sample['position'])+R[:,0]*offset
                clear,clearance,part,reason=skin.inspect(hand.bank(.001),R,p,.001)
                if not clear:counts['closed_foot_rejected']+=1;continue
                q=projected_cloth_area((points-p)@R,faces,fingers,distances[idx],depth,**{name:settings[name] for name in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
                if min(q['area_m2'],q['largest_component_area_m2'])<4e-6 or q['area_fraction']<.05 or min(q['span_x_m'],q['span_z_m'])<.0005:counts['area_rejected']+=1;continue
                c=dict(sample_index=idx,yaw_deg=base['yaw_deg']+dyaw,tilt_deg=base['tilt_deg']+dtilt,closing_sign=base['closing_sign'],roll_about_approach_deg=roll,tcp_approach_offset_m=offset,rotation_matrix=R.tolist(),grasp_center_unity_world_m=p.tolist())
                openings=[]
                for gap in [.005,.01,.02,.03,.04,.05]:
                    clear,dist,_,_=skin.inspect(hand.bank(gap,'approach',approach=.05),R,p,.001)
                    if not clear or cloth.intersections(hand.bank(gap,'approach',approach=.05),R,p):continue
                    insert=insertion(hand,skin,dict(c,gap_m=gap),sample)
                    if not insert['passed']:continue
                    clear,dist,_,_=skin.inspect(hand.bank(gap,'closure',pinch_gap=.001),R,p,.001)
                    if clear:openings.append(gap)
                if not openings:counts['approach_or_opening_rejected']+=1;continue
                counts['geometry_accepted']+=1;record=dict(number=number,candidate=c,projection=q,valid_openings_m=openings,sole_alignment=float(-R[:,0]@sole),source_state=prior['input_state']);geometric.append(record)
                rng=np.random.default_rng(1000+counts['geometry_accepted']);solutions=[]
                for seed in [k.start]+[rng.uniform(k.lower,k.upper) for _ in range(11)]:
                    sol=k.solve(p,R,seed)
                    if sol['position_error_m']>.0001 or sol['orientation_error_deg']>.5:continue
                    margin=float(np.rad2deg(np.minimum(np.array(sol['q'])-k.lower,k.upper-np.array(sol['q']))).min())
                    if margin<5:continue
                    pre=k.solve(p-R[:,0]*.05,R,np.array(sol['q']))
                    if pre['position_error_m']>.0001 or pre['orientation_error_deg']>.5:continue
                    margin=min(margin,float(np.rad2deg(np.minimum(np.array(pre['q'])-k.lower,k.upper-np.array(pre['q']))).min()))
                    if margin<5:continue
                    for gap in openings:
                        end=checker.check(np.array(sol['q']),gap);closed=checker.check(np.array(sol['q']),.001,allow_cloth_overlap=True);start=checker.check(np.array(pre['q']),gap)
                        if end['passed'] and closed['passed'] and start['passed']:solutions.append(dict(endpoint=sol,preapproach=pre,gap_m=gap,minimum_joint_limit_margin_deg=margin,open_collision=end,closed_collision=closed,preapproach_collision=start))
                if solutions:
                    record['solutions']=solutions;accepted.append(record);counts['arm_accepted']+=1
                    print('ACCEPT',number,c,'area',q['area_m2']*1e6,flush=True)
        print('refine',number,'sample',idx,dict(counts),flush=True)
    result=dict(number=number,counts=dict(counts),accepted=accepted,geometry_accepted=geometric,
        scope='16 near-clearance examples; yaw/tilt +-5 deg, approach roll +-30 deg, TCP approach offsets +-4 mm; original gap/area/skin thresholds unchanged',
        minimum_joint_limit_margin_deg=5,full_arm_endpoint_checks=True,global_reaching_path_checked=False)
    (out/'refined-report.json').write_text(json.dumps(result,indent=2)+'\n');return {k:v for k,v in result.items() if k not in ('accepted','geometry_accepted')}

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,[3,12]))
    (ROOT/'artifacts/sole-grasp-search/refined-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
