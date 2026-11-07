#!/usr/bin/env python3
"""Re-evaluate exactly the 436 screened configurations without contact dynamics."""
import argparse,csv,hashlib,json,sys,os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import yaml,fcl,trimesh
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,HANDEDNESS
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen,ray_interval
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area,validate_settings

def vector(v):return dict(zip('xyz',map(float,v)))
def placements(hand,c,gap):
    from scipy.spatial.transform import Rotation
    conversion=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])
    R=np.asarray(c['rotation_matrix']);L=R@HANDEDNESS;p=np.asarray(c['grasp_center_unity_world_m']);base=p-L@hand.grasp_center_urdf;result=[]
    for link,_ in hand.meshes.values():
        T=hand.link_transform(link,hand.joint_for_gap(gap));q=Rotation.from_matrix(L@T[:3,:3]@conversion.T).as_quat()
        result.append(dict(link=link,position=vector(base+L@T[:3,3]),rotation=dict(zip('xyzw',map(float,q)))))
    return result

def insertion(hand,skin,c,sample):
    R=np.asarray(c['rotation_matrix']);p=np.asarray(c['grasp_center_unity_world_m']);normal=np.asarray(sample['normal']);pads=[]
    for link,vertices in hand.vertices(c['gap_m']).items():
        if not (link.endswith('/left_finger_link') or link.endswith('/right_finger_link')):continue
        interval=ray_interval(ConvexHull(vertices).equations,0,0)
        if interval is None:return dict(passed=False,reason='no_finite_jaw_at_target')
        y=interval[1] if vertices[:,1].mean()<0 else interval[0]
        point=p+R[:,1]*y;pads.append(dict(link=link,point_world=point.tolist(),signed_cloth_normal_offset_m=float((point-p)@normal)))
    closest,distance,_=trimesh.proximity.closest_point_naive(skin.mesh,np.array([p]));toward=closest[0]-p
    inner=min(pads,key=lambda v:v['signed_cloth_normal_offset_m']);outer=max(pads,key=lambda v:v['signed_cloth_normal_offset_m'])
    pad_points=np.array([inner['point_world'],outer['point_world']]);_,pad_distances,_=trimesh.proximity.closest_point_naive(skin.mesh,pad_points)
    outside=bool(np.all(skin.winding(pad_points)<.5))
    directed=float((pad_points[0]-p)@toward)>0
    passed=inner['signed_cloth_normal_offset_m']<0<outer['signed_cloth_normal_offset_m'] and outside and directed and pad_distances[0]<distance[0]
    return dict(passed=bool(passed),reason='inside_outside_and_toward_foot' if passed else 'insertion_side_or_skin_direction_failed',
        inner=inner,outer=outer,target_skin_distance_m=float(distance[0]),inner_pad_skin_distance_m=float(pad_distances[0]),outer_pad_skin_distance_m=float(pad_distances[1]),pads_outside_skin=outside,
        scope='Pad-center side test plus whole-hand skin clearance and collision-free approach; not a tension/slack measurement.')

def trial(arguments):
    number,settings=arguments
    source_path=ROOT/f'artifacts/grasp-cloth-refilter/pose-{number:02d}/report.json';source=json.loads(source_path.read_text());prior=json.loads((ROOT/source['source_report']).read_text());state=json.loads((ROOT/source['source_state']).read_text())
    hand=RightHand(ROOT/prior['settings']['urdf'],prior['settings']['tip_inset_m'],prior['settings']['maximum_gap_m']);skin=SkinBody(state['source']['skin_surface']);faces=np.asarray(state['triangles']).reshape(-1,3);rest=np.asarray(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();distance_cache={};rows=[]
    for index,c in enumerate(source['accepted']):
        initial_file=ROOT/f'artifacts/grasp-closure-measurements/pose-{number:02d}/candidate-{index:03d}.json';initial=json.loads(initial_file.read_text())
        if initial['candidate']!=c or initial.get('protocol_version')!=5:raise ValueError('Source candidate or initial-state mismatch')
        points=np.asarray(initial['initial_particles_world']);current_faces=np.asarray(initial['triangles']).reshape(-1,3)
        if not np.array_equal(faces,current_faces):raise ValueError('Triangle ordering changed')
        R=np.asarray(c['rotation_matrix']);p=np.asarray(c['grasp_center_unity_world_m']);local=(points-p)@R;sample=prior['samples'][c['sample_index']]
        if c['sample_index'] not in distance_cache:distance_cache[c['sample_index']]=material_distances(rest,faces,faces[sample['face']],sample['weights'])
        cloth=ClothScreen(points,faces);opening=c['gap_m'];closing=settings['closing_gap_m'];margin=settings['body_clearance_m'];reach=settings['approach_distance_m']
        if closing>=opening:raise ValueError('Evaluation gap must be smaller than opening')
        approach=hand.bank(opening,'approach',approach=reach)
        reach_skin,reach_clearance,_,_=skin.inspect(approach,R,p,margin);reach_hits=cloth.intersections(approach,R,p)
        insert=insertion(hand,skin,c,sample)
        fingers={link:v for link,v in hand.vertices(closing).items() if link.endswith('/left_finger_link') or link.endswith('/right_finger_link')}
        projection=projected_cloth_area(local,faces,list(fingers.values()),distance_cache[c['sample_index']],depth,
            **{key:settings[key] for key in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
        end=hand.bank(closing)
        for part in end:part.obj.setTransform(fcl.Transform(np.eye(3),np.zeros(3)))
        jaws=[part for part in end if part.label.endswith('/left_finger_link') or part.label.endswith('/right_finger_link')]
        jaw_clear=not bool(fcl.collide(jaws[0].obj,jaws[1].obj,fcl.CollisionRequest(),fcl.CollisionResult()))
        end_skin,end_clearance,_,_=skin.inspect(end,R,p,margin)
        sweep=hand.bank(opening,'closure',pinch_gap=closing);sweep_skin,sweep_clearance,limiting,sweep_reason=skin.inspect(sweep,R,p,margin)
        final_cloth_hits=cloth.intersections(end,R,p)
        checks=dict(approach_skin_clear=reach_skin,approach_cloth_clear=not reach_hits,insertion_space=insert['passed'],
            closure_skin_clear=sweep_skin,closed_skin_clear=end_skin,opposite_jaws_clear=jaw_clear,
            projected_area=projection['area_m2']+1e-12>=settings['minimum_area_m2'],
            projected_fraction=projection['area_fraction']+1e-10>=settings['minimum_area_fraction'],
            connected_area=projection['largest_component_area_m2']+1e-12>=settings['minimum_component_area_m2'],
            finite_spans=projection['span_x_m']+1e-10>=settings['minimum_span_m'] and projection['span_z_m']+1e-10>=settings['minimum_span_m'])
        # Closed cloth overlap is intentionally diagnostic, not a rejection:
        # virtual closure evaluates frozen cloth; it does not simulate its deformation.
        row=dict(number=number,candidate_index=index,candidate=c,accepted=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
            opening_gap_m=opening,closing_gap_m=closing,closed_joint_m=hand.joint_for_gap(closing),closed_mimic_joint_m=hand.joint_for_gap(closing)*.5,
            execution_plan=dict(enabled=all(checks.values()),mode='geometry_only',close_to_gap_m=closing,stop_rule='configured_positive_gap_if_all_checks_pass; otherwise do_not_execute',friction_or_retention_condition=False,cloth_response=False),projection=projection,insertion=insert,
            approach_skin_clearance_m=reach_clearance,approach_cloth_intersections=reach_hits,
            closure_skin_clearance_m=sweep_clearance,closure_limiting_part=limiting,closure_skin_reason=sweep_reason,closed_skin_clearance_m=end_clearance,
            closed_cloth_intersections=final_cloth_hits,closed_cloth_overlap_allowed_for_virtual_closure=True,
            initial_state_source=str(initial_file.relative_to(ROOT)),initial_state_sha256=hashlib.sha256(initial_file.read_bytes()).hexdigest(),
            cloth_state='frozen V5 initial state; no cloth force, friction, capture constraint or retention test',
            approach_start_placements=placements(hand,dict(c,grasp_center_unity_world_m=(p-R[:,0]*reach).tolist()),opening),
            open_placements=placements(hand,c,opening),closed_placements=placements(hand,c,closing))
        rows.append(row)
    out=ROOT/f'artifacts/geometric-closure/pose-{number:02d}';out.mkdir(parents=True,exist_ok=True)
    result=dict(number=number,pitch_deg=source['pitch_deg'],roll_deg=source['roll_deg'],input_count=len(rows),accepted_count=sum(r['accepted'] for r in rows),settings=settings,candidates=rows)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('settings','candidates')}),flush=True);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=ROOT/'config/geometric_closure.yaml');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    settings=yaml.safe_load(args.config.read_text())
    validate_settings(settings)
    source=json.loads((ROOT/'artifacts/grasp-cloth-refilter/summary.json').read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:poses=list(pool.map(trial,[(p['number'],settings) for p in source['trials']]))
    rows=[r for pose in poses for r in pose['candidates']];assert len(rows)==436
    out=ROOT/'artifacts/geometric-closure';failures=Counter(k for r in rows for k,v in r['checks'].items() if not v)
    summary=dict(complete=True,input_count=len(rows),accepted_count=sum(r['accepted'] for r in rows),rejected_count=sum(not r['accepted'] for r in rows),
        poses_with_candidates=sum(p['accepted_count']>0 for p in poses),unique_ankle_states=len(poses),settings=settings,failed_checks_nonexclusive=dict(failures),
        closed_cloth_overlap_count=sum(bool(r['closed_cloth_intersections']) for r in rows),
        method='3D facet-cell clipping followed by 2D polygon union; target material patch only; exact linear closure sweeps against skin',
        physical_capture_or_hold_validated=False,closed_cloth_overlap_is_permitted=True,
        poses=[{k:p[k] for k in ('number','pitch_deg','roll_deg','input_count','accepted_count')} for p in poses])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['number','candidate_index','accepted','opening_gap_mm','closing_gap_mm','area_mm2','area_fraction','connected_area_mm2','span_x_mm','span_z_mm','approach_skin_clearance_mm','closure_skin_clearance_mm','insertion_space','failed_checks','closed_cloth_intersections']
    with (out/'candidates.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            q=r['projection'];writer.writerow(dict(number=r['number'],candidate_index=r['candidate_index'],accepted=r['accepted'],opening_gap_mm=r['opening_gap_m']*1000,closing_gap_mm=r['closing_gap_m']*1000,area_mm2=q['area_m2']*1e6,area_fraction=q['area_fraction'],connected_area_mm2=q['largest_component_area_m2']*1e6,span_x_mm=q['span_x_m']*1000,span_z_mm=q['span_z_m']*1000,approach_skin_clearance_mm=r['approach_skin_clearance_m']*1000,closure_skin_clearance_mm=r['closure_skin_clearance_m']*1000,insertion_space=r['insertion']['passed'],failed_checks=';'.join(k for k,v in r['checks'].items() if not v),closed_cloth_intersections=';'.join(r['closed_cloth_intersections'])))
    with (out/'poses.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['number','pitch_deg','roll_deg','input_count','accepted_count']);writer.writeheader();writer.writerows(summary['poses'])
    print(json.dumps({k:v for k,v in summary.items() if k!='poses'}),flush=True)
if __name__=='__main__':main()
