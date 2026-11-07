#!/usr/bin/env python3
"""Supplement the grid by proposing actual comfortable arm poses on the bulge."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np,yaml
from scipy.optimize import least_squares,minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene
from sock_dressing_simulation.grasp_geometry import SkinBody,unit
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area
from reevaluate_geometric_closure import insertion

def run(n):
    folder=ROOT/f'artifacts/bulged-cuff/pose-{n:02d}';state=json.loads((folder/'state.json').read_text());samples=json.loads((folder/'samples.json').read_text());audit=json.loads((folder/'report.json').read_text());native=json.loads((folder/'native-scene.json').read_text());scene=json.loads((ROOT/native.pop('shared_robot_scene')).read_text());scene.update(native);k=Kinematics(ROOT/audit['source_urdf'],scene)
    pnts=np.array(state['particles_world']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();hand=k.hand;skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(pnts,faces);checker=CollisionScene(k,scene,faces);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());fingers=[v for name,v in hand.vertices(.001).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
    s=state['source'];f=unit(np.array(s['toe'])-s['ankle']);d=np.array(s['ankle'])-s['heel'];sole=-unit(d-f*(d@f))
    lo=k.lower+np.deg2rad(5);hi=k.upper-np.deg2rad(5);rng=np.random.default_rng(15000+n);counts=Counter();accepted=[];geometric=[]
    for sample in samples:
      if sample['theta_index']%2 or sample['depth_m'] not in (.01,.015,.02):continue
      p=np.array(sample['position']);normal=np.array(sample['normal']);dist=material_distances(rest,faces,faces[sample['face']],sample['weights']);proposals=[]
      for seed in [k.start]+[rng.uniform(lo,hi) for _ in range(7)]:
        counts['position_ik_attempts']+=1
        result=least_squares(lambda q:k.pose(q)[0]-p,np.clip(seed,lo+1e-8,hi-1e-8),jac=lambda q:k.fast_pose(q)[2],bounds=(lo,hi),max_nfev=150,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if np.linalg.norm(k.pose(result.x)[0]-p)>.0001:continue
        proposals.append(result.x)
        def objective(q):
            R=k.pose(q)[1];alignment=abs(R[:,1]@normal);plantar=-R[:,0]@sole
            return (1-alignment)**2+.1*max(0,.5-plantar)**2+.005*np.sum(((q-k.start)/(hi-lo))**2)
        opt=minimize(objective,result.x,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=lambda q:k.pose(q)[0]-p)],options=dict(maxiter=100,ftol=1e-10))
        if np.linalg.norm(k.pose(opt.x)[0]-p)<=.0001:proposals.append(opt.x)
      for q in proposals:
        actual,R=k.pose(q);counts['arm_orientations_tested']+=1
        if not skin.inspect(hand.bank(.001),R,actual,.001)[0]:counts['closed_foot_rejected']+=1;continue
        projection=projected_cloth_area((pnts-actual)@R,faces,fingers,dist,depth,**{name:settings[name] for name in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
        if min(projection['area_m2'],projection['largest_component_area_m2'])<4e-6 or projection['area_fraction']<.05 or min(projection['span_x_m'],projection['span_z_m'])<.0005:counts['area_rejected']+=1;continue
        c=dict(sample_index=sample['index'],rotation_matrix=R.tolist(),grasp_center_unity_world_m=actual.tolist(),generator='comfortable_right_arm_configuration');gaps=[]
        for gap in [.0015,.002,.003,.004,.005,.0075,.01,.015,.02,.025,.03,.04,.05]:
            bank=hand.bank(gap,'approach',approach=.05)
            if not skin.inspect(bank,R,actual,.001)[0] or cloth.intersections(bank,R,actual):continue
            if not insertion(hand,skin,dict(c,gap_m=gap),sample)['passed']:continue
            if skin.inspect(hand.bank(gap,'closure',pinch_gap=.001),R,actual,.001)[0]:gaps.append(gap)
        if not gaps:counts['opening_or_approach_rejected']+=1;continue
        counts['geometry_accepted']+=1;record=dict(number=n,candidate=c,projection=projection,q_rad=q.tolist(),valid_openings_m=gaps,sole_alignment=float(-R[:,0]@sole));geometric.append(record)
        pre=k.solve(actual-R[:,0]*.05,R,q)
        if pre['position_error_m']>.0001 or pre['orientation_error_deg']>.5:counts['preapproach_ik_rejected']+=1;continue
        margin=float(np.rad2deg(np.minimum(np.array(pre['q'])-k.lower,k.upper-np.array(pre['q']))).min())
        if margin<5-1e-4:counts['comfort_rejected']+=1;continue
        good=[]
        for gap in gaps:
            if checker.check(q,gap)['passed'] and checker.check(q,.001,True)['passed'] and checker.check(np.array(pre['q']),gap)['passed']:good.append(gap)
        if not good:counts['whole_arm_collision_rejected']+=1;continue
        record.update(preapproach=pre,arm_valid_openings_m=good);accepted.append(record);counts['accepted']+=1;print('GUIDED ACCEPT',n,sample['index'],flush=True)
      print('guided',n,'sample',sample['index'],dict(counts),flush=True)
    (folder/'arm-guided-report.json').write_text(json.dumps(dict(number=n,counts=dict(counts),accepted=accepted,geometry_accepted=geometric,torso_ik_enabled=False,minimum_joint_margin_deg=5),indent=2)+'\n')

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,[3,12]))
