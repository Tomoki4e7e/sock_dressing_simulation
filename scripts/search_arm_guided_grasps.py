#!/usr/bin/env python3
"""Propose nearby orientations from comfortable bounded arm configurations."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np,yaml
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area
from reevaluate_geometric_closure import insertion

def run(n):
    directory=ROOT/f'artifacts/arm-reaching-safe-start/pose-{n:02d}'
    m=json.loads((directory/'manifest.json').read_text());scene=json.loads((directory/'initial-scene.json').read_text());k=Kinematics(ROOT/m['urdf'],scene)
    prior=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{n:02d}/report.json').read_text());state=json.loads((ROOT/prior['input_state']).read_text());points=np.asarray(state['particles_world']);faces=np.asarray(state['triangles']).reshape(-1,3);rest=np.asarray(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(points,faces);checker=CollisionScene(k,scene,faces)
    old=[r for r in json.loads((ROOT/f'artifacts/geometric-closure/pose-{n:02d}/report.json').read_text())['candidates'] if r['accepted']];unique={json.dumps([r['candidate']['sample_index'],r['candidate']['rotation_matrix']]):r for r in old}
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());h=k.hand;fingers=[v for name,v in h.vertices(.001).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
    lo=k.lower+np.deg2rad(5);hi=k.upper-np.deg2rad(5);counts=Counter();accepted=[];rng=np.random.default_rng(24011+n)
    for r in unique.values():
        c=r['candidate'];p=np.asarray(c['grasp_center_unity_world_m']);desired=np.asarray(c['rotation_matrix']);sample=prior['samples'][c['sample_index']];dist=material_distances(rest,faces,faces[sample['face']],sample['weights'])
        seed0=k.solve(p,desired,k.start)['q']
        for seed in [seed0,k.start]+[rng.uniform(lo,hi) for _ in range(6)]:
            counts['arm_proposals']+=1
            def objective(q):return Rotation.from_matrix(k.pose(q)[1]@desired.T).magnitude()**2+.015*np.sum(((q-k.start)/(hi-lo))**2)
            result=minimize(objective,np.clip(seed,lo,hi),method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=lambda q:k.pose(q)[0]-p)],options=dict(maxiter=180,ftol=1e-10))
            q=result.x;actual,R=k.pose(q)
            if np.linalg.norm(actual-p)>.0001:counts['position_rejected']+=1;continue
            if not skin.inspect(h.bank(.001),R,p,.001)[0]:counts['closed_foot_rejected']+=1;continue
            projection=projected_cloth_area((points-p)@R,faces,fingers,dist,depth,**{name:settings[name] for name in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
            if min(projection['area_m2'],projection['largest_component_area_m2'])<4e-6 or projection['area_fraction']<.05 or min(projection['span_x_m'],projection['span_z_m'])<.0005:counts['area_rejected']+=1;continue
            candidate=dict(c,rotation_matrix=R.tolist());gaps=[]
            for gap in [.0015,.002,.003,.004,.005,.0075,.01,.02,.03,.04,.05]:
                if not skin.inspect(h.bank(gap,'approach',approach=.05),R,p,.001)[0] or cloth.intersections(h.bank(gap,'approach',approach=.05),R,p):continue
                if not insertion(h,skin,dict(candidate,gap_m=gap),sample)['passed']:continue
                if skin.inspect(h.bank(gap,'closure',pinch_gap=.001),R,p,.001)[0]:gaps.append(gap)
            if not gaps:counts['approach_rejected']+=1;continue
            pre=k.solve(p-R[:,0]*.05,R,q)
            if pre['position_error_m']>.0001 or pre['orientation_error_deg']>.5:counts['preapproach_ik_rejected']+=1;continue
            margin=float(np.rad2deg(np.minimum(np.asarray(pre['q'])-k.lower,k.upper-np.asarray(pre['q']))).min())
            if margin<5-1e-4:counts['joint_comfort_rejected']+=1;continue
            valid=[]
            for gap in gaps:
                if checker.check(q,gap)['passed'] and checker.check(q,.001,True)['passed'] and checker.check(np.array(pre['q']),gap)['passed']:valid.append(gap)
            if not valid:counts['arm_collision_rejected']+=1;continue
            record=dict(number=n,source_candidate_index=r['candidate_index'],new_candidate=True,candidate=candidate,projection=projection,q_rad=q.tolist(),preapproach=pre,valid_openings_m=valid,minimum_joint_limit_margin_deg=min(5,margin),orientation_change_deg=float(np.rad2deg(Rotation.from_matrix(R@desired.T).magnitude())))
            accepted.append(record);counts['accepted']+=1;print('ARM GUIDED ACCEPT',n,record,flush=True)
        print('guided',n,c['sample_index'],dict(counts),flush=True)
    out=ROOT/'artifacts/arm-reaching-reselection';(out/f'arm-guided-{n:02d}.json').write_text(json.dumps(dict(number=n,counts=dict(counts),accepted=accepted,original_candidates_unchanged=True),indent=2)+'\n')

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,[3,12]))
