#!/usr/bin/env python3
"""Independently verify the radial deformation, Native match and search records."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import projected_cloth_area,material_distances
from reevaluate_geometric_closure import insertion
import yaml

def main():
    root=ROOT/'artifacts/bulged-cuff';summary=json.loads((root/'summary.json').read_text());native=json.loads((root/'native/validation.json').read_text());assert len(summary['poses'])==len(native['poses'])==32
    shared=json.loads((root/'native/shared-robot-scene.json').read_text());settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());checks=[];totals=Counter();ik_trials=0;geometric_records=0;old60=[]
    for pose in summary['poses']:
        n=pose['number'];folder=root/f'pose-{n:02d}';state=json.loads((folder/'state.json').read_text());before=np.array(json.loads((folder/'before-refined.json').read_text())['particles_world']);profile=json.loads((folder/'profile.json').read_text());points=np.array(state['particles_world']);rest=np.array(state['source']['rest_positions']);faces=np.array(state['triangles']).reshape(-1,3);edges=np.array(state['edges']).reshape(-1,2);z=np.array(profile['depth_m']);c=np.array(profile['reference_center_world']);axis=np.array(profile['reference_axis_world'])
        def radial(v):v=v-c;return np.linalg.norm(v-axis*np.sum(v*axis,axis=1)[:,None],axis=1)
        increment=radial(points)-radial(before);expected=np.zeros(len(z));inside=(z>1e-10)&(z<.03-1e-10);expected[inside]=.005*np.sin(np.pi*z[inside]/.03)**2
        assert np.max(abs(increment-expected))<1e-10 and np.array_equal(points[~inside],before[~inside])
        peak=np.isclose(z,.015,atol=1e-9,rtol=0);assert np.allclose(increment[peak],.005,atol=1e-10,rtol=0)
        original=json.loads((ROOT/state['bulge']['original_state']).read_text());v0=np.array(original['particles_world']);f0=np.array(original['triangles']).reshape(-1,3)
        area=lambda v,f:np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1).sum()/2
        area_error=abs(area(before,faces)-area(v0,f0));assert area_error<1e-9
        ring_edges=edges[np.all(peak[edges],axis=1)];weights=np.linspace(0,1,51)[None,:,None];after_ring=points[ring_edges[:,0],None]*(1-weights)+points[ring_edges[:,1],None]*weights;before_ring=before[ring_edges[:,0],None]*(1-weights)+before[ring_edges[:,1],None]*weights;center=c[np.flatnonzero(peak)[0]];direction=axis[np.flatnonzero(peak)[0]]
        def ring_radius(v):v=v-center;return np.linalg.norm(v-direction*np.sum(v*direction,axis=-1)[...,None],axis=-1)
        mesh_increment=ring_radius(after_ring)-ring_radius(before_ring)
        nv=json.loads((folder/'native-validation.json').read_text());assert nv['passed'] and nv['cloth_error_m']<5e-6 and nv['skin_error_m']<5e-6 and not nv['torso_ik_enabled']
        scene=dict(shared);scene.update(json.loads((folder/'native-scene.json').read_text()));k=Kinematics(ROOT/pose['source_urdf'],scene);checker=CollisionScene(k,scene,faces);start=checker.check(k.start,.05);assert start['passed']
        search=json.loads((folder/'search/report.json').read_text());totals.update(search['counts']);old60.extend(search['old60_parameter_recheck']);assert not search['settings']['torso_ik_enabled'] and not search['accepted']
        assert search['counts']['orientations_tested']==59136
        samples=json.loads((folder/'samples.json').read_text());skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(points,faces);fingers=[v for name,v in k.hand.vertices(.001).items() if name.endswith('/left_finger_link') or name.endswith('/right_finger_link')]
        for path in (folder/'search').glob('geometry-*.json'):
            record=json.loads(path.read_text());candidate=record['candidate'];p=np.array(candidate['grasp_center_unity_world_m']);R=np.array(candidate['rotation_matrix']);sample=samples[candidate['sample_index']];dist=material_distances(rest,faces,faces[sample['face']],sample['weights']);pr=projected_cloth_area((points-p)@R,faces,fingers,dist,z,**{name:settings[name] for name in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
            assert abs(pr['area_m2']-record['projection']['area_m2'])<1e-12 and min(pr['area_m2'],pr['largest_component_area_m2'])>=4e-6
            assert skin.inspect(k.hand.bank(.001),R,p,.001)[0]
            for g in record['valid_openings']:
                gap=g['gap_m'];bank=k.hand.bank(gap,'approach',approach=.05);assert skin.inspect(bank,R,p,.001)[0] and not cloth.intersections(bank,R,p)
                assert insertion(k.hand,skin,dict(candidate,gap_m=gap),sample)['passed'] and skin.inspect(k.hand.bank(gap,'closure',pinch_gap=.001),R,p,.001)[0]
            valid=0
            for attempt in record['ik_attempts']:
                q=np.array(attempt['q']);assert np.all(q>=k.lower-1e-7) and np.all(q<=k.upper+1e-7)
                actual,rot=k.pose(q);error=np.linalg.norm(actual-p);angle=np.rad2deg(Rotation.from_matrix(rot@R.T).magnitude());assert abs(error-attempt['position_error_m'])<1e-8 and abs(angle-attempt['orientation_error_deg'])<1e-6
                valid+=int(error<=.0001 and angle<=.5);ik_trials+=1
            assert bool(valid)==bool(record['endpoint_ik_solution_count']) and len(record['ik_attempts'])==48;geometric_records+=1
        checks.append(dict(number=n,passed=True,peak_vertex_radius_increment_mm_min=float(increment[peak].min()*1000),peak_vertex_radius_increment_mm_max=float(increment[peak].max()*1000),peak_mesh_edge_radius_increment_mm_min=float(mesh_increment.min()*1000),peak_mesh_edge_radius_increment_mm_max=float(mesh_increment.max()*1000),refinement_surface_area_error_m2=area_error,native_cloth_error_m=nv['cloth_error_m'],start_collision=start))
    assert len(old60)==60 and all(r['evaluated'] for r in old60);assert totals['orientations_tested']==1892352 and geometric_records==32
    report=dict(passed=True,poses=checks,geometry_pose_records_recomputed=geometric_records,ik_trials_recomputed=ik_trials,old60_rechecked=len(old60),all_torso_ik_disabled=True,counts=dict(totals))
    (root/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='poses'})
if __name__=='__main__':main()
