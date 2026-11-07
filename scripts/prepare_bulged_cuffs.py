#!/usr/bin/env python3
"""Create and audit all 32 +5 mm cuff-bulge initial states."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.cuff_bulge import refine_sections,radial_bulge
from sock_dressing_simulation.held_ankle_poses import audit_actual_skin
from sock_dressing_simulation.grasp_geometry import candidate_samples

def write_obj(path,vertices,faces,native_import=False):
    vertices=np.asarray(vertices).copy();faces=np.asarray(faces).copy()
    if native_import:
        # UnityMeshImporter reflects X and reverses winding on OBJ import.
        vertices[:,0]*=-1;faces=faces[:,::-1]
    with path.open('w') as f:
        for p in vertices:f.write('v '+' '.join(f'{v:.10f}' for v in p)+'\n')
        for face in faces:f.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')

def prepare(number):
    prior=json.loads((ROOT/f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text());source_path=ROOT/prior['input_state'];state=json.loads(source_path.read_text())
    setting=yaml.safe_load((ROOT/'config/cuff_bulge.yaml').read_text())
    if setting['profile']!='sine_squared' or abs(setting['center_m']-setting['band_width_m']/2)>1e-12:raise ValueError('Sine-squared profile requires the peak at the band centre')
    original_rest=np.asarray(state['source']['rest_positions']);original=np.asarray(state['particles_world']);rest,before,faces=refine_sections(original_rest,original,state['triangles'],depths=setting['refinement_depths_m']);after,profile=radial_bulge(original_rest,original,rest,before,width=setting['band_width_m'],peak=setting['peak_radius_increment_m'])
    edges=np.unique(np.sort(faces[:,[0,1,1,2,2,0]].reshape(-1,2),axis=1),axis=0)
    body,surface=audit_actual_skin(after,state['source']['skin_surface'],edges,faces)
    z=np.asarray(profile['depth_m']);increment=np.asarray(profile['radius_increment_m']);at_peak=np.isclose(z,setting['center_m'],atol=1e-9,rtol=0);outside=(z<=1e-10)|(z>=setting['band_width_m']-1e-10)
    checks=dict(peak_section_present=bool(at_peak.any()),peak_radius_exact=bool(np.allclose(increment[at_peak],setting['peak_radius_increment_m'],atol=1e-10,rtol=0)),outside_band_unchanged=bool(np.array_equal(after[outside],before[outside])),no_skin_intersection=bool(body['geometry_ok']),no_self_intersection=bool(surface['surface_ok']))
    out=ROOT/f'artifacts/bulged-cuff/pose-{number:02d}';out.mkdir(parents=True,exist_ok=True)
    state.update(mode=f"frozen_cuff_bulge_{setting['peak_radius_increment_m']*1000:g}mm_center{setting['center_m']*1000:g}mm",particles_world=after.tolist(),triangles=faces.ravel().tolist(),edges=edges.ravel().tolist(),rest_edge_lengths=np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1).tolist())
    state['source']['rest_positions']=rest.tolist();state['bulge']=dict(width_m=setting['band_width_m'],center_m=setting['center_m'],peak_radius_increment_m=setting['peak_radius_increment_m'],profile=f"{setting['peak_radius_increment_m']*1000:g}mm * sin(pi * material_depth / {setting['band_width_m']*1000:g}mm)^2 within band; zero outside",original_state=str(source_path.relative_to(ROOT)),original_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),original_vertex_count=len(original),fixed_geometry_only=True)
    (out/'state.json').write_text(json.dumps(state)+'\n');(out/'profile.json').write_text(json.dumps(profile)+'\n');(out/'before-refined.json').write_text(json.dumps(dict(particles_world=before.tolist()))+'\n');write_obj(out/'rest.obj',rest,faces,native_import=True);write_obj(out/'deformed-world.obj',after,faces)
    samples=candidate_samples(state);(out/'samples.json').write_text(json.dumps(samples)+'\n')
    report=dict(number=number,pitch_deg=prior['pitch_deg'],roll_deg=prior['roll_deg'],passed=all(checks.values()),checks=checks,vertices=len(rest),faces=len(faces),peak_section_vertices=int(at_peak.sum()),maximum_move_m=float(increment.max()),body=body,surface=surface,source_urdf=prior['settings']['urdf'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(number,checks,flush=True);return report

if __name__=='__main__':
    numbers=[r['number'] for r in json.loads((ROOT/'artifacts/geometric-closure/summary.json').read_text())['poses']]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(prepare,numbers))
    (ROOT/'artifacts/bulged-cuff/summary.json').write_text(json.dumps(dict(poses=rows,passed=all(r['passed'] for r in rows)),indent=2)+'\n')
