#!/usr/bin/env python3
"""Independently check the reported blocked trials against saved native scenes."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics

def main():
    checks=[]
    for n in [3,12]:
        out=ROOT/f'artifacts/arm-reaching/pose-{n:02d}'
        m=json.loads((out/'manifest.json').read_text());s=json.loads((out/'initial-scene.json').read_text());r=json.loads((out/'report.json').read_text())
        k=Kinematics(ROOT/m['urdf'],s);c=m['candidate']['candidate'];p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix'])
        assert r['ik_seed_count']==96 and r['preapproach_ik_seed_count']==48
        for prefix,target in [('ik_attempts',p),('preapproach_ik_attempts',p-R[:,0]*.05),('position_constrained_orientation',p)]:
            for attempt in r[prefix]:
                q=np.asarray(attempt['q']);assert np.all(q>=k.lower-1e-7) and np.all(q<=k.upper+1e-7)
                actual,orientation=k.pose(q)
                assert abs(np.linalg.norm(actual-target)-attempt['position_error_m'])<1e-8
                angle=np.rad2deg(Rotation.from_matrix(orientation@R.T).magnitude())
                assert abs(angle-attempt['orientation_error_deg'])<1e-6
        found=sum(a['position_error_m']<=.0001 and a['orientation_error_deg']<=.5 for a in r['ik_attempts'])
        assert found==r['ik_solution_count']==0
        raw=json.loads((ROOT/m['candidate']['initial_state_source']).read_text())
        cloth_error=float(np.linalg.norm(np.asarray(s['particles_world'])-raw['initial_particles_world'],axis=1).max());assert cloth_error<=5e-6
        assert not m['robot_commands_executed'] and m['physics_steps_after_initial_scene']==0
        assert not r['reaching_executed'] and not r['closure_executed']
        assert not r['initial_collision']['passed'] and not r['open_start_collision']['passed']
        assert all(a['overlap'] for a in s['self_overlap_audit'])
        checks.append(dict(number=n,passed=True,pose_iterations_recomputed=len(r['ik_attempts'])+len(r['preapproach_ik_attempts'])+len(r['position_constrained_orientation']),
            cloth_error_m=cloth_error,maximum_fk_error_m=k.calibration['maximum_position_error_m'],
            robot_commands_executed=False,reaching_executed=False,closure_executed=False))
    result=dict(passed=True,trials=checks)
    (ROOT/'artifacts/arm-reaching/validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
