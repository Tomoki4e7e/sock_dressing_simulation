#!/usr/bin/env python3
"""Validate the IK reselection, SRDF policy, and repaired native start scenes."""
import json,sys,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene

def main():
    out=ROOT/'artifacts/arm-reaching-reselection';summary=json.loads((out/'existing-summary.json').read_text());assert summary['input_configurations']==60 and summary['unique_target_poses']==52
    m=json.loads((ROOT/'artifacts/arm-reaching/pose-03/manifest.json').read_text());k=Kinematics(ROOT/m['urdf'],json.loads((ROOT/'artifacts/arm-reaching/pose-03/initial-scene.json').read_text()))
    checked=0
    for group in summary['results']:
        key=group['configurations'][0];r=json.loads((ROOT/f"artifacts/geometric-closure/pose-{key['number']:02d}/report.json").read_text())['candidates'][key['candidate_index']];c=r['candidate'];p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix'])
        assert len(group['attempts'])==48
        solutions=0
        for a in group['attempts']:
            q=np.asarray(a['q']);assert np.all(q>=k.lower-1e-7) and np.all(q<=k.upper+1e-7)
            pos,rot=k.pose(q);error=np.linalg.norm(pos-p);angle=np.rad2deg(Rotation.from_matrix(rot@R.T).magnitude())
            assert abs(error-a['position_error_m'])<1e-8 and abs(angle-a['orientation_error_deg'])<1e-6
            solutions+=int(error<=.0001 and angle<=.5);checked+=1
        assert solutions==0 and group['ik_solution_count']==0
    policy=json.loads((out/'model-self-collision-policy.json').read_text());srdf=Path(policy['source']);assert hashlib.sha256(srdf.read_bytes()).hexdigest()==policy['sha256']
    assert any({r.get('link1'),r.get('link2')}==set(policy['excluded_pair']) and r.get('reason')=='Default' for r in ET.parse(srdf).getroot().findall('disable_collisions'))
    starts=[]
    for n in [3,12]:
        folder=ROOT/f'artifacts/arm-reaching-safe-start/pose-{n:02d}';manifest=json.loads((folder/'manifest.json').read_text());scene=json.loads((folder/'initial-scene.json').read_text());model=Kinematics(ROOT/manifest['urdf'],scene)
        state=json.loads((ROOT/manifest['candidate']['initial_state_source']).read_text());checker=CollisionScene(model,scene,state['triangles']);gap=model.q0['right_gripper/finger_joint']+model.hand.gap_offset_m;collision=checker.check(model.start,gap)
        assert collision['passed'] and abs(gap-.05)<1e-5
        assert checker.model_exclusions=={frozenset(policy['excluded_pair'])}
        for a in scene['self_overlap_audit']:
            if {a['a'],a['b']}!=set(policy['excluded_pair']):assert not a['overlap']
        margin=float(np.rad2deg(np.minimum(model.start-model.lower,model.upper-model.start)).min());assert margin>29.99
        cloth_error=float(np.linalg.norm(np.array(scene['particles_world'])-state['initial_particles_world'],axis=1).max());assert cloth_error<=5e-6
        starts.append(dict(number=n,passed=True,gap_m=gap,right_joint_angles_deg=np.rad2deg(model.start).tolist(),minimum_joint_limit_margin_deg=margin,collision=collision,native_audit=scene['self_overlap_audit'],cloth_error_m=cloth_error,maximum_fk_error_m=model.calibration['maximum_position_error_m']))
    (ROOT/'artifacts/arm-reaching-safe-start/validation.json').write_text(json.dumps(dict(passed=True,trials=starts),indent=2)+'\n')
    result=dict(passed=True,existing_configurations=60,unique_target_poses=52,ik_trials_recomputed=checked,endpoint_ik_accepted=0,native_start_scenes_checked=2,srdf_exclusion_verified=True)
    (out/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
