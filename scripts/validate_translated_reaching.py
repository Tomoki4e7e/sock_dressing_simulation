#!/usr/bin/env python3
"""Independent FK, bounds, scene invariance and full robot recheck of saved trials."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
import fcl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from search_translated_reaching import load,translated,margin,good,OUT
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene,shape_vertices

def static_check(checker):
    issues=[]
    for row in checker.shapes:
        if row['moving']:continue
        for name,obj in checker.obstacles:
            if 'Ground' in name:continue
            d=float(fcl.distance(row['obj'],obj,fcl.DistanceRequest(),fcl.DistanceResult()))
            required=0 if name=='frozen_cloth' else .005
            if d<=0 or d<required:issues.append(dict(pair=[row['link'],name],distance_m=max(0,d)))
    return issues

def main():
    summary=json.loads((OUT/'summary.json').read_text());k=Kinematics(ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf',load(39));bound=np.array(summary['translation_bounds_m']);counts=Counter();pairs=Counter();checks=[]
    for i,row in enumerate(summary['results']):
        record=json.loads((OUT/f'target-{i:03d}.json').read_text());assert record['source']==row['source'];source,sep,idx=row['source'].partition(':');raw=json.loads((ROOT/source).read_text())
        if sep:
            r=next(r for r in raw['candidates'] if r['candidate_index']==int(idx));c=r['candidate'];state=json.loads((ROOT/r['initial_state_source']).read_text());gaps=[r['opening_gap_m']]
        else:c=raw['candidate'];state=json.loads((ROOT/raw['source_state']).read_text());gaps=[r['gap_m'] for r in raw['valid_openings']]
        if 'particles_world' not in state:state['particles_world']=state['initial_particles_world']
        p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);trials=0
        for a in record['attempts']:
            q=np.array(a['q']);d=np.array(a['translation_m']);assert (abs(d)<=bound+1e-8).all();assert margin(k,q)>=5-1e-7
            T=k.fk(q,only=['right_gripper/gripper_base'])['right_gripper/gripper_base'];actual=k.A@(T[:3,3]+T[:3,:3]@k.hand.grasp_center_urdf)+k.b
            rot=k.A@T[:3,:3]@np.diag([1,1,-1]);assert abs(np.linalg.norm(actual-p-d)-a['position_error_m'])<1e-8;assert abs(np.rad2deg(Rotation.from_matrix(rot@R.T).magnitude())-a['orientation_error_deg'])<1e-6;counts['trials']+=1
            if not good(a):continue
            counts['endpoint_ik_pass']+=1;moved=translated(load(row['number']),d,state['particles_world'])
            assert np.allclose(np.array(moved['particles_world'])-(p+d),np.array(state['particles_world'])-p,atol=1e-12)
            checker=CollisionScene(k,moved,state['triangles']);collision=checker.check(q,gaps[0]);issues=static_check(checker);counts['fixed_robot_collision_trials']+=bool(issues)
            for issue in collision['issues']:pairs[tuple(issue['pair'])]+=1
            counts['endpoint_collision_clear']+=collision['passed'] and not issues;trials+=1
        checks.append(dict(source=row['source'],ik_trials_recomputed=len(record['attempts']),endpoint_collisions_recomputed=trials))
    assert counts['trials']==summary['aggregate']['trials']
    assert counts['endpoint_ik_pass']==summary['aggregate']['endpoint_ik_trials']
    assert counts['endpoint_collision_clear']==0
    report=dict(passed=True,counts=dict(counts),collision_pairs=[dict(pair=list(p),count=n) for p,n in pairs.most_common()],targets=checks,scope='FK and target residual independent recomputation; endpoint collisions including fixed robot versus translated environment; saved snapshots, not live Unity execution')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2));print(dict(counts))
if __name__=='__main__':main()
