#!/usr/bin/env python3
"""Retain full-body and preapproach diagnostics for any bounded IK survivors."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene

def main():
    survivors=[]
    for path in (ROOT/'artifacts/bulged-cuff').glob('pose-*/search/geometry-*.json'):
        record=json.loads(path.read_text())
        if record['endpoint_ik_solution_count']:survivors.append((path,record))
    if len(survivors)!=1:raise RuntimeError('The diagnostic report must be reviewed for a different survivor set')
    path,d=survivors[0];folder=path.parent.parent;metadata=json.loads((folder/'report.json').read_text());state=json.loads((folder/'state.json').read_text());native=json.loads((folder/'native-scene.json').read_text());scene=json.loads((ROOT/native.pop('shared_robot_scene')).read_text());scene.update(native);k=Kinematics(ROOT/metadata['source_urdf'],scene);checker=CollisionScene(k,scene,state['triangles'])
    valid=[r for r in d['ik_attempts'] if r['position_error_m']<=1e-4 and r['orientation_error_deg']<=.5];results=[];p=np.array(d['candidate']['grasp_center_unity_world_m']);R=np.array(d['candidate']['rotation_matrix'])
    for v in valid:
        q=np.array(v['q']);results.append(dict(ik=v,open_collision=checker.check(q,.03),closed_collision=checker.check(q,.001,True),preapproach_ik=k.solve(p-R[:,0]*.05,R,q),minimum_joint_margin_deg=float(np.rad2deg(np.minimum(q-k.lower,k.upper-q)).min())))
    out=ROOT/'artifacts/bulged-cuff';(out/'closest-ik-diagnostics.json').write_text(json.dumps(dict(number=d['number'],geometry_index=d['index'],diagnostic_only=True,executed=False,results=results),indent=2)+'\n')
    pre=p-R[:,0]*.05;solutions,attempts=k.multistart(pre,R,count=48)
    (out/'closest-preapproach-ik.json').write_text(json.dumps(dict(number=d['number'],geometry_index=d['index'],target_world_m=pre.tolist(),solution_count=len(solutions),solutions=solutions,attempts=attempts,torso_ik_enabled=False),indent=2)+'\n')
    print(dict(number=d['number'],valid_ik_trials=len(valid),open_collision_free=sum(r['open_collision']['passed'] for r in results),preapproach_solutions=len(solutions)))
if __name__=='__main__':main()
