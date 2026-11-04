#!/usr/bin/env python3
"""Maximize wrist-limit margin for the feasible expanded-cuff IK target."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sock_dressing_simulation.arm_reaching import Kinematics,ARM

def main():
    rows=[]
    for path in (ROOT/'artifacts/bulged-cuff').glob('pose-*/search/geometry-*.json'):
        record=json.loads(path.read_text())
        if not record['endpoint_ik_solution_count']:continue
        folder=path.parent.parent;audit=json.loads((folder/'report.json').read_text());native=json.loads((folder/'native-scene.json').read_text());scene=json.loads((ROOT/native.pop('shared_robot_scene')).read_text());scene.update(native);k=Kinematics(ROOT/audit['source_urdf'],scene)
        c=record['candidate'];p=np.array(c['grasp_center_unity_world_m']);R=np.array(c['rotation_matrix']);valid=[a for a in record['ik_attempts'] if a['position_error_m']<=1e-4 and a['orientation_error_deg']<=.5];seeds=[]
        for a in valid:
            q=np.array(a['q'])
            if all(np.linalg.norm(q-s)>.03 for s in seeds):seeds.append(q)
        attempts=[]
        for q in seeds:
            margin=np.minimum(q-k.lower,k.upper-q).min()
            def inequalities(x):
                pos,rot=k.pose(x[:7]);angle=Rotation.from_matrix(rot@R.T).magnitude()
                return np.r_[x[:7]-k.lower-x[7],k.upper-x[:7]-x[7],1-np.sum((pos-p)**2)/1e-8,1-(angle/np.deg2rad(.5))**2]
            result=minimize(lambda x:-x[7],np.r_[q,max(0,margin-1e-8)],method='SLSQP',bounds=list(zip(k.lower,k.upper))+[(0,float(np.min(k.upper-k.lower)/2))],constraints=[dict(type='ineq',fun=inequalities)],options=dict(maxiter=400,ftol=1e-11))
            q=result.x[:7];pos,rot=k.pose(q);error=float(np.linalg.norm(pos-p));angle=float(np.rad2deg(Rotation.from_matrix(rot@R.T).magnitude()));margins=np.rad2deg(np.minimum(q-k.lower,k.upper-q))
            attempts.append(dict(q_rad=q.tolist(),minimum_joint_margin_deg=float(margins.min()),limiting_joint=ARM[int(np.argmin(margins))],position_error_m=error,orientation_error_deg=angle,valid_pose=bool(error<=.00010001 and angle<=.50001),optimizer_success=bool(result.success)))
        feasible=[a for a in attempts if a['valid_pose'] and a['optimizer_success']]
        best=max(feasible,key=lambda a:a['minimum_joint_margin_deg']) if feasible else None
        rows.append(dict(number=record['number'],geometry_index=record['index'],attempts=attempts,best=best,comfortable_solution_found=bool(best and best['minimum_joint_margin_deg']>=5),torso_ik_enabled=False,scope='Diagnostic optimization within the original 0.1 mm / 0.5 degree pose tolerances; no motion or changed joint limits'))
        print(record['number'],record['index'],best,flush=True)
    (ROOT/'artifacts/bulged-cuff/margin-optimization.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':main()
