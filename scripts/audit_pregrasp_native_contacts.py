#!/usr/bin/env python3
"""Independent endpoint audit using measured native finger mesh coordinates."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,gzip,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import fcl
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from verify_contact_pregrasp_continuous import make_scene,inventory,write,sha256
from sock_dressing_simulation.continuous_pregrasp import SweptPolicy,eroded_vertices
from sock_dressing_simulation.grasp_geometry import convex_geometry,HANDEDNESS
from sock_dressing_simulation.arm_reaching import ROS_UNITY,shape_vertices
from native_pregrasp_geometry import native_update,native_terminal


def run(job):
    row,out,source=job;folder=out/'drives'/row['id']/'native-route-initial';result=folder/'replay-result.json';snapshot=folder/'native-scene.json.gz'
    if not result.exists()or not snapshot.exists():return None
    r=json.loads(result.read_text())
    if not r.get('completed'):return None
    binding=dict(motion_result_sha256=sha256(result),snapshot_sha256=sha256(snapshot),source_route_sha256=sha256(source/'routes'/f'{row["id"]}.json'),auditor_source_sha256=sha256(Path(__file__)))
    target=out/'native-endpoint-audit'/f'{row["id"]}.json'
    if target.exists():
        previous=json.loads(target.read_text())
        if all(previous.get(k)==v for k,v in binding.items()):return previous
    last=None
    with(folder/'motion-trace.jsonl').open()as stream:
        for line in stream:
            if line.strip():last=line
    measured=json.loads(last)
    if not measured.get('native_links'):return None
    ref=json.load(gzip.open(snapshot,'rt'));refs={v['name']:v for v in ref['links']};scene,route,initial=make_scene(row,source,SweptPolicy());corrections={}
    for side in ('left','right'):
        for suffix in ('left_finger_link','right_finger_link'):
            name=side+'_gripper/'+suffix;frame=refs[name];R=Rotation.from_quat(frame['quaternion']).as_matrix()@ROS_UNITY@HANDEDNESS;t=np.array(frame['position'])
            raw=[s for s in ref['robot_colliders']if s['link']==name];old=scene.base.groups[name];rows=[];cores=[]
            if len(raw)!=len(old):raise ValueError('Native claw part count differs: '+name)
            for current,prior in zip(raw,old):
                local=(shape_vertices(current)-t)@R;g,v,_=convex_geometry(local)
                rows.append(dict(link=name,obj=fcl.CollisionObject(g),vertices=v,moving=True))
                g,_,_=convex_geometry(eroded_vertices(v,scene.policy.permitted_intrusion_m-scene.base.padding.get(name,0.)));cores.append(fcl.CollisionObject(g))
                # AABB-centre change summarizes the removed FK calibration offset.
                corrections[name]=float(np.linalg.norm((v.min(0)+v.max(0)-prior['vertices'].min(0)-prior['vertices'].max(0))/2))
            scene.base.groups[name]=rows;scene.base.group_vertices[name]=np.concatenate([v['vertices']for v in rows]);scene.base.group_starts[name]=np.r_[0,np.cumsum([len(v['vertices'])for v in rows])[:-1]];scene.eroded[name]=cores
    x=np.array(measured['actual_x']);native_update(scene,x,measured['native_links']);contacts=native_terminal(scene,route);failures=[]
    broad=scene.broad_slacks()
    for i in np.flatnonzero(broad<scene.policy.numerical_allowance_m):
        slack,mode=scene.slack(int(i),x,np.zeros(16))
        if slack < -1e-7:failures.append(dict(pair=list(scene.pairs[i]),mode=mode,slack_m=slack))
    result=dict(id=row['id'],motion_completed=True,contact_passed=all(c['relaxed_passed']for c in contacts.values()),terminal_scene_passed=not failures,
                contacts=contacts,terminal_scene_failures=failures,removed_reference_frame_offset_m=corrections,
                **binding,
                method='Invert the measured initial Unity finger transforms to recover native mesh coordinates; transform these meshes with final measured individual finger transforms. This removes fixed-torso FK reference-calibration residuals from the endpoint test.')
    write(out/'native-endpoint-audit'/f'{row["id"]}.json',result);print('NATIVE ENDPOINT AUDIT',row['id'],result['contact_passed'],result['terminal_scene_passed'],flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-pregrasp-verification');p.add_argument('--source',type=Path,default=ROOT/'artifacts/contact-pregrasp-start-search');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        for f in as_completed([pool.submit(run,(r,a.output.resolve(),a.source.resolve()))for r in inventory(ROOT)]):f.result()
