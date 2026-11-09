#!/usr/bin/env python3
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse,json,sys,time,hashlib
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_fixed_torso_starts import context
from search_centered_reaching import REGISTERED_MANIFEST,URDF,write
from search_contact_pregrasp import contact_scene
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from sock_dressing_simulation.continuous_pregrasp import SweptScene,SweptPolicy,trajectory_intervals,certify_interval


def make_scene(row,source,policy):
    initial=yaml.safe_load((source/'candidate-initial.yaml').read_text());route=json.loads((source/'routes'/f'{row["id"]}.json').read_text())
    if not route['passed']or route['condition']!=row or route['geometry_manifest_sha256']!=sha256(REGISTERED_MANIFEST):raise ValueError('Stale or unqualified route')
    for side in ('left','right'):np.testing.assert_array_equal(route['initial_'+side+'_q_rad'],initial[side+'_q_rad'])
    np.testing.assert_array_equal(route['translation_m'],initial['translation_m'])
    state=load_state(ROOT,row);k,base,*_,scene=context(row['number'],state,REGISTERED_MANIFEST,left_foot_floor_allowance=initial['left_foot_floor_allowance_m'],human_collider_manifest=initial.get('human_collider_manifest'))
    contact_scene(base,{s:route['final_contact_checks'][s]['foot_finger']for s in ('left','right')})
    base.swept_skin_sources={'human_full_skin':(np.array(scene['human_skin']['vertices']),np.array(scene['human_skin']['triangles']).reshape(-1,3))}
    for obstacle in scene['obstacles']:
        if 'skin'in obstacle['name']and obstacle['kind']=='MeshCollider'and not obstacle['convex']:
            base.swept_skin_sources[obstacle['name']]=(np.array(obstacle['vertices_world']),np.array(obstacle['triangles']).reshape(-1,3))
    swept=SweptScene(k,base,state,route,initial['translation_m'],policy)
    digest=hashlib.sha256()
    def add(value):
        value=np.ascontiguousarray(value);digest.update(str(value.shape).encode());digest.update(value.tobytes())
    add(k.A);add(k.b);digest.update(json.dumps(k.q0,sort_keys=True).encode());digest.update(json.dumps(base.pairs).encode())
    for name in sorted(base.groups):digest.update(name.encode());add(base.group_vertices[name])
    for name in sorted(base.obstacle_points):digest.update(name.encode());add(base.obstacle_points[name])
    for name,(v,f)in sorted(base.swept_skin_sources.items()):digest.update(name.encode());add(v);add(f)
    swept.geometry_digest=digest.hexdigest()
    return swept,route,initial


def verify_one(job):
    row,source,out,policy,resume=job;path=out/'continuous'/f'{row["id"]}.json';routepath=source/'routes'/path.name
    began=time.monotonic();scene,route,initial=make_scene(row,source,policy)
    binding=dict(route_sha256=sha256(routepath),state_sha256=row['state_sha256'],initial_sha256=sha256(source/'candidate-initial.yaml'),
                 urdf_sha256=sha256(URDF),geometry_manifest_sha256=sha256(REGISTERED_MANIFEST),geometry_digest=scene.geometry_digest,policy=asdict(policy),
                 verifier_source_sha256=sha256(ROOT/'sock_dressing_simulation/continuous_pregrasp.py'))
    if resume and path.exists():
        old=json.loads(path.read_text())
        if old.get('binding')==binding:return old
    intervals=trajectory_intervals(route,initial);results=[];deadline=began+policy.condition_timeout_s
    floor=min(y+initial['translation_m'][1]-scene.base.floor_top for y in scene.base.floor_minima.values())
    if floor < -1e-9:raise ValueError('Static scene floor policy failed')
    for i,interval in enumerate(intervals):
        checked=certify_interval(scene,interval['start'],interval['end'],deadline=deadline)
        checked.update(index=i,stage=interval['stage'],side=interval['side'],duration_s=interval['duration_s']);results.append(checked)
        if checked['status']!='certified':break
    status=results[-1]['status']if results else'unresolved'
    result=dict(id=row['id'],condition=row,status=status,passed=status=='certified'and len(results)==len(intervals),
                binding=binding,intervals_checked=len(results),intervals_total=len(intervals),intervals=results,
                midpoint_configurations=scene.updates,fcl_queries=scene.queries,elapsed_s=time.monotonic()-began,
                scope='Exact commanded joint polylines with monotone quintic timing; frozen cloth/human and current registered robot surfaces. Local intrusion is bounded by absence of surface intersection with the claw eroded inward by the permitted depth. Unresolved is not a pass.')
    write(path,result);print('CONTINUOUS',row['id'],status,result['intervals_checked'],len(intervals),round(result['elapsed_s'],1),flush=True);return result


def report(out,source,policy):
    rows=[json.loads(p.read_text())for p in sorted((out/'continuous').glob('*.json'))]
    summary=dict(expected=70,completed=len(rows),certified=sum(r['passed']for r in rows),failed=sum(r['status']=='failed'for r in rows),unresolved=sum(r['status']=='unresolved'for r in rows),
                 all_70_certified=len(rows)==70 and all(r['passed']for r in rows),source=str(source),policy=asdict(policy),rows=[{k:r[k]for k in ('id','status','passed','elapsed_s','intervals_checked','intervals_total','fcl_queries')}for r in rows])
    write(out/'continuous-summary.json',summary);print('SUMMARY',{k:v for k,v in summary.items()if k!='rows'},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'artifacts/contact-pregrasp-start-search');p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-pregrasp-verification');p.add_argument('--ids',nargs='*');p.add_argument('--workers',type=int,default=6);p.add_argument('--seconds',type=float,default=600.);p.add_argument('--depth',type=int,default=24);p.add_argument('--resume',action='store_true');p.add_argument('--report-only',action='store_true');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);source=a.source.resolve();policy=SweptPolicy(maximum_depth=a.depth,condition_timeout_s=a.seconds)
    write(out/'policy.json',dict(asdict(policy),cloth_frozen=True,joint_drives_only=True,final_closure_performed=False,skin_region='Triangles wholly inside one 30 mm rim-edge capsule and belonging to the saved target-leg surface; conservative whole-triangle subset',intrusion_metric='Maximum depth of the eligible cloth/skin surface inside each convex claw, tested with an inward-offset convex core; remote triangles retain original margins'))
    if not a.report_only:
        rows=[r for r in inventory(ROOT)if not a.ids or r['id']in a.ids]
        with ProcessPoolExecutor(max_workers=a.workers)as pool:
            for f in as_completed([pool.submit(verify_one,(r,source,out,policy,a.resume))for r in rows]):f.result()
    report(out,source,policy)

if __name__=='__main__':main()
