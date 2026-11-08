#!/usr/bin/env python3
"""Recheck every recorded PhysX contact at the geometric numerical tolerance."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from sock_dressing_simulation.contact_pregrasp import opening_edges
from search_centered_reaching import write

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(job):
    path,out,source=job;id=path.parent.parent.name;result=path.parent/'replay-result.json'
    if not result.exists():return
    if path.stat().st_mtime_ns>result.stat().st_mtime_ns:return
    record=json.loads(result.read_text());binding=dict(motion_result_sha256=sha(result),auditor_source_sha256=sha(Path(__file__)))
    target=out/'native-contact-event-audit'/f'{id}.json'
    if target.exists():
        old=json.loads(target.read_text())
        if all(old.get(k)==v for k,v in binding.items()):return
    route=json.loads((source/'routes'/f'{id}.json').read_text());initial=yaml.safe_load((source/'candidate-initial.yaml').read_text());delta=np.array(initial['translation_m'])
    rim=None;scene=None;negative=0;allowed=0;failures=[];largest=0.
    with path.open()as stream:
        for line in stream:
            if '"contacts": []'in line:continue
            sample=json.loads(line)
            for c in sample['contacts']:
                if c['separation_m']>=-1e-7:continue
                a,b=c['a_link'],c['b_link'];moving=any(n.startswith(('left_arm/','right_arm/','left_gripper/','right_gripper/'))for n in(a,b))
                if not moving:continue
                if a and b:
                    if scene is None:
                        from verify_contact_pregrasp_continuous import make_scene
                        from sock_dressing_simulation.continuous_pregrasp import SweptPolicy
                        scene,_,_=make_scene(route['condition'],source,SweptPolicy())
                    if a in scene.base.group and b in scene.base.group and scene.base.excluded(a,b):continue
                negative+=1;largest=max(largest,-c['separation_m']);claw=next((n for n in(a,b)if n.endswith(('/left_finger_link','/right_finger_link'))),None)
                valid=False
                if claw and not(a and b)and any(c[n]=='right_leg_actual_skin_rigid'for n in('a','b')):
                    if rim is None:
                        state=json.loads((ROOT/route['condition']['state']).read_text());rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();rim=np.array(state['particles_world'])[opening_edges(state['triangles'],depth)]
                    side='left'if claw.startswith('left_')else'right';point=np.array(c['point'])-delta;v=rim[:,1]-rim[:,0];u=np.clip(np.einsum('ij,ij->i',point-rim[:,0],v)/np.maximum(np.sum(v*v,axis=1),1e-20),0,1)
                    near=np.linalg.norm(rim[:,0]+u[:,None]*v-point,axis=1).min()<=.03;window=np.linalg.norm(np.array(sample[side+'_tcp_world_m'])-route['terminal'][side]['tcp_world_m'])<=.005
                    valid=bool(near and window and c['separation_m']>=-.0005)
                if valid:allowed+=1
                else:failures.append(dict(time_s=sample['time_s'],stage=sample['stage'],contact=c))
    write(target,dict(id=id,passed=not failures,negative_contact_events=negative,allowed_local_contact_events=allowed,prohibited_events=failures,maximum_negative_separation_m=largest,
        numerical_allowance_m=1e-7,**binding,scope='Every recorded native contact, including the online 10 micrometre noise-filter range, rechecked at 0.1 micrometre. Only claw-to-actual-skin contacts within the agreed 5 mm terminal/30 mm rim window and 0.5 mm intrusion pass. Cloth surface/core geometry remains a separate check.'))
    print('CONTACT EVENT AUDIT',id,not failures,len(failures),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/contact-pregrasp-verification');p.add_argument('--source',type=Path,default=ROOT/'artifacts/contact-pregrasp-start-search');p.add_argument('--workers',type=int,default=4);a=p.parse_args();out=a.output.resolve()
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        for f in as_completed([pool.submit(run,(r,out,a.source.resolve()))for r in(out/'drives').glob('*/native-route-initial/motion-trace.jsonl')]):f.result()
