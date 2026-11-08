#!/usr/bin/env python3
"""Independently test jaw swaps and reversed tool approaches at promising poses."""
import os
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from shapely.errors import GEOSException
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,geometric_check,URDF
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances


def one(payload):
    row,sources,out=payload;state=load_state(ROOT,row);h=RightHand(URDF);model=type('HandOnly',(),dict(hand=h))();skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());distances={};accepted=[];counts={};seen=set()
    def bank(g,kind='endpoint'):
        if len(h._banks)>160:h._banks={k:v for k,v in h._banks.items() if k[0]==.001}
        return h.bank(float(g),kind,pinch_gap=.001)
    for source in sources:
        r=source['candidate'];p=np.array(r['position']);sample=r['sample'];R0=np.array(r['rotation'])
        for axis,flip in [('original',np.eye(3)),('swap_jaws',np.diag([1,-1,-1])),('reverse_y',np.diag([-1,1,-1])),('reverse_z',np.diag([-1,-1,1]))]:
            R=R0@flip;key=tuple(np.round(np.r_[p,R.ravel()],7))
            if key in seen:continue
            seen.add(key);counts['tested']=counts.get('tested',0)+1
            if not skin.inspect(bank(.001),R,p,.001)[0]:counts['closed_skin']=counts.get('closed_skin',0)+1;continue
            anchor=(sample['face'],tuple(sample['weights']))
            if anchor not in distances:distances[anchor]=material_distances(rest,faces,faces[sample['face']],sample['weights'])
            lo=.001;hi=.05
            if skin.inspect(bank(.05,'closure'),R,p,.001)[0]:lo=.05
            else:
                for _ in range(15):
                    m=(lo+hi)/2
                    if skin.inspect(bank(m,'closure'),R,p,.001)[0]:lo=m
                    else:hi=m
            safe=max(.00100001,lo-2e-7)
            gaps=[]
            for gap in [safe]+list(np.linspace(safe,.00100001,9)[1:]):
                if skin.inspect(bank(gap,'closure'),R,p,.001)[0] and not cloth.intersections(bank(gap),R,p):gaps.append(float(gap));break
            if not gaps:counts['opening_cloth']=counts.get('opening_cloth',0)+1;continue
            try:geometry,reason=geometric_check(model,state,dict(sample,position=p.tolist()),R,skin,cloth,settings,distances[anchor],approach_distance=0,opening_gaps=gaps)
            except (GEOSException,ValueError):geometry=None;reason='numerical_rejected'
            counts[reason]=counts.get(reason,0)+1
            if geometry:
                if any(np.linalg.norm(p-a['position'])<.001 and Rotation.from_matrix(R@np.array(a['rotation']).T).magnitude()<np.deg2rad(5) for a in accepted):continue
                accepted.append(dict(position=p.tolist(),rotation=R.tolist(),sample=sample,geometry=geometry,transform=axis,source=source['source']));print('SYMMETRY GRASP',row['id'],axis,len(accepted),flush=True)
        if len(accepted)>=12:break
    write(out/'geometry'/f'{row["id"]}.json',dict(condition=row,criteria_sha256=sha256(ROOT/'config/geometric_closure.yaml'),accepted=accepted,counts=counts,scope='Only independently rechecked tool geometries; no arm IK or approach path inferred'))
    print('SYMMETRY DONE',row['id'],len(accepted),counts,flush=True);return dict(id=row['id'],accepted=len(accepted),counts=counts)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=ROOT/'artifacts/common-grasp-start-search');ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();root=a.root.resolve();out=root/'symmetry';groups={}
    for folder in [root,root/'transport',root/'transfer-all']:
        for path in sorted((folder/'geometry').glob('*.json')):
            d=json.loads(path.read_text());rows=d['accepted']+d.get('open_promising',[])[:8]+d.get('promising',[])[:12]
            for row in rows:groups.setdefault(d['condition']['id'],[]).append(dict(candidate=row,source=str(path.relative_to(ROOT))))
    completed=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,(r,groups.get(r['id'],[]),out)) for r in inventory(ROOT)]
        for f in as_completed(futures):completed.append(f.result());write(out/'progress.json',dict(completed=len(completed),with_geometry=sum(r['accepted']>0 for r in completed),rows=completed))

if __name__=='__main__':main()
