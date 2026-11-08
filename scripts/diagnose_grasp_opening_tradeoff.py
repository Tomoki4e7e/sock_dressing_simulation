#!/usr/bin/env python3
"""Separate the skin-safe and cloth-clear opening ranges at reviewed poses.

These fixed-pose width sweeps diagnose individual search candidates; they do
not establish that no other hand pose exists.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/common-grasp-tradeoff-mpl')
import json,sys
from pathlib import Path
import numpy as np,fcl
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import write,URDF
from sock_dressing_simulation.common_opening_approach import inventory,load_state,sha256
from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen


def main():
    out=ROOT/'artifacts/common-grasp-start-search';probe=json.loads((out/'continuous-opening-probe.json').read_text());cases=[dict(r,id=f'B-{r["number"]:02d}') for r in probe['rows'] if r['seed']==0]
    positive=json.loads((out/'transport/geometry/A-10.json').read_text())['accepted'][0];cases.insert(0,dict(positive,id='A-10'))
    rows=[]
    for record in cases:
        condition=next(r for r in inventory(ROOT) if r['id']==record['id']);state=load_state(ROOT,condition);hand=RightHand(URDF);skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles']);R=np.array(record['rotation']);p=np.array(record['position']);base=hand.bank(.001)
        lo=.001;hi=.05
        for _ in range(25):
            g=(lo+hi)/2
            if skin.inspect(hand.bank(g,'closure',pinch_gap=.001),R,p,.001)[0]:lo=g
            else:hi=g
        widths=np.unique(np.r_[np.linspace(.00101,.005,81),np.linspace(.005,.05,91),max(.00100001,lo-1e-7)])
        checks=[]
        for g in widths:
            hits=[]
            for part in base:
                coefficient=0 if part.label.endswith('/gripper_base') else (-.5 if '/left_' in part.label else .5)
                part.obj.setTransform(fcl.Transform(R,p+R@np.array([0,coefficient*(g-.001),0])))
                if fcl.collide(part.obj,cloth.obj,fcl.CollisionRequest(),fcl.CollisionResult()):hits.append(part.label)
            checks.append(dict(gap_m=float(g),closure_skin_safe=bool(g<=lo),cloth_clear=not hits,cloth_intersections=hits))
        free=[r['gap_m'] for r in checks if r['cloth_clear']];feasible=[r['gap_m'] for r in checks if r['cloth_clear'] and r['closure_skin_safe']]
        rows.append(dict(condition=condition,pose=record,maximum_skin_safe_gap_m=lo,minimum_sampled_cloth_clear_gap_m=min(free) if free else None,sampled_feasible_openings_m=feasible,checks=checks));print('WIDTH TRADEOFF',record['id'],'skin max mm',lo*1000,'cloth sampled min mm',min(free)*1000 if free else None,'feasible',len(feasible),flush=True)
    write(out/'opening-tradeoff.json',dict(rows=rows,criteria_sha256=sha256(ROOT/'config/geometric_closure.yaml'),scope='Fixed-pose diagnostics only: exact full-hand cloth collision and convex skin closure sweep; no global impossibility claim'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(rows),1,figsize=(10,2.5*len(rows)),constrained_layout=True)
    for ax,row in zip(np.atleast_1d(axs),rows):
        g=np.array([r['gap_m'] for r in row['checks']])*1000;skin=np.array([r['closure_skin_safe'] for r in row['checks']]);cloth=np.array([r['cloth_clear'] for r in row['checks']]);ax.step(g,skin.astype(float),where='mid',label='Skin-safe closure (1 mm margin)');ax.step(g,cloth.astype(float)+1.3,where='mid',label='Cloth-clear open hand');ax.fill_between(g,-.2,2.5,where=skin&cloth,color='green',alpha=.15,label='Both conditions');ax.set(title=row['condition']['id'],xlabel='Open jaw gap (mm)',xlim=(1,50),yticks=[0,1,1.3,2.3],yticklabels=['Fail','Pass','Fail','Pass']);ax.grid(alpha=.2);ax.legend(loc='center right',fontsize=8)
    fig.savefig(out/'opening-tradeoff.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
