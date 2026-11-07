"""Prompt refinement for lit head images without clipping masks to geometry."""
from __future__ import annotations
import copy
import cv2
import numpy as np
from .head_camera_profile import head_prompts,interior_point
from .perception_audit import projection_precision


def initialize_lit(sam,rgb,refs,minimum_precision=.95):
    prompts=head_prompts(rgb,refs);attempts=[]
    r,g,b=[rgb[:,:,i].astype(float) for i in range(3)]
    blue=(b>r*1.2+5)&(b>g*1.05+3)&(r<150)&refs['sock']
    rows=np.indices(blue.shape)[0];ys=np.where(blue)[0]
    if len(ys):
        for low,high in ((.05,.3),(.7,.95)):
            band=blue&(rows>=np.quantile(ys,low))&(rows<=np.quantile(ys,high))
            if band.any():
                prompts['sock']['points'].insert(0,interior_point(band));prompts['sock']['labels'].insert(0,1)
    for attempt in range(4):
        sam.load_first_frame(rgb[:,:,::-1].copy())
        for obj,kind in ((1,'sock'),(2,'foot')):
            p=prompts[kind]
            _,ids,logits=sam.add_new_points(0,obj,np.array(p['points'],dtype='float32'),np.array(p['labels'],dtype='int32'))
        masks={kind:logits[ids.index(obj),0].float().cpu().numpy()>.2 for obj,kind in ((1,'sock'),(2,'foot'))}
        precision={k:projection_precision(m,refs[k],2)['tolerant'] for k,m in masks.items()}
        attempts.append(dict(attempt=attempt,precision=precision,prompts=copy.deepcopy(prompts)))
        if all(v>=minimum_precision for v in precision.values()):return masks,attempts
        for kind in ('sock','foot'):
            if precision[kind]>=minimum_precision:continue
            support=cv2.dilate(refs[kind].astype('uint8'),np.ones((5,5),dtype='uint8'))>0
            wrong=masks[kind]&~support
            for _ in range(4):
                if wrong.sum()<64:break
                point=interior_point(wrong)
                prompts[kind]['points'].append(point);prompts[kind]['labels'].append(0)
                cv2.circle(wrong.view('uint8'),tuple(point),20,0,-1)
    raise RuntimeError(f'Lit prompt initialization failed: {attempts[-1]}')
