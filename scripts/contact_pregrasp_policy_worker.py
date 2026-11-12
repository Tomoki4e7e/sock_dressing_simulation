#!/usr/bin/env python3
"""CUDA-only live pregrasp policy with the frozen training normalization."""
import argparse,contextlib,json,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import train_contact_pregrasp as p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--episode',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
        config=json.loads((a.run/'configuration.json').read_text());stats=json.loads((a.run/'normalization.json').read_text());m=config['model']
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        model=p.SAMDAMSARNN(rec_dim=m['rec_dim'],k_dim=m['k_dim'],joint_dim=36,temperature=m['temperature'],heatmap_size=m['heatmap_size'],im_size=m['im_size']).cuda().eval();model.load_state_dict(saved['model_state_dict'])
        sam=p.load_sam(p.SAM,512,'cuda');depth=p.load_depth(p.DEPTH,'cuda')
        meta=json.loads((a.episode/'metadata.json').read_text());s=json.loads((p.ROOT/meta['condition']['state']).read_text());delta=np.array(meta['translation_m']);cam=meta['camera_transform'];skin=s['source']['skin_surface']
        refs=dict(sock=p.projected_mesh(np.array(s['particles_world'])+delta,s['triangles'],cam,meta['camera_fov'],(640,480)),foot=p.projected_mesh(np.array(skin['vertices'])+delta,skin['triangles'],cam,meta['camera_fov'],(640,480)))
        low=np.array(stats['joint_min']);high=np.array(stats['joint_max']);span=high-low;constant=span<=1e-7
    state=None;initialized=False
    print(json.dumps(dict(ready=True,epoch=saved['epoch'],device='cuda',gpu=torch.cuda.get_device_name(),model_devices=dict(samdam=str(next(model.parameters()).device),sam2=str(next(sam.parameters()).device),depth=str(next(depth.parameters()).device)))),flush=True)
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('quit'):break
        with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
            rgb=np.array(Image.open(r['rgb']).convert('RGB'));assert rgb.shape==(480,640,3)
            if not initialized:masks,attempts=p.initialize_lit(sam,rgb,refs);initialized=True
            else:
                ids,logits=sam.track(rgb[:,:,::-1].copy());masks={kind:logits[ids.index(obj),0].float().cpu().numpy()>.2 for obj,kind in ((1,'sock'),(2,'foot'))}
            precision={k:p.projection_precision(v,refs[k],2)['tolerant'] for k,v in masks.items()}
            reset=False
            if any(v<.95 for v in precision.values()) or any(v.sum()<64 for v in masks.values()):
                masks,attempts=p.initialize_lit(sam,rgb,refs);reset=True
                precision={k:p.projection_precision(v,refs[k],2)['tolerant'] for k,v in masks.items()}
            if any(v<.95 for v in precision.values()) or any(v.sum()<64 for v in masks.values()):raise RuntimeError('Live SAM mask QA failed')
            raw=depth.infer_image(rgb[:,:,::-1].copy(),input_size=518);assert np.isfinite(raw).all()
            measured=np.r_[r['angle'],r['torque']];normalized=(measured-low)/np.where(constant,1.,span);normalized[constant]=.5
            inputs=[np.array(Image.fromarray(rgb).resize((64,64)),dtype='float32').transpose(2,0,1)/255,normalized.astype('float32')]
            for k in ('sock','foot'):
                dm=cv2.resize(cv2.GaussianBlur(raw*masks[k],(0,0),15),(64,64),interpolation=cv2.INTER_AREA)
                support=cv2.resize(cv2.GaussianBlur(masks[k].astype('float32'),(0,0),15),(64,64),interpolation=cv2.INTER_AREA)
                inputs.append(((dm-stats['depth_min']*support)/(stats['depth_max']-stats['depth_min']))[None])
            image,pred,_,_,state,_=model(*[torch.from_numpy(x[None]).cuda() for x in inputs],state)
            decoded=pred[0].cpu().numpy()*span+low;decoded[constant]=low[constant]
            assert np.isfinite(decoded).all()
        print(json.dumps(dict(action=decoded[:18].tolist(),predicted_torque=decoded[18:].tolist(),input_out_of_range_fraction=float(((normalized<0)|(normalized>1)).mean()),sam_precision=precision,sam_reset=reset,measured_blend=0.)),flush=True)

if __name__=='__main__':main()
