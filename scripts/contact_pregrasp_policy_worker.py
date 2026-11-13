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
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--episode',type=Path,required=True)
    ap.add_argument('--flow-ode-steps',type=int,default=8);ap.add_argument('--flow-execute-steps',type=int,choices=[1,2,4],default=1);ap.add_argument('--flow-seed',type=int,default=0);a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
        config=json.loads((a.run/'configuration.json').read_text());stats=json.loads((a.run/'normalization.json').read_text());m=config['model']
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        is_flow=m.get('name')=='SAMDAMSARNNFlow'
        if is_flow:
            from models.SAMDAMSARNN_Flow import SAMDAMSARNNFlow
            from sock_dressing_simulation.config import load_config
            from sock_dressing_simulation.joints import JointMap
            model=SAMDAMSARNNFlow(**config['flow_model']).cuda().eval()
            joint_map=JointMap.from_config(load_config(p.ROOT/'config/arm_reaching_safe_start.yaml'))
            rules=[joint_map.mimic[side+'_gripper/mimic_joint'] for side in ('left','right')]
            mimic_multipliers=[float(rule['multiplier']) for rule in rules]
            mimic_offsets=[float(rule['offset']) for rule in rules]
            from sock_dressing_simulation.flow_chunk_execution import FlowChunkExecutor
        else:
            model=p.SAMDAMSARNN(rec_dim=m['rec_dim'],k_dim=m['k_dim'],joint_dim=36,temperature=m['temperature'],heatmap_size=m['heatmap_size'],im_size=m['im_size']).cuda().eval()
        model.load_state_dict(saved['model_state_dict'],strict=True)
        sam=p.load_sam(p.SAM,512,'cuda');depth=p.load_depth(p.DEPTH,'cuda')
        # This camera predictor overwrites image_size=1024 after constructing
        # its prompt encoder. Restore the configured encoder resolution.
        sam.image_size=sam.sam_image_embedding_size*sam.backbone_stride
        assert sam.image_size==512
        # FP32 SAM attention needs the CUDA math kernel when Flash attention
        # only supports half precision on the installed torch/GPU combination.
        import sam2.modeling.sam.transformer as sam_attention
        sam_attention.MATH_KERNEL_ON=True
        meta=json.loads((a.episode/'metadata.json').read_text());s=json.loads((p.ROOT/meta['condition']['state']).read_text());delta=np.array(meta['translation_m']);cam=meta['camera_transform'];skin=s['source']['skin_surface']
        refs=dict(sock=p.projected_mesh(np.array(s['particles_world'])+delta,s['triangles'],cam,meta['camera_fov'],(640,480)),foot=p.projected_mesh(np.array(skin['vertices'])+delta,skin['triangles'],cam,meta['camera_fov'],(640,480)))
        low=np.array(stats['joint_min']);high=np.array(stats['joint_max']);span=high-low;constant=span<=1e-7
        if is_flow:
            executor=FlowChunkExecutor(model,low[:18],high[:18],mimic_multipliers,mimic_offsets,
                                       a.flow_ode_steps,a.flow_execute_steps,a.flow_seed)
    state=None;initialized=False
    print(json.dumps(dict(ready=True,epoch=saved['epoch'],device='cuda',precision='float32',sam_image_size=sam.image_size,gpu=torch.cuda.get_device_name(),model_type=m.get('name','SAMDAMSARNN'),flow_seed=a.flow_seed if is_flow else None,ode_steps=a.flow_ode_steps if is_flow else None,execute_steps=a.flow_execute_steps if is_flow else None,flow_noise_schedule='one draw per observation including held steps' if is_flow else None,lstm_updates='every observation',action_selection=f'Execute {a.flow_execute_steps} slots of 8-step chunk; observe every 0.2 seconds' if is_flow else 'One-step prediction',model_devices=dict(samdam=str(next(model.parameters()).device),sam2=str(next(sam.parameters()).device),depth=str(next(depth.parameters()).device)))),flush=True)
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('quit'):break
        with contextlib.redirect_stdout(sys.stderr),torch.inference_mode():
            rgb=np.array(Image.open(r['rgb']).convert('RGB'));assert rgb.shape==(480,640,3)
            if not initialized:masks,attempts=p.initialize_lit(sam,rgb,refs);initialized=True
            else:
                ids,logits=sam.track(rgb[:,:,::-1].copy());masks={kind:logits[ids.index(obj),0].float().cpu().numpy()>.2 for obj,kind in ((1,'sock'),(2,'foot'))}
            precision={k:p.projection_precision(v,refs[k],2)['tolerant'] for k,v in masks.items()}
            reset=False;reset_error=None
            if any(v<.95 for v in precision.values()) or any(v.sum()<64 for v in masks.values()):
                try:
                    refreshed,attempts=p.initialize_lit(sam,rgb,refs)
                    masks=refreshed;reset=True
                    precision={k:p.projection_precision(v,refs[k],2)['tolerant'] for k,v in masks.items()}
                except (ValueError,RuntimeError) as error:
                    # Occlusion is a possible policy outcome. Keep this frame's
                    # actual tracker masks rather than replacing them with
                    # projected ground truth or stopping the rollout.
                    if not str(error).startswith(('Visible sock opening not found',
                            'No foreground for a SAM prompt','Lit prompt initialization failed')):
                        raise
                    reset_error=str(error)
            qa_passed=all(v>=.95 for v in precision.values()) and all(v.sum()>=64 for v in masks.values())
            raw=depth.infer_image(rgb[:,:,::-1].copy(),input_size=518);assert np.isfinite(raw).all()
            measured=np.r_[r['angle'],r['torque']];normalized=(measured-low)/np.where(constant,1.,span);normalized[constant]=.5
            inputs=[np.array(Image.fromarray(rgb).resize((64,64)),dtype='float32').transpose(2,0,1)/255,normalized.astype('float32')]
            for k in ('sock','foot'):
                dm=cv2.resize(cv2.GaussianBlur(raw*masks[k],(0,0),15),(64,64),interpolation=cv2.INTER_AREA)
                support=cv2.resize(cv2.GaussianBlur(masks[k].astype('float32'),(0,0),15),(64,64),interpolation=cv2.INTER_AREA)
                inputs.append(((dm-stats['depth_min']*support)/(stats['depth_max']-stats['depth_min']))[None])
            tensors=[torch.from_numpy(x[None]).cuda() for x in inputs]
            if is_flow:
                # Live contact efforts can exceed the offline training range.
                # Keep GPU inference in FP32 to avoid FP16 overflow.
                observation=model(*tensors,state)
                state=observation.state
                action,flow_plan=executor.step(observation.condition,tensors[1][:,:18],int(r['index']))
                predicted_torque=observation.torque.float()[0].cpu().numpy()*span[18:]+low[18:]
                predicted_torque[constant[18:]]=low[18:][constant[18:]]
                decoded=np.r_[action[0].cpu().numpy(),predicted_torque]
            else:
                image,pred,_,_,state,_=model(*tensors,state)
                decoded=pred[0].cpu().numpy()*span+low;decoded[constant]=low[constant]
            assert np.isfinite(decoded).all()
        result=dict(action=decoded[:18].tolist(),predicted_torque=decoded[18:].tolist(),input_out_of_range_fraction=float(((normalized<0)|(normalized>1)).mean()),sam_precision=precision,sam_reset=reset,sam_qa_passed=qa_passed,sam_reset_error=reset_error,sam_mask_pixels={k:int(v.sum()) for k,v in masks.items()},measured_blend=0.)
        if is_flow:result['flow_plan']=flow_plan
        print(json.dumps(result),flush=True)

if __name__=='__main__':main()
