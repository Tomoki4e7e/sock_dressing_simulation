#!/usr/bin/env python3
"""GPU SAM2/depth preprocessing, matched full-BPTT training and offline evaluation.

Original recordings are read-only. The compact float32 cache blurs masked depth
at native resolution before resizing; depth scale and signal ranges fit train only.
Logical batches use exact transition-weighted gradient accumulation over episodes
(the active No-SK model has no batch-dependent layers), preserving full BPTT.
"""
from __future__ import annotations
import argparse, hashlib, json, os, random, sys, time, traceback
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
import torch
from torchvision.transforms import ColorJitter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT.parent/'ShareSet/share_folder/src')]
from models.SAMDAMSARNN import SAMDAMSARNN
from utils.callback import LossScheduler
from process_reaching_shareset import load_sam, load_depth, digest, write_json
from sock_dressing_simulation.head_camera_profile import projected_mesh
from sock_dressing_simulation.lit_perception import initialize_lit
from sock_dressing_simulation.perception_audit import projection_precision

SAM = Path('/home/cm-pc/catkin_ws/src/SegmentAnythingModel2/checkpoints/sam2_hiera_small.pt')
DEPTH = Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth')


def atomic_npz(path, **arrays):
    with path.with_suffix('.tmp').open('wb') as f: np.savez_compressed(f, **arrays)
    path.with_suffix('.tmp').replace(path)


def prepare_manifest(source, out):
    index = json.loads((source/'dataset-index.json').read_text())
    assert index['all_70_passed'] and len(index['rows']) == 70
    # Group by physical ankle pitch/roll across ALL stages, avoiding condition leakage.
    keys = sorted({(r['pitch_deg'], r['roll_deg']) for r in index['rows']})
    rng = random.Random(4); rng.shuffle(keys)
    assignment = {key: ('test' if i < 4 else 'validation' if i < 8 else 'train') for i,key in enumerate(keys)}
    rows = []
    for r in index['rows']:
        row = dict(id=r['id'], stage=r['stage'], pitch_deg=r['pitch_deg'], roll_deg=r['roll_deg'],
                   frames=r['frames'], path=str(source/r['episode']), split=assignment[r['pitch_deg'],r['roll_deg']])
        ep=Path(row['path']); meta=json.loads((ep/'metadata.json').read_text())
        assert meta['reaching_learning_ready'] and meta['contact_pregrasp_learning_ready']
        assert json.loads((ep/'raw-validation.json').read_text())['passed']
        row['source_sha256']={n:digest(ep/n) for n in ('angle.csv','torque.csv','frames.csv','metadata.json')}
        rows.append(row)
    write_json(out/'manifest.json',dict(source=str(source),source_index_sha256=digest(source/'dataset-index.json'),
        split_seed=4,split_group='physical pitch/roll shared across A-E; every split covers all five stages',rows=rows,
        counts={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')}))
    return rows


def preprocess(rows,out,pilot=False):
    cache=out/'cache';cache.mkdir(exist_ok=True)
    sam=load_sam(SAM,512,'cuda');depth=load_depth(DEPTH,'cuda')
    selected=rows[:1] if pilot else rows
    for row in selected:
        target=cache/(row['id']+'.npz');report=cache/(row['id']+'.json')
        if target.exists() and report.exists():continue
        ep=Path(row['path']);m=json.loads((ep/'metadata.json').read_text())
        state=json.loads((ROOT/m['condition']['state']).read_text());delta=np.asarray(m['translation_m'])
        skin=state['source']['skin_surface'];camera=m['camera_transform']
        refs=dict(sock=projected_mesh(np.asarray(state['particles_world'])+delta,state['triangles'],camera,m['camera_fov'],(640,480)),
                  foot=projected_mesh(np.asarray(skin['vertices'])+delta,skin['triangles'],camera,m['camera_fov'],(640,480)))
        frames=[];rgbrows=[];masked=[];supports=[];minimum=float('inf');maximum=-float('inf');qa=[];resets=[];hashes=[]
        start=time.time(); count=min(row['frames'],3) if pilot else row['frames']
        with torch.inference_mode():
            for i in range(count):
                filename=ep/f'camera_right/{i}.png';rgb=np.asarray(Image.open(filename).convert('RGB'))
                if i==0:
                    masks,attempts=initialize_lit(sam,rgb,refs)
                else:
                    ids,logits=sam.track(rgb[:,:,::-1].copy())
                    masks={kind:logits[ids.index(obj),0].float().cpu().numpy()>.2 for obj,kind in ((1,'sock'),(2,'foot'))}
                precision={k:projection_precision(v,refs[k],2)['tolerant'] for k,v in masks.items()}
                if any(v<.95 for v in precision.values()) or any(v.sum()<64 for v in masks.values()):
                    masks,attempts=initialize_lit(sam,rgb,refs);resets.append(i)
                    precision={k:projection_precision(v,refs[k],2)['tolerant'] for k,v in masks.items()}
                if any(v<.95 for v in precision.values()) or any(v.sum()<64 for v in masks.values()):
                    raise RuntimeError(f'Mask QA failed {row["id"]}/{i}: {precision}')
                qa.append(dict(frame=i,precision=precision,areas={k:int(v.sum()) for k,v in masks.items()}))
                if i%2==0:
                    raw=depth.infer_image(rgb[:,:,::-1].copy(),input_size=518)
                    assert np.isfinite(raw).all() and raw.min()>=0
                    minimum=min(minimum,float(raw.min()));maximum=max(maximum,float(raw.max()))
                    # Blur is linear, so normalize using the frozen train range later.
                    masked.append(np.stack([cv2.resize(cv2.GaussianBlur(raw*masks[k],(0,0),15),(64,64),interpolation=cv2.INTER_AREA) for k in ('sock','foot')]))
                    supports.append(np.stack([cv2.resize(cv2.GaussianBlur(masks[k].astype('float32'),(0,0),15),(64,64),interpolation=cv2.INTER_AREA) for k in ('sock','foot')]))
                    rgbrows.append(np.array(Image.fromarray(rgb).resize((64,64))));frames.append(i);hashes.append(digest(filename))
                if i in (0,count//2,count-1):
                    preview=rgb.copy()
                    for k,color in (('sock',np.array([30,130,255])),('foot',np.array([255,130,30]))):
                        preview[masks[k]]=(preview[masks[k]]*.5+color*.5).astype('uint8')
                    Image.fromarray(preview).save(cache/f'{row["id"]}-mask-{i}.jpg')
                if i%50==0:
                    write_json(out/'progress.json',dict(completed=False,phase='perception',episode=row['id'],frame=i,total_frames=count,device='cuda',time=time.time()))
                    print('PERCEPTION',row['id'],i,count,flush=True)
        signals=[]
        for name in ('angle','torque'):
            values=np.loadtxt(ep/(name+'.csv'),delimiter=',',dtype='float32',ndmin=2)
            assert values.shape==(row['frames'],18) and np.isfinite(values).all()
            if name=='torque':values=savgol_filter(values,29,2,axis=0)
            signals.append(values[frames])
        atomic_npz(target,rgb=np.stack(rgbrows),masked=np.stack(masked),support=np.stack(supports),joint=np.concatenate(signals,axis=1),frames=np.array(frames))
        write_json(report,dict(id=row['id'],split=row['split'],depth_min=minimum,depth_max=maximum,
            frame_rgb_sha256=hashes,qa=qa,resets=resets,elapsed_s=time.time()-start,device='cuda',
            sam_sha256=digest(SAM),depth_sha256=digest(DEPTH),source_sha256=row['source_sha256']))
        print('PREPARED',row['id'],time.time()-start,flush=True)
    del sam,depth;torch.cuda.empty_cache()


def load_arrays(rows,out):
    training=[json.loads((out/'cache'/f'{r["id"]}.json').read_text()) for r in rows if r['split']=='train']
    lo=min(r['depth_min'] for r in training);hi=max(r['depth_max'] for r in training)
    raw={r['id']:{k:v for k,v in np.load(out/'cache'/f'{r["id"]}.npz').items()} for r in rows}
    joints=np.concatenate([raw[r['id']]['joint'] for r in rows if r['split']=='train'])
    low=joints.min(0);high=joints.max(0);span=high-low;constant=span<=1e-7
    write_json(out/'normalization.json',dict(source_split='train',joint_min=low.tolist(),joint_max=high.tolist(),constant_columns=np.flatnonzero(constant).tolist(),depth_min=lo,depth_max=hi,torque_source='recorded motor drive, no synthesized gravity',depth_clipped=False))
    datasets={s:[] for s in ('train','validation','test')}
    for r in rows:
        d=raw[r['id']];normalized=(d['joint']-low)/np.where(constant,1.,span);normalized[:,constant]=.5
        depth=(d['masked']-lo*d['support'])/(hi-lo)
        row=dict(id=r['id'],rgb=torch.from_numpy(d['rgb'].transpose(0,3,1,2).copy()).float()/255,
                 joint=torch.from_numpy(normalized.astype('float32')),sock=torch.from_numpy(depth[:,0:1]),foot=torch.from_numpy(depth[:,1:2]),frames=d['frames'],reference=d['joint'])
        assert all(torch.isfinite(row[k]).all() for k in ('rgb','joint','sock','foot'))
        datasets[r['split']].append(row)
    return datasets,low,high


def epoch(model,rows,optimizer,scaler,scheduler,device,h):
    training=optimizer is not None;model.train(training)
    # The archived forward computes SK outputs but discards them before its
    # No-SK recurrent path. Freeze unused BatchNorm statistics for singleton
    # microbatches; these modules do not affect any learned prediction.
    model.attn_joint.eval();model.attn_torque.eval()
    order=list(rows)
    if training:random.shuffle(order)
    totals=np.zeros(4);loss_total=0.;jitter=ColorJitter(contrast=.5,brightness=.5,saturation=.1)
    for begin in range(0,len(order),h['batch_size']):
        batch=order[begin:begin+h['batch_size']]
        n=sum(len(r['joint'])-1 for r in batch);pn=sum(len(r['joint'])-2 for r in batch)
        pw=float(scheduler(h['pt_loss']))
        if training:optimizer.zero_grad(set_to_none=True)
        for row in batch:
            with torch.set_grad_enabled(training):
                rgb,joint,sock,foot=[row[k].unsqueeze(0).to(device) for k in ('rgb','joint','sock','foot')]
                xrgb=jitter(rgb[0]).unsqueeze(0) if training else rgb
                if training:xrgb=xrgb+torch.randn_like(xrgb)*h['stdev'];xjoint=joint+torch.randn_like(joint)*h['stdev']
                else:xjoint=joint
                state=None;images=[];preds=[];encs=[];decs=[]
                with torch.autocast(device_type='cuda'):
                    for t in range(joint.shape[1]-1):
                        image,pred,enc,dec,state,_=model(xrgb[:,t],xjoint[:,t],sock[:,t],foot[:,t],state)
                        images.append(image);preds.append(pred);encs.append(enc);decs.append(dec)
                    pred=torch.stack(preds,1);length=joint.shape[1]-1
                    components=[(torch.stack(images,1)-rgb[:,1:]).square().mean()*length/n,
                        (pred[:,:,:18]-joint[:,1:,:18]).square().mean()*length/n,
                        (pred[:,:,18:]-joint[:,1:,18:]).square().mean()*length/n,
                        (torch.stack(decs[:-1],1)-torch.stack(encs[1:],1)).square().mean()*(length-1)/pn]
                    loss=sum(v*w for v,w in zip(components,(h['img_loss'],h['angle_loss'],h['torque_loss'],pw)))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                if training:scaler.scale(loss).backward()
                totals+=np.array([v.detach().item() for v in components]);loss_total+=loss.detach().item()
                del loss,components,images,preds,encs,decs,state,pred,rgb,joint,sock,foot,xrgb,xjoint
        if training:
            scaler.unscale_(optimizer)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite gradient')
            scaler.step(optimizer);scaler.update()
    nb=(len(order)+h['batch_size']-1)//h['batch_size']
    return dict(loss=loss_total/nb,**dict(zip(('image','angle','torque','points'),(totals/nb).tolist())),point_weight=pw)


def infer(model,rows,low,high,out,epoch_number):
    dest=out/f'inference-epoch-{epoch_number}';dest.mkdir(exist_ok=True);reports=[]
    model.eval()
    with torch.inference_mode():
        for row in rows:
            state=None;predictions=[];predicted_images=[];start=time.time()
            for t in range(len(row['joint'])-1):
                image,pred,_,_,state,_=model(*[row[k][None,t].cuda() for k in ('rgb','joint','sock','foot')],state)
                predictions.append(pred[0].cpu().numpy());predicted_images.append(image[0].cpu().numpy())
            prediction=np.stack(predictions)*(high-low)+low
            const=(high-low)<=1e-7;prediction[:,const]=low[const]
            truth=row['reference'][1:];error=prediction-truth
            episode=dest/row['id'];episode.mkdir(exist_ok=True)
            np.savetxt(episode/'predicted_state.csv',prediction,delimiter=',');np.savetxt(episode/'reference_next_state.csv',truth,delimiter=',')
            metrics=dict(id=row['id'],frames=len(prediction),device='cuda',mode='observed-input one-step recurrent prediction',
                left_arm_rmse_rad=float(np.sqrt(np.mean(error[:,:7]**2))),right_arm_rmse_rad=float(np.sqrt(np.mean(error[:,9:16]**2))),
                arm_effort_rmse_Nm=float(np.sqrt(np.mean(error[:,[18+i for i in list(range(7))+list(range(9,16))]]**2))),
                finger_rmse_m=float(np.sqrt(np.mean(error[:,[7,16]]**2))),elapsed_s=time.time()-start)
            from video_reaching_inference import MP4
            writer=MP4(episode/'offline-prediction.mp4',(768,300),5)
            for t,p in enumerate(predicted_images):
                images=[row['rgb'][t].numpy(),row['rgb'][t+1].numpy(),np.clip(p,0,1)]
                canvas=np.zeros((300,768,3),dtype='uint8')
                for i,img in enumerate(images):canvas[30:286,i*256:(i+1)*256]=cv2.resize((img.transpose(1,2,0)*255).astype('uint8'),(256,256))
                cv2.putText(canvas,f'{row["id"]} epoch {epoch_number} | observed / next / prediction',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
                writer.append(canvas)
            writer.close();write_json(episode/'metrics.json',metrics);reports.append(metrics)
    write_json(dest/'report.json',dict(completed=True,epoch=epoch_number,device='cuda',rows=reports,
        scope='Held-out observed-input prediction; this does not establish autonomous physical reaching success.'))


def train(rows,out,config,resume=False):
    datasets,low,high=load_arrays(rows,out);h=config['hyperparameters'];m=config['model'];device=torch.device('cuda')
    model=SAMDAMSARNN(rec_dim=m['rec_dim'],k_dim=m['k_dim'],joint_dim=36,temperature=m['temperature'],heatmap_size=m['heatmap_size'],im_size=m['im_size']).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=h['learning_rate'],weight_decay=h['weight_decay'])
    scaler=torch.cuda.amp.GradScaler();scheduler=LossScheduler(1000,'s');start_epoch=0;best=float('inf');history=[]
    cp=out/'model';cp.mkdir(exist_ok=True);checkpoint=cp/'SARNN_latest.pth'
    if checkpoint.exists():
        if not resume:raise RuntimeError('Existing checkpoint; pass --resume')
        saved=torch.load(checkpoint,map_location='cuda');model.load_state_dict(saved['model_state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict']);scaler.load_state_dict(saved['scaler'])
        scheduler.counter=saved['scheduler_counter'];start_epoch=saved['epoch'];best=saved['best'];history=saved['history']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all(saved['cuda_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
    for e in range(start_epoch,h['num_epochs']):
        tick=time.time();training=epoch(model,datasets['train'],optimizer,scaler,scheduler,device,h)
        validation=epoch(model,datasets['validation'],None,scaler,scheduler,device,h)
        result=dict(epoch=e+1,train=training,validation=validation,duration_s=time.time()-tick)
        history.append(result)
        saved=dict(epoch=e+1,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scaler=scaler.state_dict(),scheduler_counter=scheduler.counter,
            best=min(best,validation['loss']),history=history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state())
        if e==start_epoch or (e+1)%10==0 or e+1==h['num_epochs']:
            torch.save(saved,checkpoint.with_suffix('.tmp'));checkpoint.with_suffix('.tmp').replace(checkpoint)
        if validation['loss']<best:
            best=validation['loss'];torch.save(dict(epoch=e+1,model_state_dict=model.state_dict(),validation_loss=best),cp/'SARNN_best.pth')
        remaining=(h['num_epochs']-e-1)*np.mean([r['duration_s'] for r in history[-10:]])
        write_json(out/'progress.json',dict(completed=False,phase='training',device='cuda',gpu=torch.cuda.get_device_name(),epoch=e+1,target_epochs=h['num_epochs'],estimated_remaining_s=float(remaining),latest=result))
        with (cp/'loss.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print('EPOCH',json.dumps(result),'ETA_SECONDS',remaining,flush=True)
        if e==0:
            # Intermediate diagnostic on validation only; test remains untouched.
            infer(model,datasets['validation'],low,high,out,e+1)
        if e+1==h['num_epochs']:
            torch.save(saved,cp/f'SARNN_{e+1}.pth')
            infer(model,datasets['test'],low,high,out,e+1)
    write_json(out/'progress.json',dict(completed=True,phase='complete',device='cuda',epochs=h['num_epochs'],checkpoint_sha256=digest(checkpoint)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'artifacts/contact-pregrasp-learning-exact-v2');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--pilot',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(0);torch.cuda.manual_seed_all(0);np.random.seed(0);random.seed(0)
    if not torch.cuda.is_available():raise RuntimeError('GPU is required; CPU fallback is disabled')
    try:
        rows=prepare_manifest(a.source.resolve(),out)
        import yaml
        original=ROOT.parent/'ShareSet/share_folder/configs/config.yaml'
        config=yaml.safe_load(original.read_text());config['training']=dict(device='cuda',full_sequence_bptt=True,seed=0,logical_batch_size=config['hyperparameters']['batch_size'],physical_microbatch_size=1,gradient_accumulation='transition-weighted; optimizer once per logical batch',source_config=str(original),source_sha256=digest(original))
        config['preprocessing']=dict(sam_device='cuda',depth_device='cuda',native_size=[640,480],crop=[0,0,640,480],resize=64,gaussian_radius=15,depth_range_fit='train only',torque='recorded drive forces without gravity addition',depth_cache='float32 Gaussian masked depth, no intermediate 8-bit quantization')
        write_json(out/'configuration.json',config)
        preprocess(rows,out,a.pilot)
        if not a.pilot:train(rows,out,config,a.resume)
    except BaseException as error:
        write_json(out/'failure.json',dict(failed=True,error=repr(error),traceback=traceback.format_exc(),time=time.time()));raise

if __name__=='__main__':main()
