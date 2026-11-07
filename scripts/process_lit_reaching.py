#!/usr/bin/env python3
"""SAM2 on corrected lit RGB; cache exact training depth for train-only scaling."""
from __future__ import annotations
import argparse,fcntl,io,json,shutil,sys,time,zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from process_reaching_shareset import load_sam,load_depth,digest,write_json
from process_head_reaching_shareset import inspect_masks,masks_from
from sock_dressing_simulation.head_camera_profile import head_references
from sock_dressing_simulation.lit_perception import initialize_lit

SAM=Path('/home/cm-pc/catkin_ws/src/SegmentAnythingModel2/checkpoints/sam2_hiera_small.pt')
DEPTH=Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth')


def save_raw(path,raw):
    if shutil.disk_usage(path.parent).free<1_000_000_000:raise RuntimeError('Less than 1GB free; keep the incomplete staging data and inspect')
    buffer=io.BytesIO();np.lib.format.write_array(buffer,raw.astype('float32'),allow_pickle=False)
    temporary=path.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:archive.writestr('depth.npy',buffer.getvalue())
    temporary.replace(path)


def mask_quality(masks,refs,truth):
    checks,areas,precision=inspect_masks(masks,refs)
    iou=float((masks['sock']&truth).sum()/max(1,int((masks['sock']|truth).sum())))
    checks.update(sock_projection=precision['sock']>=.95,foot_projection=precision['foot']>=.95,visible_sock_iou=iou>=.90)
    return checks,areas,precision,iou


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--episode',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--phase',choices=['train-cache','full'],required=True)
    a=ap.parse_args();ep=a.episode.resolve();out=a.output.resolve();meta=json.loads((ep/'metadata.json').read_text())
    # Hold this descriptor for the entire process. A queued supervisor and an
    # additional worker may request the same episode; never write it twice.
    processing_lock=(ep/'.lit-processing.lock').open('w')
    fcntl.flock(processing_lock,fcntl.LOCK_EX)
    previous_report=ep/'perception/validation.json'
    if previous_report.exists():
        previous=json.loads(previous_report.read_text())
        if previous.get('passed') and previous.get('processing_source_sha256')==digest(__file__):
            if a.phase=='full' and not previous.get('depth_pending'):return
            if a.phase=='train-cache' and (out/'depth-raw-cache'/ep.name/'stats.json').exists():return
    if not meta.get('raw_validation_passed') or meta.get('lighting_profile')!='balanced':raise ValueError('Validated balanced lighting renders required')
    if a.phase=='train-cache' and meta['split']!='train':raise ValueError('Held-out data cannot fit depth calibration')
    count=meta['frames'];folder=ep/'perception';folder.mkdir(exist_ok=True);cache=out/'depth-raw-cache'/ep.name;cache.mkdir(parents=True,exist_ok=True)
    for path in ('camera_right_mask/sock_mask','camera_right_mask/foot_mask','camera_depth','depth_mask/sock_depth','depth_mask/foot_depth'):(ep/path).mkdir(parents=True,exist_ok=True)
    refs=head_references(meta,ROOT);cal=json.loads((out/'data/data_sock_reaching_v1/train/depth.json').read_text()) if a.phase=='full' else None
    source_sha=digest(__file__);source_dir=out/'processing-sources';source_dir.mkdir(exist_ok=True)
    if not (source_dir/(source_sha+'.py')).exists():shutil.copy2(__file__,source_dir/(source_sha+'.py'))
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[];resets=[];depth_rows=[];sam_high=None;thumbs=[];started=time.monotonic()
    selected=set(np.linspace(0,count-1,8,dtype=int))
    with torch.inference_mode():
        sam=load_sam(SAM,512);model=load_depth(DEPTH)
        for index in range(count):
            rgb=np.array(Image.open(ep/f'camera_right/{index}.png').convert('RGB'));truth=np.array(Image.open(ep/f'qa/visible_sock/{index}.png'))>0
            if index==0:
                masks,attempts=initialize_lit(sam,rgb,refs);write_json(folder/'initial-prompts.json',attempts)
            else:
                ids,logits=sam.track(rgb[:,:,::-1].copy());masks=masks_from(ids,logits)
            checks,areas,precision,iou=mask_quality(masks,refs,truth)
            if not all(checks.values()):
                previous=folder/'before-lit-reset'/str(index);previous.mkdir(parents=True,exist_ok=True)
                for kind,mask in masks.items():Image.fromarray(mask.astype('uint8')*255).save(previous/(kind+'.png'))
                write_json(previous/'checks.json',dict(checks=checks,iou=iou))
                masks,attempts=initialize_lit(sam,rgb,refs);checks,areas,precision,iou=mask_quality(masks,refs,truth);resolution=512
                if not all(checks.values()):
                    if sam_high is None:sam_high=load_sam(SAM,1024)
                    masks,attempts=initialize_lit(sam_high,rgb,refs);checks,areas,precision,iou=mask_quality(masks,refs,truth);resolution=1024
                resets.append(dict(frame=index,resolution=resolution,attempts=attempts))
            if not all(checks.values()):
                write_json(folder/'failure.json',dict(frame=index,checks=checks,visible_iou=iou,areas=areas));raise RuntimeError(f'Lit mask QA failed: {ep.name}/{index}: {checks} IoU={iou}')
            for kind,mask in masks.items():Image.fromarray(mask.astype('uint8')*255).save(ep/f'camera_right_mask/{kind}_mask/{index}.png')
            raw_file=cache/f'{index}.npz'
            if raw_file.exists():
                with np.load(raw_file) as array:raw=array['depth']
            else:raw=model.infer_image(rgb[:,:,::-1].copy(),input_size=518)
            if not np.isfinite(raw).all() or raw.min()<0:raise ValueError('Invalid depth prediction')
            row=dict(frame=index,passed=True,checks=checks,areas=areas,projection_precision_2px=precision,visible_sock_iou=iou)
            if a.phase=='train-cache':
                if not raw_file.exists():save_raw(raw_file,raw)
                depth_rows.append(dict(frame=index,rgb_sha256=digest(ep/f'camera_right/{index}.png'),minimum=float(raw.min()),maximum=float(raw.max())))
                row['depth_clip_fraction']=None
            else:
                normalized=(raw-cal['depth_min'])/(cal['depth_max']-cal['depth_min']);encoded=np.clip(normalized*255,0,255).astype('uint8')
                Image.fromarray(encoded).save(ep/f'camera_depth/{index}.png')
                upper={}
                for kind,mask in masks.items():
                    Image.fromarray(np.where(mask,encoded,0).astype('uint8')).save(ep/f'depth_mask/{kind}_depth/{index}.png')
                    upper[kind]=float((raw[mask]>cal['depth_max']).mean())
                checks.update(depth_finite=True,depth_nonzero=all(bool(encoded[m].any()) for m in masks.values()),depth_target_upper_clip=all(v<=.02 for v in upper.values()))
                row.update(depth_clip_fraction=float(((normalized<0)|(normalized>1)).mean()),target_depth_upper_clip_fraction=upper,passed=all(checks.values()))
                if not row['passed']:raise RuntimeError('Lit held-out depth exceeds frozen training range')
            old=np.array(Image.open(Path(meta['source_episode'])/f'camera_right_mask/sock_mask/{index}.png'))>0
            row['previous_visible_sock_iou']=float((old&truth).sum()/max(1,int((old|truth).sum())))
            n,labels,stats,_=cv2.connectedComponentsWithStats(masks['sock'].astype('uint8'))
            row['components_over_20px']=sum(int(v)>=20 for v in stats[1:,cv2.CC_STAT_AREA])
            rows.append(row)
            if index in selected:
                overlay=rgb.copy()
                for kind,color in [('sock',[255,40,40]),('foot',[40,255,40])]:
                    mask=masks[kind];overlay[mask]=overlay[mask]*.5+np.array(color)*.5
                im=Image.fromarray(overlay).resize((480,360));ImageDraw.Draw(im).text((5,5),f'frame {index} visible IoU={iou:.3f}',fill='red');thumbs.append(im)
            if index%25==0:
                write_json(folder/'progress.json',dict(processed=index+1,total=count,mask_passed=True,depth_phase=a.phase,last=row,resets=len(resets),elapsed_s=time.monotonic()-started));print('LIT PERCEPTION',ep.name,index+1,'/',count,'IoU',round(iou,4),flush=True)
    if a.phase=='train-cache':
        write_json(cache/'stats.json',dict(episode=str(ep),number=meta['workbook_pattern']['number'],split='train',count=count,
            minimum=min(r['minimum'] for r in depth_rows),maximum=max(r['maximum'] for r in depth_rows),rows=depth_rows,checkpoint_sha256=digest(DEPTH)))
    sheet=Image.new('RGB',(1440,360*((len(thumbs)+2)//3)),'white')
    for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*480,(i//3)*360))
    sheet.save(folder/'review.jpg',quality=90)
    report=dict(passed=True,depth_pending=a.phase=='train-cache',frames=count,rows=rows,resets=resets,camera_profile=meta['camera_profile'],
        lighting_profile='balanced',processing_source_sha256=source_sha,mask_processing_source_sha256=source_sha,
        sam_checkpoint_sha256=digest(SAM),depth_checkpoint_sha256=digest(DEPTH),sam_image_size=512,source_camera='camera_right',
        visible_reference_scope='Independent ID rendering for QA only; all training masks are SAM outputs',
        angle_sha256=digest(ep/'angle.csv'),elapsed_s=time.monotonic()-started)
    if cal is not None:report.update(depth_calibration=cal,depth_calibration_version=cal['calibration_version'])
    write_json(folder/'validation.json',report)
    meta.update(perception_passed=a.phase=='full',reaching_learning_ready=a.phase=='full',depth_recalibration_pending=a.phase=='train-cache',
        mask_source_camera='camera_right',depth_source_camera='camera_right',mask_lighting_profile='balanced',mask_validation_passed=True)
    for old_key in ('head_isolated_repaired_frames','perception_repaired_frames'):meta.pop(old_key,None)
    meta['perception_prompt_resets']=[r['frame'] for r in resets]
    if cal is not None:meta['depth_calibration_version']=cal['calibration_version']
    write_json(ep/'metadata.json',meta);print('LIT MASKS PASSED',ep.name,a.phase,flush=True)


if __name__=='__main__':main()
