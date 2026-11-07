#!/usr/bin/env python3
"""ShareSet right-head RGB -> SAM2 + Depth Anything; left RGB stays stereo raw."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from process_reaching_shareset import load_sam,load_depth,digest,write_json
from sock_dressing_simulation.head_camera_profile import head_references,head_prompts
from sock_dressing_simulation.perception_audit import projection_precision

PROFILE='head-stereo-zero-offset-v2'


def calibrate(args,model):
    target=args.output/'data/data_sock_reaching_v1/train/depth.json'
    if target.exists():
        result=json.loads(target.read_text())
        if result.get('lighting_profile'):
            raise ValueError('Lighting-corrected data require process_lit_reaching.py and its visible-shape QA')
        if result.get('camera_profile')!=PROFILE or result['checkpoint_sha256']!=digest(args.depth_checkpoint):
            raise ValueError('Depth calibration belongs to another camera/checkpoint')
        if result.get('calibration_version')!='full-training-frames-v3':
            raise ValueError('Sparse depth calibration is obsolete; use recalibrate_head_depth.py')
        return result
    samples=[]
    episodes=sorted((args.output/'data/data_sock_reaching_v1/train').glob('*/metadata.json'))
    if len(episodes)!=8:raise ValueError('All eight raw training episodes are required for depth calibration')
    for metadata_path in episodes:
        episode=metadata_path.parent
        meta=json.loads((episode/'metadata.json').read_text())
        if meta.get('lighting_profile'):
            raise ValueError('Lighting-corrected data require finish_lit_reaching_dataset.py')
        number=meta['workbook_pattern']['number']
        if number not in (10,13,25,38):raise ValueError('Unexpected calibration condition')
        if not meta.get('raw_validation_passed') or meta.get('camera_profile')!=PROFILE:raise ValueError('Head calibration training episode is not ready')
        for index in range(meta['frames']):
            path=episode/'camera_right'/f'{index}.png'
            raw=model.infer_image(cv2.imread(str(path)),input_size=518)
            samples.append(dict(path=str(path),sha256=digest(path),number=number,split='train',minimum=float(raw.min()),maximum=float(raw.max())))
    result=dict(camera_profile=PROFILE,depth_min=0.,depth_max=max(s['maximum'] for s in samples)*(1+1/255),
                training_frames=len(samples),calibration_version='full-training-frames-v3',
                checkpoint_sha256=digest(args.depth_checkpoint),source='Every physical right-head training frame; native ReLU floor and one-bin upper headroom',samples=samples)
    if result['depth_max']<=result['depth_min']:raise ValueError('Degenerate head depth calibration')
    write_json(target,result);return result


def initialize(sam,rgb,points):
    sam.load_first_frame(rgb[:,:,::-1].copy())
    for obj,kind in ((1,'sock'),(2,'foot')):
        p=points[kind]
        _,ids,logits=sam.add_new_points(0,obj,np.array(p['points'],dtype='float32'),np.array(p['labels'],dtype='int32'))
    return ids,logits


def masks_from(ids,logits):
    return {kind:logits[ids.index(obj),0].float().cpu().numpy()>.2 for obj,kind in ((1,'sock'),(2,'foot'))}


def inspect_masks(masks,refs):
    areas={k:int(m.sum()) for k,m in masks.items()}
    precision={k:projection_precision(m,refs[k],2)['tolerant'] for k,m in masks.items()}
    checks=dict(sock_nonempty=20<=areas['sock']<1280*960*.6,foot_nonempty=100<=areas['foot']<1280*960*.6,
                distinct=not np.array_equal(masks['sock'],masks['foot']),nonoverlap=not bool((masks['sock']&masks['foot']).any()),
                sock_projection=precision['sock']>=.8,foot_projection=precision['foot']>=.8)
    return checks,areas,precision


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'artifacts/reaching-shareset-head-v2')
    ap.add_argument('--episode',type=Path);ap.add_argument('--calibrate-only',action='store_true')
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--sam-checkpoint',type=Path,default=Path('/home/cm-pc/catkin_ws/src/SegmentAnythingModel2/checkpoints/sam2_hiera_small.pt'))
    ap.add_argument('--depth-checkpoint',type=Path,default=Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth'))
    args=ap.parse_args();args.output=args.output.resolve();torch.set_num_threads(args.threads);torch.set_grad_enabled(False)
    with torch.inference_mode():
        depth_model=load_depth(args.depth_checkpoint);cal=calibrate(args,depth_model)
        if args.calibrate_only:print('HEAD CALIBRATION READY',cal['depth_min'],cal['depth_max'],flush=True);return
        if args.episode is None:ap.error('--episode is required')
        ep=args.episode.resolve();meta=json.loads((ep/'metadata.json').read_text())
        if meta.get('camera_profile')!=PROFILE or not meta.get('raw_validation_passed'):raise ValueError('Physical head stereo raw data are required')
        count=meta['frames'];expected={str(i) for i in range(count)}
        for name in ('camera_right','camera_left','camera_overview'):
            if {p.stem for p in (ep/name).glob('*.png')}!=expected:raise ValueError('Stereo/confirmation image IDs differ')
        refs=head_references(meta,ROOT);folder=ep/'perception';folder.mkdir(exist_ok=True)
        for kind,mask in refs.items():Image.fromarray(mask.astype('uint8')*255).save(folder/f'reference-{kind}-amodal.png')
        for name in ('camera_depth','camera_right_mask/sock_mask','camera_right_mask/foot_mask','depth_mask/sock_depth','depth_mask/foot_depth'):
            (ep/name).mkdir(parents=True,exist_ok=True)
        sam=load_sam(args.sam_checkpoint,512);rows=[];resets=[];points=None;thumbs=[];start=time.monotonic()
        selected=set(np.linspace(0,count-1,8,dtype=int))
        for index in range(count):
            rgb=np.array(Image.open(ep/'camera_right'/f'{index}.png').convert('RGB'))
            if index==0:
                points=head_prompts(rgb,refs);ids,logits=initialize(sam,rgb,points)
                write_json(folder/'initial-prompts.json',points)
            else:ids,logits=sam.track(rgb[:,:,::-1].copy())
            masks=masks_from(ids,logits);checks,areas,precision=inspect_masks(masks,refs)
            if not all(checks.values()):
                original=folder/'before-prompt-reset'/str(index);original.mkdir(parents=True,exist_ok=True)
                for kind,mask in masks.items():Image.fromarray(mask.astype('uint8')*255).save(original/f'{kind}.png')
                points=head_prompts(rgb,refs);ids,logits=initialize(sam,rgb,points)
                masks=masks_from(ids,logits);checks,areas,precision=inspect_masks(masks,refs)
                resets.append(dict(frame=index,reason='Mask semantic/projection QA requested a fresh current-image prompt',prompts=points))
                if len(resets)>max(50,count//3):raise RuntimeError('Head tracker requires excessive prompt resets')
            raw=depth_model.infer_image(rgb[:,:,::-1].copy(),input_size=518)
            normalized=(raw-cal['depth_min'])/(cal['depth_max']-cal['depth_min'])
            encoded=np.clip(normalized*255,0,255).astype('uint8')
            checks.update(depth_finite=bool(np.isfinite(raw).all()),depth_nonzero=all(bool(encoded[m].any()) for m in masks.values()))
            row=dict(frame=index,passed=all(checks.values()),checks=checks,areas=areas,projection_precision_2px=precision,
                     depth_clip_fraction=float(((normalized<0)|(normalized>1)).mean()))
            rows.append(row)
            Image.fromarray(encoded).save(ep/'camera_depth'/f'{index}.png')
            for kind,mask in masks.items():
                Image.fromarray(mask.astype('uint8')*255).save(ep/'camera_right_mask'/f'{kind}_mask'/f'{index}.png')
                Image.fromarray(np.where(mask,encoded,0).astype('uint8')).save(ep/'depth_mask'/f'{kind}_depth'/f'{index}.png')
            if index in selected:
                overlay=rgb.copy()
                for kind,color in (('sock',[255,40,40]),('foot',[40,255,40])):
                    mask=masks[kind];overlay[mask]=overlay[mask]*.5+np.array(color)*.5
                thumb=Image.fromarray(overlay).resize((480,360));ImageDraw.Draw(thumb).text((5,5),f'HEAD RIGHT frame {index} QA={row["passed"]}',fill='red');thumbs.append(thumb)
            if index%25==0:
                write_json(folder/'progress.json',dict(processed=index+1,total=count,last=row,resets=len(resets),elapsed_s=time.monotonic()-start))
                print('HEAD PERCEPTION',ep.name,index+1,'/',count,'QA',row['passed'],'areas',areas,flush=True)
        sheet=Image.new('RGB',(1440,360*((len(thumbs)+2)//3)),'white')
        for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*480,(i//3)*360))
        sheet.save(folder/'review.jpg',quality=90)
        result=dict(passed=all(row['passed'] for row in rows),frames=count,rows=rows,camera_profile=PROFILE,
                    source_camera='camera_right: physical Dry-AIREC head See3CAM right',camera_left='physical See3CAM left; raw stereo companion',
                    excluded_camera='camera_overview: confirmation only',sam_image_size=512,sam_roi=[0,0,1280,960],sam_threshold=.2,
                    prompt_method='Visible sock cuff + projected frozen sock/right-lower-leg references; no reference pixels copied into model outputs',
                    projection_raster_tolerance_px=2,projection_triangle_fill='Union of faces; not even/odd polygon fill',
                    resets=resets,sam_checkpoint_sha256=digest(args.sam_checkpoint),depth_checkpoint_sha256=digest(args.depth_checkpoint),
                    depth_calibration=cal,processing_source_sha256=digest(__file__),angle_sha256=digest(ep/'angle.csv'),elapsed_s=time.monotonic()-start)
        write_json(folder/'validation.json',result)
        meta.update(perception_passed=result['passed'],reaching_learning_ready=result['passed'],
                    depth_encoding='uint8 relative Depth Anything V2 depth; frozen right-head training calibration',
                    depth_source_camera='camera_right',mask_source_camera='camera_right',
                    learning_image_directories=['camera_right','camera_left'],confirmation_image_directory='camera_overview',
                    perception_prompt_resets=[r['frame'] for r in resets])
        write_json(ep/'metadata.json',meta)
        write_json(folder/'progress.json',dict(completed=True,processed=count,total=count,passed=result['passed'],last=rows[-1]))
        if not result['passed']:raise RuntimeError(f'Head camera mask QA failed: {ep}')
        print('HEAD PERCEPTION PASSED',ep.name,count,flush=True)


if __name__=='__main__':main()
