#!/usr/bin/env python3
"""Compare lighting with independent visible-cloth silhouettes and ShareSet models."""
from pathlib import Path
import argparse,json,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from process_reaching_shareset import load_sam,load_depth,write_json
from process_head_reaching_shareset import initialize,masks_from,inspect_masks
from sock_dressing_simulation.head_camera_profile import head_references,head_prompts
from repair_head_reaching_perception import repair_prediction
from sock_dressing_simulation.lit_perception import initialize_lit


def visible_cloth(rgb):
    r,g,b=[rgb[:,:,i].astype(float) for i in range(3)]
    return (b>g*1.2+5)&(g>r*1.2+2)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('probe',type=Path);ap.add_argument('--metadata',type=Path,default=ROOT/'artifacts/reaching-shareset-head-v2/data/data_sock_reaching_v1/train/pose-10-held-A-v00-trajectory/metadata.json');a=ap.parse_args()
    report=json.loads((a.probe/'lighting-probe.json').read_text());meta=json.loads(a.metadata.read_text());refs=head_references(meta,ROOT)
    truth=visible_cloth(np.array(Image.open(a.probe/'visibility-reference-right.png')))
    truth &= cv2.dilate(refs['sock'].astype('uint8'),np.ones((5,5),dtype='uint8'))>0
    Image.fromarray(truth.astype('uint8')*255).save(a.probe/'visible-cloth-qa.png')
    torch.set_num_threads(2);torch.set_grad_enabled(False);rows=[];images=[];depths=[]
    with torch.inference_mode():
        sam=load_sam(Path('/home/cm-pc/catkin_ws/src/SegmentAnythingModel2/checkpoints/sam2_hiera_small.pt'),512)
        depth=load_depth(Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth'))
        for candidate in report['records']:
            name=candidate['profile'];folder=a.probe/name;rgb=np.array(Image.open(folder/'camera_right.png').convert('RGB'))
            prompts=head_prompts(rgb,refs);ids,logits=initialize(sam,rgb,prompts);masks=masks_from(ids,logits)
            checks,areas,precision=inspect_masks(masks,refs);initial_checks=checks.copy();attempts=[]
            if not all(checks.values()):
                masks,attempts=initialize_lit(sam,rgb,refs)
                checks,areas,precision=inspect_masks(masks,refs)
            mask=masks['sock'];intersection=int((mask&truth).sum())
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'))
            components=[int(v) for v in stats[1:,cv2.CC_STAT_AREA]]
            row=dict(profile=name,checks=checks,initial_checks=initial_checks,refinement_attempts=len(attempts),areas=areas,projection_precision=precision,
                visible_sock_iou=intersection/max(1,int((mask|truth).sum())),visible_sock_recall=intersection/max(1,int(truth.sum())),
                visible_sock_precision=intersection/max(1,int(mask.sum())),components_over_20px=sum(v>=20 for v in components),
                largest_component_fraction=max(components,default=0)/max(1,int(mask.sum())),
                cloth_black_fraction=float((rgb[truth].max(1)<5).mean()),cloth_mean_rgb=rgb[truth].mean(0).tolist())
            for kind,m in masks.items():Image.fromarray(m.astype('uint8')*255).save(folder/f'sam-{kind}.png')
            raw=depth.infer_image(rgb[:,:,::-1].copy(),input_size=518);depths.append(raw);images.append((rgb,masks))
            rows.append(row);write_json(folder/'lighting-evaluation.json',dict(**row,prompts=prompts));print(json.dumps(row),flush=True)
    maximum=max(float(d.max()) for d in depths);sheet=Image.new('RGB',(1600,260*len(rows)),'white');draw=ImageDraw.Draw(sheet)
    for i,(row,(rgb,masks),raw) in enumerate(zip(rows,images,depths)):
        encoded=np.clip(raw/maximum*255,0,255).astype('uint8')
        panels=[rgb,masks['sock'].astype('uint8')*255,np.where(masks['sock'],encoded,0).astype('uint8'),masks['foot'].astype('uint8')*255,np.where(masks['foot'],encoded,0).astype('uint8')]
        for j,panel in enumerate(panels):sheet.paste(Image.fromarray(panel).convert('RGB').resize((320,240)),(j*320,i*260+20))
        draw.text((5,i*260+4),f'{row["profile"]}: visible sock IoU={row["visible_sock_iou"]:.4f} / components={row["components_over_20px"]}',fill='black')
    sheet.save(a.probe/'model-comparison.jpg');write_json(a.probe/'lighting-evaluation.json',dict(rows=rows,truth_scope='Unlit rendering only for independent visibility QA, not substituted into SAM outputs',preview_depth_scale=maximum))


if __name__=='__main__':main()
