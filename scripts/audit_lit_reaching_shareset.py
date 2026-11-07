#!/usr/bin/env python3
"""Reopen every saved lit stream and independently verify shape and provenance."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.head_camera_profile import head_references
from sock_dressing_simulation.perception_audit import projection_precision
from sock_dressing_simulation.reaching_collection import sha256
from sock_dressing_simulation.learning_lighting import PROFILES

STREAMS = {'camera_right': 'RGB', 'camera_left': 'RGB', 'camera_overview': 'RGB',
           'camera_depth': 'L', 'camera_right_mask/sock_mask': 'L',
           'camera_right_mask/foot_mask': 'L', 'depth_mask/sock_depth': 'L',
           'depth_mask/foot_depth': 'L', 'qa/visible_sock': 'L'}


def overlap(mask, truth):
    return float((mask & truth).sum() / max(1, int((mask | truth).sum())))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'artifacts/reaching-shareset-lit-v3')
    a = ap.parse_args(); out = a.output.resolve(); cv2.setNumThreads(1)
    collection = json.loads((out/'collection.json').read_text())
    if sha256(Path(collection['source_collection'])) != collection['source_collection_sha256']:
        raise ValueError('Original twelve-route manifest changed')
    original=json.loads(Path(collection['source_collection']).read_text())
    if [(r['id'],r['split'],r['frames']) for r in collection['rows']] != [(r['id'],r['split'],r['frames']) for r in original['rows']]:
        raise ValueError('Adopted routes, split or frame counts changed')
    cal = json.loads((out/'data/data_sock_reaching_v1/train/depth.json').read_text())
    if (cal['depth_min'] != 0 or cal['calibration_version'] != 'full-training-frames-v3'
            or cal['lighting_profile'] != 'balanced' or cal['training_frames'] != 4542):
        raise ValueError('Wrong depth calibration')
    if len(cal['samples']) != 8 or any(s['split'] != 'train' or s['number'] not in (10,13,25,38) for s in cal['samples']):
        raise ValueError('Depth calibration includes held-out data')
    if abs(cal['depth_max']-cal['training_maximum']*256/255) > 1e-10:
        raise ValueError('Depth upper headroom changed')
    if cal['training_maximum'] != max(s['maximum'] for s in cal['samples']):
        raise ValueError('Training maximum differs from cached estimates')
    if sha256(Path('/home/cm-pc/catkin_ws/src/DepthAnythingModel-V2/checkpoints/depth_anything_v2_vits.pth')) != cal['checkpoint_sha256']:
        raise ValueError('Depth checkpoint changed')
    fitted={Path(s['episode']).name:s for s in cal['samples']}
    if set(fitted)!={r['id'] for r in collection['rows'] if r['split']=='train'}:
        raise ValueError('Depth calibration does not cover exactly the training episodes')

    def audit(row):
        ep = Path(row['path']); meta = json.loads((ep/'metadata.json').read_text())
        source = Path(meta['source_episode']); previous = json.loads((source/'metadata.json').read_text())
        if row['split']!=meta['split'] or meta['split']!=previous['split']:
            raise ValueError('Episode split changed')
        p = json.loads((ep/'perception/validation.json').read_text()); n = meta['frames']
        if not (meta['lighting_profile'] == 'balanced' and meta['raw_validation_passed']
                and meta['render_state_alignment_passed'] and meta['reaching_learning_ready']
                and p['passed'] and not p.get('depth_pending', True)
                and meta['depth_calibration_version'] == cal['calibration_version']):
            raise ValueError(f'Incomplete episode: {ep}')
        if meta['learning_ready'] or meta['dynamic_grasp_validated'] or meta['rendering_is_dynamics_replay']:
            raise ValueError('Unsupported dynamics/model claim')
        if any(key in meta for key in ('head_isolated_repaired_frames','perception_repaired_frames')):
            raise ValueError('Obsolete annotations from the previous image processing remain')
        lighting=meta['lighting']['actual']
        if any(meta['lighting'].get(key)!=value for key,value in PROFILES['balanced'].items()):
            raise ValueError('Episodes use different lighting parameters')
        if (lighting['zero_normals'] != 0 or lighting['physical_particles'] != 1057
                or not lighting['normals_separated'] or lighting['fill_lights'] != 2):
            raise ValueError('Lighting/normal correction was not applied')
        if any(abs(lighting[key]-target)>1e-6 for key,target in
               (('ambient',.2),('sun_intensity',.65),('shadow_strength',.25))):
            raise ValueError('Native light intensities differ from the requested profile')
        if sha256(source/'metadata.json') != meta['source_metadata_sha256']:
            raise ValueError('Source metadata changed')
        for name, digest in meta['source_signal_sha256'].items():
            if sha256(ep/name) != digest or sha256(source/name) != digest:
                raise ValueError(f'Teaching signals changed: {name}')
        if meta['geometry_record'] != previous['geometry_record'] or meta['dataset_inputs'] != previous['dataset_inputs']:
            raise ValueError('Geometry/source selection changed')
        for item in meta['dataset_inputs']:
            if sha256(ROOT/item['file']) != item['sha256']:raise ValueError('Source geometry input changed')
        if sha256(ROOT/meta['trajectory']) != meta['trajectory_sha256']:
            raise ValueError('Source trajectory changed')
        source_snapshot = out/'processing-sources'/(p['mask_processing_source_sha256']+'.py')
        if sha256(source_snapshot) != p['mask_processing_source_sha256']:
            raise ValueError('Processing source snapshot changed')
        if p['depth_checkpoint_sha256'] != cal['checkpoint_sha256']:
            raise ValueError('Depth estimates use a different checkpoint')
        if row['split']=='train':
            samples=fitted[ep.name]
            if samples['count']!=n or len(samples['rows'])!=n:raise ValueError('Training calibration frame count differs')
            for index,item in enumerate(samples['rows']):
                if item['frame']!=index or sha256(ep/f'camera_right/{index}.png')!=item['rgb_sha256']:
                    raise ValueError('Training RGB changed after depth calibration')
        expected = {str(i) for i in range(n)}
        for folder in STREAMS:
            if {f.stem for f in (ep/folder).glob('*.png')} != expected:
                raise ValueError(f'Stream IDs differ: {folder}')
        poses = [json.loads(line) for line in (ep/'camera_poses.jsonl').read_text().splitlines()]
        old_poses = [json.loads(line) for line in (source/'camera_poses.jsonl').read_text().splitlines()]
        frame_rows=list(csv.DictReader((source/'frames.csv').open()))
        if len(poses) != n or len(p['rows']) != n:raise ValueError('Frame count differs')
        for i, pose in enumerate(poses):
            if pose['frame'] != i or pose['step'] != old_poses[i]['step'] or abs(pose['baseline_m']-.084) > 1e-4:
                raise ValueError('Camera frame/clock/baseline differs')
            if pose['step']!=int(frame_rows[i]['step']) or pose['source_time_s']!=float(frame_rows[i]['time_s']):
                raise ValueError('Render/source frame clock differs')
            if pose['joint_error_rad'] > 1e-6 or pose['joint_error_m'] > 1e-7 or pose['cloth_error_m'] > 2e-5:
                raise ValueError('Recorded pose changed during render')
            for side in ('right','left'):
                c = pose['cameras'][side]; err = pose['source_camera_pose_errors'][side]
                if c['parent_link'] != f'head/see3cam_{side}/camera_color_frame' or np.linalg.norm(c['local_position']) > 1e-6:
                    raise ValueError('Camera mount changed')
                if np.max(abs((np.asarray(c['local_rotation'])+180)%360-180)) > 1e-4:
                    raise ValueError('Camera rotation offset changed')
                if err['position_m'] > 3e-5 or err['rotation_deg'] > .003:raise ValueError('Source camera pose changed')
                old=old_poses[i]['cameras'][side]
                if np.linalg.norm(np.asarray(c['position'])-old['position'])>3e-5:
                    raise ValueError('Saved camera position differs from source')
                q=np.asarray(c['quaternion']);q=q/np.linalg.norm(q)
                old_q=np.asarray(old['quaternion']);old_q=old_q/np.linalg.norm(old_q)
                if np.rad2deg(2*np.arccos(np.clip(abs(q@old_q),0,1)))>.003:
                    raise ValueError('Saved camera rotation differs from source')
        refs = head_references(meta,ROOT); before=[]; after=[]; old_parts=[]; new_parts=[]
        saturation = dict(sock=0.,foot=0.); precision_min = dict(sock=1.,foot=1.)
        def read(folder,i):
            with Image.open(ep/folder/f'{i}.png') as im:
                if im.size != (1280,960) or im.mode != STREAMS[folder]:raise ValueError('Image format differs')
                return np.array(im)
        for i in range(n):
            images = {name:read(name,i) for name in STREAMS}
            views = [images[name] for name in ('camera_right','camera_left','camera_overview')]
            if any(cv2.meanStdDev(v)[1].max() < 1 for v in views):raise ValueError('Blank view')
            if np.array_equal(views[0],views[1]) or np.array_equal(views[0],views[2]):raise ValueError('Duplicated view')
            depth=images['camera_depth']; masks={}; truth=images['qa/visible_sock']>0
            for kind in ('sock','foot'):
                values=images[f'camera_right_mask/{kind}_mask']
                if ((values!=0)&(values!=255)).any():raise ValueError('Nonbinary SAM mask')
                mask=values>0; masks[kind]=mask
                if int(mask.sum()) != p['rows'][i]['areas'][kind]:raise ValueError('Stored mask area changed')
                masked=images[f'depth_mask/{kind}_depth']
                if not np.array_equal(masked,np.where(mask,depth,0)) or not masked.any():raise ValueError('Masked depth differs')
                fraction=float((depth[mask]==255).mean()); saturation[kind]=max(saturation[kind],fraction)
                if fraction > .02:raise ValueError(f'Target depth saturates: {ep.name}/{i}/{kind}: {fraction}')
                precision=projection_precision(mask,refs[kind],2)['tolerant']
                precision_min[kind]=min(precision_min[kind],precision)
                if precision < .95 or abs(precision-p['rows'][i]['projection_precision_2px'][kind])>1e-12:
                    raise ValueError('Projection precision fails or changed')
            if (masks['sock']&masks['foot']).any():raise ValueError('Masks overlap')
            iou=overlap(masks['sock'],truth)
            if iou < .90 or abs(iou-p['rows'][i]['visible_sock_iou'])>1e-12:raise ValueError('SAM misses visible cloth')
            with Image.open(source/f'camera_right_mask/sock_mask/{i}.png') as im:old=np.array(im)>0
            old_iou=overlap(old,truth)
            if abs(old_iou-p['rows'][i]['previous_visible_sock_iou'])>1e-12:raise ValueError('Before comparison changed')
            for mask,parts in ((old,old_parts),(masks['sock'],new_parts)):
                _,_,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'))
                parts.append(sum(int(v)>=20 for v in stats[1:,cv2.CC_STAT_AREA]))
            before.append(old_iou);after.append(iou)
        print('STORED LIT AUDIT PASSED',ep.name,flush=True)
        return dict(id=ep.name,split=row['split'],frames=n,minimum_projection_precision_2px=precision_min,
                    maximum_target_depth_upper_saturation_fraction=saturation,
                    before_visible_iou=before,after_visible_iou=after,before_components=old_parts,after_components=new_parts)

    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(audit,collection['rows']))
    oldroot=Path(collection['source_collection']).parent
    migration=json.loads((oldroot/'camera-migration.json').read_text());preserved=0
    for item in migration['files']:
        if sha256(out/item['preserved'])!=item['sha256'] or sha256(Path(migration['overview_archive'])/item['original'])!=item['sha256']:
            raise ValueError('Original overview changed')
        preserved+=1
    before=np.concatenate([r['before_visible_iou'] for r in results]);after=np.concatenate([r['after_visible_iou'] for r in results])
    summary=lambda x:dict(mean=float(x.mean()),minimum=float(x.min()),p10=float(np.quantile(x,.1)),median=float(np.median(x)))
    result=dict(passed=True,episodes=len(results),frames=len(after),decoded_images=len(after)*9,
                original_overview_copies_verified=preserved,source_signals_byte_identical=True,
                before_visible_sock_iou=summary(before),after_visible_sock_iou=summary(after),
                metric_scope='IoU with independently rendered visible simulated cloth at the same pose; real reference is qualitative only',
                model_accuracy_validated=False,dynamic_grasp_validated=False,
                audit_source_sha256=sha256(__file__),rows=results)
    (out/'stored-data-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ALL STORED LIT DATA PASSED',result['frames'],result['before_visible_sock_iou'],result['after_visible_sock_iou'],flush=True)


if __name__=='__main__':main()
