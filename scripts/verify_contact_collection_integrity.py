#!/usr/bin/env python3
"""Check accepted episode PNGs and immutable input bindings; support incremental runs."""
import argparse,hashlib,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(status_path):
    status=json.loads(status_path.read_text());episode=status_path.parent/'episode'
    if not status.get('finished') or not status.get('passed'):return None
    meta_path=episode/'metadata.json';meta=json.loads(meta_path.read_text());binding=dict(metadata_sha256=sha(meta_path),status_sha256=sha(status_path))
    output=episode/'integrity.json'
    assert meta['contact_pregrasp_learning_ready'] and meta['raw_validation_passed']
    assert meta['human_collision_model']=='planning_skin' and meta['drive_profile']=='paired_calibrated'
    assert meta['final_closure_performed'] is False and meta['cloth_frozen']
    assert sha(ROOT/meta['trajectory'])==meta['trajectory_sha256']
    assert sha(ROOT/meta['shared_initial_configuration'])==meta['shared_initial_sha256']
    for proof in meta['planning_proofs'].values():assert sha(ROOT/proof['path'])==proof['sha256']
    audit=meta['exact_human_contact'];assert audit['passed'] and audit['robot_skin_pairs_ignored']==0 and audit['proxy_robot_pairs_still_enabled']==0
    # Recheck referenced inputs even when the expensive image scan is cached.
    if output.exists():
        previous=json.loads(output.read_text())
        if previous.get('binding')==binding and previous.get('passed'):return previous
    expected={str(i) for i in range(meta['frames'])};count=0
    for camera in ('left','right','overview'):
        images=list((episode/('camera_'+camera)).glob('*.png'));assert {p.stem for p in images}==expected
        for path in images:
            with Image.open(path) as image:
                assert image.mode=='RGB' and list(image.size)==meta['image_size'];image.verify()
            count+=1
    result=dict(id=status['id'],passed=True,binding=binding,png_files_verified=count,
                frames_per_camera=meta['frames'],image_size=meta['image_size'],initial_sha256=meta['shared_initial_sha256'])
    output.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--workers',type=int,default=3);args=p.parse_args();out=args.dataset.resolve()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:rows=[r for r in pool.map(verify,sorted((out/'attempts').glob('*/status.json'))) if r]
    assert len({r['initial_sha256'] for r in rows})<=1
    result=dict(expected=70,verified_episodes=len(rows),all_70_verified=len(rows)==70,total_png_files_verified=sum(r['png_files_verified'] for r in rows),rows=rows)
    (out/'integrity-summary.json').write_text(json.dumps(result,indent=2)+'\n');print('INTEGRITY',len(rows),'episodes',result['total_png_files_verified'],'PNGs',flush=True)


if __name__=='__main__':main()
