#!/usr/bin/env python3
"""Resume the adopted-route collection, perception, audit and ShareSet smoke run."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from collect_reaching_shareset import split_for
from sock_dressing_simulation.reaching_collection import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/reaching-shareset-v1')
    ap.add_argument('--collection-python', default='/tmp/sock-diversity-venv/bin/python')
    ap.add_argument('--ml-python', default='/usr/bin/python3')
    ap.add_argument('--collection-workers', type=int, default=3)
    ap.add_argument('--perception-workers', type=int, default=6)
    ap.add_argument('--perception-threads', type=int, default=2)
    ap.add_argument('--training-epochs', type=int, default=1)
    a = ap.parse_args();out = a.output.resolve();out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               MPLCONFIGDIR='/tmp/sock-collection-mpl', PYTHONUNBUFFERED='1')
    rows = json.loads((ROOT / 'artifacts/diverse-left-reaching/corrected-collection/manifest.json').read_text())['rows']
    episodes = {r['id']: out / 'data/data_sock_reaching_v1' / split_for(r['number']) / r['id'] for r in rows}
    def command(cmd, log, check=True):
        with log.open('w') as stream:
            return subprocess.run(cmd, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=check)
    def collect(ids=None):
        cmd = [a.collection_python, str(ROOT / 'scripts/collect_reaching_shareset.py'), '--output', str(out),
               '--workers', str(a.collection_workers), '--python', a.collection_python]
        if ids:cmd += ['--ids'] + ids
        command(cmd, out / ('pilot-raw.log' if ids else 'collection.log'))
    def audit_perception(episode):
        cmd = [a.ml_python, str(ROOT / 'scripts/audit_reaching_perception.py'), str(episode)]
        result = command(cmd, out / (episode.name + '-perception-audit.log'), check=False)
        if result.returncode:
            command([a.ml_python, str(ROOT / 'scripts/repair_reaching_perception.py'), str(episode)],
                    out / (episode.name + '-perception-repair.log'))
            command(cmd, out / (episode.name + '-perception-audit.log'))
    def perception(episode, sample=False):
        report = episode / ('pilot-perception-512' if sample else 'perception') / 'validation.json'
        if report.exists():
            previous = json.loads(report.read_text())
            if not sample and previous.get('processing_profile') == 'roi-v2':
                audit_perception(episode)
                previous = json.loads(report.read_text())
            if (previous['passed'] and previous.get('sam_non_overlap_masks')
                    and previous.get('processing_profile') == 'roi-v2'
                    and previous.get('processing_source_sha256') == sha256(ROOT / 'scripts/process_reaching_shareset.py')
                    and previous.get('sam_image_size') == 512
                    and previous.get('angle_sha256') == sha256(episode / 'angle.csv')):
                return
        cmd = [a.ml_python, str(ROOT / 'scripts/process_reaching_shareset.py'), '--output', str(out),
               '--episode', str(episode), '--threads', str(a.perception_threads)]
        if sample:cmd.append('--sample-only')
        print('PERCEPTION START', episode.name, 'sample' if sample else 'full', flush=True)
        result = command(cmd, out / (episode.name + ('-pilot.log' if sample else '-perception.log')), check=sample)
        if not sample:
            if not report.exists():result.check_returncode();raise RuntimeError('Perception produced no full report')
            audit_perception(episode)
        print('PERCEPTION DONE', episode.name, 'sample' if sample else 'full', flush=True)
    pilot_ids = ['pose-10-held-A-v00-trajectory', 'pose-38-held-A-v00-trajectory']
    collect(pilot_ids)
    for key in pilot_ids:perception(episodes[key], sample=True)
    print('PILOT GATE PASSED', flush=True)
    # Separate Unity processes and CPU model workers have independent episodes.
    with ThreadPoolExecutor(max_workers=a.perception_workers) as pool:
        raw_log = (out / 'collection.log').open('w')
        raw = subprocess.Popen([a.collection_python, str(ROOT / 'scripts/collect_reaching_shareset.py'),
            '--output', str(out), '--workers', str(a.collection_workers), '--python', a.collection_python],
            env=env, cwd=ROOT, stdout=raw_log, stderr=subprocess.STDOUT)
        pending = set(episodes);futures = {}
        try:
            while pending:
                for key in sorted(pending):
                    metadata = episodes[key] / 'metadata.json'
                    if metadata.exists() and json.loads(metadata.read_text()).get('raw_validation_passed'):
                        futures[key] = pool.submit(perception, episodes[key]);pending.remove(key)
                for future in futures.values():
                    if future.done():future.result()
                if raw.poll() is not None:
                    if raw.returncode:raise RuntimeError(f'Raw collection failed; inspect {out / "collection.log"}')
                    if pending:raise RuntimeError('Collection ended with missing qualified episodes')
                if pending:time.sleep(10)
            for future in futures.values():future.result()
            if raw.wait():raise RuntimeError('Collection failed')
        finally:
            raw_log.close()
            if raw.poll() is None:raw.terminate();raw.wait()
    print('ALL 12 EPISODES QUALIFIED', flush=True)
    share = ROOT.parent / 'ShareSet'
    config = share / 'share_folder/configs/reaching_simulation.yaml'
    if out != ROOT / 'artifacts/reaching-shareset-v1':
        import yaml
        payload = yaml.safe_load(config.read_text())
        payload['dataset'].update(root=str(out / 'data'), manifest=str(out / 'data/dataset_reaching.yaml'))
        config = out / 'training-config.yaml';config.write_text(yaml.safe_dump(payload))
    for representative in (True, False):
        name = 'model-representative-smoke' if representative else 'model-all12-smoke'
        cmd = [a.ml_python, str(share / 'share_folder/src/main/SAMDAMSARNN_learning_reaching.py'),
               '--config', str(config), '--output', str(out / name), '--epochs', str(a.training_epochs)]
        if representative:cmd.append('--representative-only')
        print('TRAINING START', name, flush=True);command(cmd, out / (name + '.log'))
        print('TRAINING DONE', name, flush=True)
    results = []
    for row in rows:
        ep = episodes[row['id']];meta = json.loads((ep / 'metadata.json').read_text())
        report = json.loads((ep / 'perception/validation.json').read_text())
        raw_qa = json.loads((ep / 'raw-validation.json').read_text())
        results.append(dict(id=row['id'], number=row['number'], split=split_for(row['number']), frames=meta['frames'],
            duration_s=raw_qa['duration_s'], raw_passed=raw_qa['passed'], perception_passed=report['passed'],
            repaired_frames=len(report.get('repaired_frames', [])),
            reaching_learning_ready=meta['reaching_learning_ready'], dynamic_grasp_validated=False,
            path=str(ep), angle_sha256=sha256(ep / 'angle.csv'), torque_sha256=sha256(ep / 'torque.csv'),
            review=str((ep / 'perception/review-audited.jpg').relative_to(out))))
    final = dict(completed=True, count=len(results), total_frames=sum(r['frames'] for r in results),
        rows=results, scope='Frozen-cloth reaching and geometric closure only', dynamic_grasp_validated=False,
        model_accuracy_validated=False, training='Representative and all-route one-epoch smoke checks',
        scripts={str(p.relative_to(ROOT.parent)):sha256(p) for p in
            [Path(__file__), ROOT / 'scripts/collect_reaching_shareset.py', ROOT / 'scripts/process_reaching_shareset.py',
             ROOT / 'sock_dressing_simulation/reaching_collection.py', share / 'share_folder/src/data/reaching.py',
             ROOT / 'scripts/audit_reaching_perception.py', ROOT / 'sock_dressing_simulation/perception_audit.py',
             ROOT / 'scripts/repair_reaching_perception.py',
             share / 'share_folder/src/models/reaching_trainer.py',
             share / 'sam2_realtime/sam2/sam2_camera_predictor.py', share / 'sam2_realtime/sam2/modeling/sam2_base.py']})
    (out / 'report.json').write_text(json.dumps(final, indent=2) + '\n')
    with (out / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    cards = ''.join('<article><h2>' + html.escape(r['id']) + '</h2><p>' + html.escape(
        f"{r['split']} / {r['frames']} frames / raw={r['raw_passed']} / perception={r['perception_passed']}") +
        '</p><a href="' + r['review'] + '"><img src="' + r['review'] + '"></a></article>' for r in results)
    (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>ShareSet reaching collection</title>'
        '<style>body{font:16px sans-serif;margin:2rem;background:#f4f6fa;color:#172435}article{background:white;padding:1rem;margin:1rem 0}img{width:100%;max-width:1100px}</style>'
        '<h1>ShareSet reaching collection: 12 routes</h1><p>Frozen-cloth reaching and geometric closure. '
        'No deformation, retention, pulling, or trained-policy success claim.</p><p>' +
        f"{final['total_frames']} synchronized frames. 8 train / 2 validation / 2 test episodes.</p>" +
        '<p>Review crops show the SAM input region. Projected-mesh support uses a 2px image raster tolerance; '
        'isolated failed frames may be re-seeded with SAM2 at 1024 resolution. '
        'Original masks, prompts, hashes and strict QA reports are retained. Native collision/grasp tolerances are unchanged.</p>'
        '<p><a href="report.json">JSON report</a> · <a href="episodes.csv">Episode table</a> · '
        '<a href="data/dataset_reaching.yaml">Dataset manifest</a></p>' + cards)
    print('PIPELINE COMPLETE', final['count'], final['total_frames'], flush=True)


if __name__ == '__main__':main()
