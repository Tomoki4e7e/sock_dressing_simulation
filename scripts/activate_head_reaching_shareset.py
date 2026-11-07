#!/usr/bin/env python3
"""Preserve original overview images and point the canonical dataset at verified head stereo."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1];sys.path.insert(0, str(ROOT))
from sock_dressing_simulation.reaching_collection import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--head', type=Path, default=ROOT / 'artifacts/reaching-shareset-head-v2')
    ap.add_argument('--canonical', type=Path, default=ROOT / 'artifacts/reaching-shareset-v1')
    ap.add_argument('--archive', type=Path, default=ROOT / 'artifacts/reaching-shareset-overview-v1-archive')
    a = ap.parse_args();head = a.head.resolve();canonical = a.canonical.absolute();archive = a.archive.absolute()
    report = json.loads((head / 'report.json').read_text())
    if not (report['completed'] and report['count'] == 12 and report.get('camera_profile') == 'head-stereo-zero-offset-v2'):
        raise ValueError('All 12 head episodes must pass before activation')
    if canonical.is_symlink() and canonical.resolve() == head:
        if not (head / 'camera-migration.json').exists():raise ValueError('Missing original-image preservation manifest')
        print('Head dataset already active');return
    if archive.exists() or canonical.is_symlink() or not canonical.is_dir():raise ValueError('Inspect canonical/archive paths before activation')
    old_report = json.loads((canonical / 'report.json').read_text())
    old_rows = {row['id']: row for row in old_report['rows']}
    baseline_path = head / 'original-overview-before-migration.json'
    baseline = ({item['path']: item['sha256'] for item in json.loads(baseline_path.read_text())['files']}
                if baseline_path.exists() else {})
    images = [];episodes = []
    for row in report['rows']:
        rel = Path('data/data_sock_reaching_v1') / row['split'] / row['id']
        ep = head / rel;old_ep = canonical / rel
        if old_rows[row['id']]['frames'] != row['frames']:raise ValueError('Original/new frame counts differ')
        meta = json.loads((ep / 'metadata.json').read_text())
        if not (meta['raw_validation_passed'] and meta['perception_passed'] and meta['reaching_learning_ready']):
            raise ValueError('Unqualified head episode')
        target = ep / 'confirmation/camera_overview_original';target.mkdir(parents=True, exist_ok=True)
        original = old_ep / 'camera_right'
        expected = {str(i) for i in range(row['frames'])}
        if {p.stem for p in original.glob('*.png')} != expected:raise ValueError('Original overview image IDs differ')
        for index in range(row['frames']):
            src = original / f'{index}.png';dst = target / src.name;digest = sha256(src)
            if baseline and baseline.get(str(src.relative_to(canonical))) != digest:
                raise ValueError('Original overview changed since the preservation baseline was recorded')
            if not dst.exists():shutil.copy2(src, dst)
            if sha256(dst) != digest:raise ValueError(f'Original overview preservation failed: {dst}')
            images.append(dict(original=str((rel / 'camera_right' / src.name)),
                preserved=str(dst.relative_to(head)), sha256=digest))
        episodes.append(dict(id=row['id'], frames=row['frames'], original_archive=str(archive / rel),
            confirmation=str(target.relative_to(head))))
        meta.update(confirmation_image_directories=['camera_overview', 'confirmation/camera_overview_original'],
            original_overview_confirmation_directory='confirmation/camera_overview_original',
            original_overview_preservation='Byte-identical copies; SHA-256 in camera-migration.json at dataset root')
        (ep / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    manifest = dict(completed=True, images=len(images), old_canonical=str(canonical),
        head_dataset=str(head), overview_archive=str(archive), episodes=episodes, files=images,
        roles=dict(camera_right='physical right head camera', camera_left='physical left head camera',
            camera_overview='new synchronized overview for confirmation only',
            original_overview='confirmation/camera_overview_original: exact byte copies of original camera_right'),
        relocation_note='Archive metadata is unchanged. Its historical reaching-shareset-v1 paths refer to this overview archive, not the newly activated head dataset.')
    # All copies are verified before changing the canonical path. Retain the entire old pipeline and models.
    (head / 'camera-migration.json').write_text(json.dumps(manifest, indent=2) + '\n')
    canonical.rename(archive)
    try:canonical.symlink_to(os.path.relpath(head, canonical.parent), target_is_directory=True)
    except BaseException:
        archive.rename(canonical)
        raise
    (archive / 'RELOCATION.json').write_text(json.dumps(dict(original_root=str(canonical), relocated_root=str(archive),
        reason='Original overview dataset and overview-trained models retained without modification',
        current_head_dataset=str(head)), indent=2) + '\n')
    report.update(canonical_dataset=str(canonical), original_overview_archive=str(archive),
        original_overview_preserved_frames=len(images), original_overview_manifest='camera-migration.json')
    (head / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    links = ''.join('<li><a href="'+html.escape(item['confirmation'])+'/0.png">'+html.escape(item['id'])+'</a></li>' for item in episodes)
    with (head / 'report.html').open('a') as stream:
        stream.write('<article><h2>元の俯瞰画像（変更せず保存）</h2><p>各経路のconfirmation/camera_overview_original/に元のcamera_rightを保持しています。'
            '<a href="camera-migration.json">全画像のSHA-256照合記録</a>。旧データ・旧モデル一式の退避先：'+html.escape(str(archive))+'</p><ul>'+links+'</ul></article>')
    print('HEAD ACTIVE', canonical, '->', head)
    print('PRESERVED ORIGINAL OVERVIEW IMAGES', len(images), 'archive', archive)


if __name__ == '__main__':main()
