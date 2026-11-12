#!/usr/bin/env python3
"""Repeat a saved pregrasp experiment with excluded episodes and fresh weights."""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
import traceback
from pathlib import Path

import train_contact_pregrasp as training


def prepare(previous, source, exclusions, out, epochs):
    baseline = json.loads((previous / 'manifest.json').read_text())
    config = json.loads((previous / 'configuration.json').read_text())
    assert Path(baseline['source']).resolve() == source
    assert training.digest(source / 'dataset-index.json') == baseline['source_index_sha256']
    excluded_ids = exclusions.read_text().split()
    assert excluded_ids and len(excluded_ids) == len(set(excluded_ids)), 'Invalid exclusion list'
    excluded = set(excluded_ids)
    assert excluded <= {r['id'] for r in baseline['rows']}, 'Unknown exclusion IDs'
    rows = [copy.deepcopy(r) for r in baseline['rows'] if r['id'] not in excluded]
    groups = {s: {(r['pitch_deg'], r['roll_deg']) for r in rows if r['split'] == s}
              for s in ('train', 'validation', 'test')}
    assert all(groups.values())
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test']
                or groups['validation'] & groups['test'])
    assert all({r['stage'] for r in rows if r['split'] == s} == set('ABCDE') for s in groups)

    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    records = []
    for row in rows:
        episode = Path(row['path'])
        episode.relative_to(source)
        report_path = previous / 'cache' / f'{row["id"]}.json'
        report = json.loads(report_path.read_text())
        assert report['id'] == row['id'] and report['split'] == row['split']
        assert report['source_sha256'] == row['source_sha256']
        for name, expected in row['source_sha256'].items():
            assert training.digest(episode / name) == expected, f'Changed source: {episode / name}'
        npz_path = previous / 'cache' / f'{row["id"]}.npz'
        with training.np.load(npz_path) as data:
            frames = data['frames']
            assert training.np.array_equal(frames, training.np.arange(0, row['frames'], 2))
            assert len(report['frame_rgb_sha256']) == len(frames)
            for frame, expected in zip(frames, report['frame_rgb_sha256']):
                assert training.digest(episode / f'camera_right/{frame}.png') == expected
            assert all(training.np.isfinite(data[k]).all() for k in data.files)
        hashes = {}
        for path in (npz_path, report_path):
            target = cache / path.name
            expected = training.digest(path)
            if not target.exists():
                shutil.copy2(path, target)
            assert training.digest(target) == expected, f'Changed cache: {target}'
            hashes[path.suffix] = expected
        records.append(dict(id=row['id'], split=row['split'], cache_sha256=hashes))
        print('VERIFIED_CACHE', row['id'], flush=True)

    original_config = copy.deepcopy(config)
    config['hyperparameters']['num_epochs'] = epochs
    differences = {k for k in config['hyperparameters']
                   if config['hyperparameters'][k] != original_config['hyperparameters'][k]}
    assert differences <= {'num_epochs'}
    counts = {s: sum(r['split'] == s for r in rows) for s in groups}
    manifest = {**baseline, 'rows': rows, 'counts': counts,
                'previous_run': str(previous), 'excluded_ids': excluded_ids,
                'exclusion_file': str(exclusions), 'exclusion_sha256': training.digest(exclusions),
                'split_policy': 'Preserve previous split for every retained episode; exclude from all splits'}
    training.write_json(out / 'manifest.json', manifest)
    training.write_json(out / 'configuration.json', config)
    shutil.copy2(exclusions, out / 'excluded-episodes.txt')
    for name in ('configuration.json', 'manifest.json'):
        shutil.copy2(previous / name, out / f'previous-{name}')
    training.write_json(out / 'verification.json', dict(
        passed=True, previous_run=str(previous), fresh_initialization=True,
        excluded_ids=excluded_ids, counts=counts, split_assignment_unchanged=True,
        physical_groups_disjoint=True, all_stages_present_in_each_split=True,
        configuration_changed_keys=['hyperparameters.num_epochs'],
        source_signals_and_retained_rgb_hashes_verified=True,
        cache_reused_without_normalization=True, normalization_refit_on_retained_train_only=True,
        training_script_sha256=training.digest(Path(training.__file__)), rows=records))
    return rows, config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous-run', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--exclude-file', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=5000)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    assert args.epochs > 0
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        if (out / 'model/SARNN_latest.pth').exists() and not args.resume:
            raise RuntimeError('Existing weights; use a new output directory or --resume')
        if not training.torch.cuda.is_available():
            raise RuntimeError('CUDA is required; no CPU fallback')
        if args.resume:
            manifest = json.loads((out / 'manifest.json').read_text())
            config = json.loads((out / 'configuration.json').read_text())
            verification = json.loads((out / 'verification.json').read_text())
            assert verification['training_script_sha256'] == training.digest(Path(training.__file__))
            assert manifest['source'] == str(args.source.resolve())
            assert manifest['previous_run'] == str(args.previous_run.resolve())
            assert manifest['exclusion_sha256'] == training.digest(args.exclude_file.resolve())
            assert config['hyperparameters']['num_epochs'] == args.epochs
            rows = manifest['rows']
        else:
            training.write_json(out / 'progress.json', dict(completed=False, phase='verifying_cache'))
            rows, config = prepare(args.previous_run.resolve(), args.source.resolve(),
                                   args.exclude_file.resolve(), out, args.epochs)
            shutil.copy2(Path(training.__file__), out / 'training-program.py')
            shutil.copy2(Path(__file__), out / 'rerun-program.py')
        training.torch.set_num_threads(2)
        training.cv2.setNumThreads(1)
        seed = config['training']['seed']
        training.torch.manual_seed(seed)
        training.torch.cuda.manual_seed_all(seed)
        training.np.random.seed(seed)
        random.seed(seed)
        print('START_TRAINING', json.dumps(dict(
            epochs=args.epochs, fresh_initialization=not args.resume,
            counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'validation', 'test')},
            gpu=training.torch.cuda.get_device_name(), torch=training.torch.__version__)), flush=True)
        training.train(rows, out, config, resume=args.resume)
    except BaseException as error:
        training.write_json(out / 'failure.json', dict(
            failed=True, error=repr(error), traceback=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    main()
