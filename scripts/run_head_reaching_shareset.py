#!/usr/bin/env python3
"""Collect synchronized physical head stereo, then process and smoke-train ShareSet."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'artifacts/reaching-shareset-head-v2')
    ap.add_argument('--collection-python', default='/tmp/sock-diversity-venv/bin/python')
    ap.add_argument('--ml-python', default='/usr/bin/python3')
    ap.add_argument('--collection-workers', type=int, default=4)
    ap.add_argument('--perception-workers', type=int, default=6)
    a = ap.parse_args()
    out = a.output.resolve();out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR='/tmp/sock-collection-mpl', PYTHONUNBUFFERED='1')
    with (out / 'collection.log').open('a') as raw_log, (out / 'pipeline.log').open('a') as pipeline_log:
        raw = subprocess.Popen([a.collection_python, str(ROOT / 'scripts/collect_reaching_shareset.py'),
            '--head-cameras', '--output', str(out), '--python', a.collection_python,
            '--workers', str(a.collection_workers)], cwd=ROOT, env=env, stdout=raw_log, stderr=subprocess.STDOUT)
        processor = subprocess.Popen([a.collection_python, str(ROOT / 'scripts/finish_head_reaching_shareset.py'),
            '--output', str(out), '--ml-python', a.ml_python, '--workers', str(a.perception_workers)],
            cwd=ROOT, env=env, stdout=pipeline_log, stderr=subprocess.STDOUT)
        try:
            status = raw.wait()
            if status:
                processor.terminate();processor.wait()
                raise RuntimeError('Head collection failed; inspect collection.log and retain the failed episode before retrying')
            if processor.wait():raise RuntimeError('Head processing failed; inspect pipeline.log')
        finally:
            for child in (raw, processor):
                if child.poll() is None:child.terminate();child.wait()


if __name__ == '__main__':main()
