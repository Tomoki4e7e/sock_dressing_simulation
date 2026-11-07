#!/usr/bin/env python3
"""Audit the selected frozen-cloth surrogate and produce inspection figures."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sock-geometric-review-mpl')
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from photograph_geometric_closure import CHOICES
from sock_dressing_simulation.grasp_geometry import RightHand
from sock_dressing_simulation.geometric_closure import projected_cloth_area, material_distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/geometric-closure-photos-clean')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(exist_ok=True)
    settings = yaml.safe_load((ROOT / 'config/geometric_closure.yaml').read_text())
    kwargs = {k: settings[k] for k in ('half_extent_m', 'edge_clearance_m',
              'material_radius_m', 'band_width_m', 'topology_tolerance_m')}
    results = []
    for number, index in CHOICES:
        pose = json.loads((ROOT / f'artifacts/geometric-closure/pose-{number:02d}/report.json').read_text())
        r = next(r for r in pose['candidates'] if r['candidate_index'] == index)
        c = r['candidate']
        prior = json.loads((ROOT / f'artifacts/right-grasp-geometry/pose-{number:02d}/report.json').read_text())
        source = json.loads((ROOT / f'artifacts/grasp-cloth-refilter/pose-{number:02d}/report.json').read_text())
        state = json.loads((ROOT / source['source_state']).read_text())
        initial = json.loads((ROOT / r['initial_state_source']).read_text())
        points = np.asarray(initial['initial_particles_world'])
        faces = np.asarray(initial['triangles']).reshape(-1, 3)
        rest = np.asarray(state['source']['rest_positions'])
        depth = rest[:, 2] - rest[:, 2].min()
        sample = prior['samples'][c['sample_index']]
        distances = material_distances(rest, faces, faces[sample['face']], sample['weights'])
        hand = RightHand(ROOT / prior['settings']['urdf'])
        local = (points - c['grasp_center_unity_world_m']) @ np.asarray(c['rotation_matrix'])
        def project(gap, offset=None):
            fingers = [v for k, v in hand.vertices(gap).items()
                       if k.endswith('/left_finger_link') or k.endswith('/right_finger_link')]
            return projected_cloth_area(local if offset is None else local-offset,
                                        faces, fingers, distances, depth, **kwargs)
        gaps = []
        for gap in (.0005, .00075, .001, .00125, .0015, .002):
            q = project(gap)
            gaps.append(dict(gap_mm=gap*1000, area_mm2=q['area_m2']*1e6,
                             connected_area_mm2=q['largest_component_area_m2']*1e6))
        if abs(gaps[2]['area_mm2']-r['projection']['area_m2']*1e6)>1e-7:
            raise RuntimeError('Saved and recomputed area disagree')
        perturbations = []
        for axis in range(3):
            for sign in (-1, 1):
                offset = np.zeros(3)
                offset[axis] = sign*.00025
                q = project(.001, offset)
                perturbations.append(dict(axis='xyz'[axis], offset_mm=sign*.25,
                    area_mm2=q['area_m2']*1e6, connected_area_mm2=q['largest_component_area_m2']*1e6,
                    area_checks_pass=bool(q['area_m2']>=4e-6 and q['largest_component_area_m2']>=4e-6)))
        results.append(dict(number=number, candidate_index=index, pitch_deg=pose['pitch_deg'],
                            roll_deg=pose['roll_deg'], depth_mm=sample['depth_m']*1000,
                            opening_gap_mm=r['opening_gap_m']*1000,
                            area_mm2=r['projection']['area_m2']*1e6,
                            area_fraction=r['projection']['area_fraction'],
                            closure_skin_clearance_mm=r['closure_skin_clearance_m']*1000,
                            gap_sensitivity=gaps, tcp_perturbation_area_only=perturbations))
    all_rows = []
    for path in (ROOT / 'artifacts/geometric-closure').glob('pose-*/report.json'):
        all_rows.extend(json.loads(path.read_text())['candidates'])
    thresholds = []
    area_keys = {'projected_area', 'connected_area'}
    for threshold in (2, 4, 6, 8):
        passed = [r for r in all_rows if all(v for k, v in r['checks'].items() if k not in area_keys)
                  and min(r['projection']['area_m2'], r['projection']['largest_component_area_m2'])*1e6+1e-6>=threshold]
        thresholds.append(dict(threshold_mm2=threshold, configurations=len(passed),
                                ankle_states=len({r['number'] for r in passed})))
    (out / 'sensitivity.json').write_text(json.dumps(dict(
        candidates=results, threshold_sensitivity=thresholds,
        scope='Gap and TCP changes re-evaluate area only, not new execution plans; frozen cloth, no capture labels.'), indent=2)+'\n')
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), layout='constrained')
    for ax, (number, index), r in zip(axes[0], CHOICES, results):
        pose = json.loads((ROOT / f'artifacts/geometric-closure/pose-{number:02d}/report.json').read_text())
        row = pose['candidates'][index]
        ax.add_patch(Rectangle((-4, -4), 8, 8, fill=False, ls='--', color='.4'))
        for polygon in row['projection']['polygons_xz_m']:
            v = np.asarray(polygon['exterior'])*1000
            ax.fill(v[:, 0], v[:, 1], color='#2277bb')
            for hole in polygon['holes']:
                v = np.asarray(hole)*1000
                ax.fill(v[:, 0], v[:, 1], color='white')
        ax.plot(0, 0, 'k+')
        ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.5), aspect='equal', xlabel='Approach x (mm)',
               ylabel='Jaw width z (mm)', title=f"No.{number} / {index}\n{r['area_mm2']:.3f} mm² at 1 mm gap")
        ax.grid(alpha=.2)
    for ax, r in zip(axes[1], results):
        ax.plot([v['gap_mm'] for v in r['gap_sensitivity']],
                [v['area_mm2'] for v in r['gap_sensitivity']], 'o-')
        ax.axhline(4, color='red', ls='--')
        ax.axvline(1, color='.5', ls=':')
        ax.set(xlabel='Evaluation gap (mm)', ylabel='Projected area (mm²)',
               title='Frozen-cloth gap sensitivity')
        ax.grid(alpha=.2)
    fig.suptitle('Finite 8 x 8 mm opposing-pad window; fixed target material region\nArea is a geometric surrogate, not a physical capture measurement')
    fig.savefig(out / 'area-review.png', dpi=150)
    plt.close(fig)
    print(json.dumps(dict(candidates=results, thresholds=thresholds)), flush=True)


if __name__ == '__main__':
    main()
