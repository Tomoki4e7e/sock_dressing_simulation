#!/usr/bin/env python3
"""Render actual cloth, skin, finite hand hulls, contact witnesses and rim."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/contact-pregrasp-mpl')
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
from scipy.spatial import ConvexHull
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from search_centered_reaching import URDF
from sock_dressing_simulation.grasp_geometry import RightHand
from sock_dressing_simulation.common_opening_approach import inventory, load_state
from sock_dressing_simulation.contact_pregrasp import opening_edges, material_band


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'artifacts/contact-pregrasp-start-search')
    ap.add_argument('--ids', nargs='*')
    args = ap.parse_args()
    out = args.output.resolve()
    initial = yaml.safe_load((out/'candidate-initial.yaml').read_text())
    delta = np.asarray(initial['translation_m'])
    hand = RightHand(URDF)
    rows = [r for r in inventory(ROOT) if not args.ids or r['id'] in args.ids]
    (out/'figures').mkdir(exist_ok=True)
    rendered = []
    for row in rows:
        path = out/'routes'/f'{row["id"]}.json'
        route = json.loads(path.read_text()) if path.exists() else {}
        if not route.get('passed'):
            continue
        state = load_state(ROOT, row)
        points = np.asarray(state['particles_world'])+delta
        faces = np.asarray(state['triangles'], dtype=int).reshape(-1, 3)
        rest = np.asarray(state['source']['rest_positions'])
        depth = rest[:, 2]-rest[:, 2].min()
        skin = state['source']['skin_surface']
        skin_v = np.asarray(skin['vertices'])+delta
        skin_f = np.asarray(skin['triangles'], dtype=int).reshape(-1, 3)
        band_v, band_f = material_band(points, faces, depth, .015, .030)
        edges = opening_edges(faces, depth)
        fig = plt.figure(figsize=(14, 6), constrained_layout=True)
        for index, (elevation, azimuth) in enumerate(((20, -65), (15, 20))):
            ax = fig.add_subplot(1, 2, index+1, projection='3d')
            ax.add_collection3d(Poly3DCollection(skin_v[skin_f], facecolor='#bcb4aa', alpha=.16, edgecolor='none'))
            ax.add_collection3d(Poly3DCollection(points[faces], facecolor='#80bca3', alpha=.12, edgecolor='none'))
            ax.add_collection3d(Poly3DCollection(band_v[band_f], facecolor='#34a36e', alpha=.5, edgecolor='none'))
            ax.add_collection3d(Line3DCollection(points[edges], colors='black', linewidths=1.4))
            all_points = [points[depth <= .030]]
            for side, color in (('left', '#2778bd'), ('right', '#d66a28')):
                endpoint = route['terminal'][side]
                c = route['final_contact_checks'][side]
                p = np.asarray(c['position_m'])+delta
                R = np.asarray(c['rotation'])
                for vertices in hand.vertices(endpoint['opening_gap_m']).values():
                    v = vertices@R.T+p
                    if index == 0:
                        all_points.append(v)
                    ax.add_collection3d(Poly3DCollection(v[ConvexHull(v).simplices], facecolor=color, alpha=.24, edgecolor=color, linewidth=.15))
                for contact, mark in (('cloth', 'o'), ('foot', '^')):
                    w = np.asarray(c['contacts'][contact]['witnesses_m'])+delta
                    ax.scatter(*w.T, color=color, marker=mark, s=28, depthshade=False)
                start = np.asarray(initial[side+'_tcp_world_m'])
                if index == 0:
                    ax.scatter(*start, color=color, marker='x', s=65, label=side+' shared start')
                    all_points.append(start[None, :])
            bounds = np.concatenate(all_points)
            center = (bounds.min(0)+bounds.max(0))/2
            half = np.max(np.ptp(bounds, axis=0))/2+.015
            ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), zlim=(center[2]-half, center[2]+half),
                   xlabel='World X (m)', ylabel='World Y (m, up)', zlabel='World Z (m)')
            ax.set_box_aspect((1, 1, 1))
            ax.set_title('Whole-hand placement' if index == 0 else 'Claw contacts and opening rim', fontsize=10)
            ax.view_init(elev=elevation, azim=azimuth, vertical_axis='y')
            if index == 0:
                ax.legend(fontsize=8)
        contacts = route['final_contact_checks']
        distances = ' / '.join(f'{s}: cloth {contacts[s]["contacts"]["cloth"]["distance_m"]*1000:.3f}, foot {contacts[s]["contacts"]["foot"]["distance_m"]*1000:.3f} mm' for s in ('left', 'right'))
        fig.suptitle(row['id']+' | Open contact goals, no grasp closure\n'+distances, fontsize=12)
        fig.savefig(out/'figures'/f'{row["id"]}.png', dpi=140)
        plt.close(fig)
        rendered.append(row['id'])
    available=[]
    for row in inventory(ROOT):
        path=out/'routes'/f'{row["id"]}.json'
        if path.exists()and json.loads(path.read_text()).get('passed')and (out/'figures'/f'{row["id"]}.png').exists():
            available.append(row['id'])
    (out/'figures'/'index.json').write_text(json.dumps(dict(rendered=available,updated_ids=rendered,
        initial_pair_index=initial['source_candidate_pair_index'],
        initial_left_q_rad=initial['left_q_rad'],initial_right_q_rad=initial['right_q_rad']),indent=2)+'\n')
    print('Rendered', len(rendered), 'paired-contact goals')


if __name__ == '__main__':
    main()
