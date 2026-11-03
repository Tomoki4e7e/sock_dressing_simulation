#!/usr/bin/env python3
"""Historical visual-scene preflight, not an audit of the new toe-contact mode.

Never labels a geometric point a feasible grasp.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    dtype = np.dtype([('normal', '<f4', 3), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    if len(data) != 84 + count * 50:
        raise ValueError('Expected binary STL')
    return np.frombuffer(data, dtype=dtype, offset=84, count=count)['vertices'].astype(float)


def section(triangles, x):
    points = []
    for triangle in triangles:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[i], triangle[j]
            if (a[0] - x) * (b[0] - x) < 0:
                points.append(a + (b-a) * ((x-a[0])/(b[0]-a[0])))
    if not points:
        raise ValueError('No finger section at requested grasp center')
    points = np.asarray(points)
    return dict(x_m=float(x), bounds_min_m=points.min(0).tolist(),
                bounds_max_m=points.max(0).tolist(),
                width_m=float(np.ptp(points[:, 1])),
                thickness_envelope_m=float(np.ptp(points[:, 2])))


def band_distances(points, edges, opening):
    graph = [[] for _ in points]
    for a, b in edges:
        length = float(np.linalg.norm(points[a]-points[b]))
        graph[a].append((int(b), length)); graph[b].append((int(a), length))
    distances = np.full(len(points), np.inf)
    distances[opening] = 0
    queue = [(0., int(i)) for i in opening]
    heapq.heapify(queue)
    while queue:
        distance, a = heapq.heappop(queue)
        if distance > distances[a]:
            continue
        for b, length in graph[a]:
            candidate = distance + length
            if candidate < distances[b]:
                distances[b] = candidate
                heapq.heappush(queue, (candidate, b))
    return distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, default=ROOT/'artifacts/sock-pose-pattern/held-gravity/01-S01/visual-state.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/right-grasp-preflight')
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    saved_report = json.loads(args.state.with_name('report.json').read_text())
    points = np.asarray(state['particles_world'])
    rest = np.asarray(state['source']['rest_positions'])
    edges = np.asarray(state['edges']).reshape(-1, 2)
    opening = np.flatnonzero(np.isclose(rest[:, 2], rest[:, 2].min(), atol=1e-6, rtol=0))
    distance = band_distances(points, edges, opening)
    band = np.flatnonzero(distance <= .030 + 1e-9)
    # A second definition is recorded because current surface and material
    # distances differ on stretched cloth. Neither uses world-axis projection.
    material_band = np.flatnonzero(rest[:, 2]-rest[:, 2].min() <= .030 + 1e-9)

    urdf_path = ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf'
    urdf = ET.parse(urdf_path).getroot()
    finger = urdf.find("link[@name='right_gripper/right_finger_link']/collision/geometry/mesh")
    mesh_path = urdf_path.parent/finger.attrib['filename']
    scale = np.fromstring(finger.attrib.get('scale', '1 1 1'), sep=' ')
    triangles = load_stl(mesh_path)*scale
    tip_joint = urdf.find("joint[@name='right_gripper/right_finger_tip_joint']/origin")
    nominal_tip = np.fromstring(tip_joint.attrib['xyz'], sep=' ')[0]
    physical_tip = float(triangles[:, :, 0].max())
    actual_section = section(triangles, physical_tip-.010)
    nominal_section = section(triangles, nominal_tip-.010)
    # At this axial section: right inner y=.010+q/2+z_min,
    # left inner y=-.010-q/2-z_min. This is a local envelope,
    # not a flat pad fit or a whole-jaw collision test.
    z_min = actual_section['bounds_min_m'][2]
    q_50 = .050-.020-2*z_min
    joint = urdf.find("joint[@name='right_gripper/finger_joint']/limit")
    q_limits = [float(joint.attrib[k]) for k in ('lower', 'upper')]
    drape = saved_report.get('post_hold_gravity_drape', {})
    pinned_length = drape.get('pinned_material_length_m', 0.)
    fixed_band = material_band[rest[material_band, 2]-rest[:, 2].min() <= pinned_length+1e-9]
    blockers = []
    if drape.get('mesh_vertices_are_followers') or drape.get('native_mesh_finite_mass_particles') == 0:
        blockers.append('Current saved scene uses follower mesh vertices, not native dynamic cloth for jaw contact.')
    if pinned_length >= .030:
        blockers.append('Entire 30 mm material band is inside the supported/fixed 75 mm region.')
    fit_path = ROOT/'artifacts/sock-pose-pattern/preinsert-fitted/A/report.json'
    fit = json.loads(fit_path.read_text())
    result = dict(status='physical_search_stopped_at_preflight',
        state=str(args.state.resolve()), state_sha256=hashlib.sha256(args.state.read_bytes()).hexdigest(),
        initial_state_selection='Historical held-gravity/01-S01 visual scene; no live simulator capture and no toe-contact mode audit.',
        requested=dict(maximum_inner_gap_m=.050, tip_inset_m=.010, band_width_m=.030),
        finger=dict(mesh=str(mesh_path.resolve()), bounds_min_m=triangles.min((0, 1)).tolist(),
                    bounds_max_m=triangles.max((0, 1)).tolist(), nominal_tip_x_m=float(nominal_tip),
                    physical_tip_x_m=physical_tip, tip_difference_m=physical_tip-float(nominal_tip),
                    physical_tip_inset_section=actual_section, nominal_tip_inset_section=nominal_section,
                    q_for_50mm_section_envelope_m=q_50, q_limits_m=q_limits,
                    gap_within_urdf_limits=bool(q_limits[0] <= q_50 <= q_limits[1])),
        band=dict(definition='Shortest edge-graph distance on current cloth from opening; discrete vertex approximation to surface geodesic.',
                  opening_count=len(opening), current_surface_vertex_count=len(band),
                  current_surface_vertex_indices=band.tolist(),
                  material_coordinate_vertex_count=len(material_band),
                  material_coordinate_vertex_indices=material_band.tolist(),
                  material_band_fixed_vertex_count=len(fixed_band),
                  vertex_distances_m=[dict(index=int(i), distance_m=float(distance[i])) for i in band]),
        current_physics=dict(model=drape.get('physics_model'),
                             native_mesh_finite_mass_particles=drape.get('native_mesh_finite_mass_particles'),
                             pinned_material_length_m=pinned_length),
        alternative_fitted_state=dict(path=str(fit_path),
            nonfoot_colliders_disabled=fit['final'].get('nonfoot_colliders_disabled'),
            physical_foot_contact_gap_m=fit['final'].get('physical_foot_contact_gap_m')),
        blockers=blockers,
        feasible_grasp_count=None,
        not_evaluated=['Whole gripper collision and closure sweep', 'Right arm IK and reachability',
                       'Other cloth layers and double-sided contact', 'Physical closure, slip and holding'],
        note='Band vertices are search-domain samples, not certified grasp candidates. Thickness is a section envelope, not a proof of infeasibility.')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    skin = state['source']['skin_surface']
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    sv = np.asarray(skin['vertices']); sf = np.asarray(skin['triangles']).reshape(-1, 3)
    ax.add_collection3d(Poly3DCollection(sv[sf], facecolor='gray', edgecolor='none', alpha=.12))
    ax.scatter(*points.T, s=3, color='lightsteelblue', label='Saved cloth')
    ax.scatter(*points[band].T, s=14, color='orange', label='30 mm current-surface band (NOT feasible grasps)')
    ax.scatter(*points[opening].T, s=16, color='red', label='Opening')
    center=points.mean(0); span=np.ptp(points,axis=0).max()*.65
    ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(center[2]-span,center[2]+span)
    ax.set_xlabel('Unity X [m]');ax.set_ylabel('Unity Y [m]');ax.set_zlabel('Unity Z [m]')
    ax.set_box_aspect((1,1,1)); ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Preflight only: saved 01-S01, physics search blocked')
    fig.tight_layout();fig.savefig(args.output/'band.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(report=str(args.output/'report.json'), current_band_vertices=len(band),
                         material_band_vertices=len(material_band), finger=actual_section,
                         q_50_m=q_50, blockers=blockers), indent=2))


if __name__ == '__main__':
    main()
