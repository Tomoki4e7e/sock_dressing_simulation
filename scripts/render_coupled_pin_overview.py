#!/usr/bin/env python3
"""Render recorded geometry only; this does not run or qualify a simulation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy.spatial.transform import Rotation

from report_coupled_pin_diagnostics import rows, summarize


def foot_faces(row):
    faces = []
    for geometry in row['dressing_qa']['foot_collision_safety']['foot_collider_geometry']:
        if not geometry['collision_active']:
            continue
        if geometry['shape_type'] == 'convex_skin_mesh':
            vertices = np.asarray(geometry['mesh_vertices_world'])
            indices = np.asarray(geometry['mesh_triangles']).reshape(-1, 3)
        else:
            corners = np.array([[x, y, z] for x in [-.5, .5] for y in [-.5, .5] for z in [-.5, .5]])
            vertices = Rotation.from_quat(geometry['rotation']).apply(
                (corners * geometry['size'] + geometry['center']) * geometry['scale']) + geometry['position']
            indices = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                                [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
        faces.extend(vertices[indices])
    return faces


def plotted(points):
    # Unity uses Y up; matplotlib uses Z up.
    return np.asarray(points)[..., [0, 2, 1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = rows(args.diagnostics)
    if not records:
        raise ValueError('No recorded frames')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output / '.plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    summaries = [summarize(row) for row in records]
    triangles = np.asarray(records[0]['particle_surface_triangles']).reshape(-1, 3)
    clocks = [row['dressing_qa']['foot_collision_safety']['cloth_physics_clock'] for row in records]
    step_counts = np.diff([clocks[0]['fixed_steps'] - 18] + [c['fixed_steps'] for c in clocks])
    if not np.all(step_counts == 18):
        raise ValueError('Overview expects the fixed replay: 18 steps of 20 ms per command')
    if not all(np.isclose(c['last_step_time_s'], .02, atol=1e-7) for c in clocks):
        raise ValueError('Physics step duration differs from 20 ms')
    # Nine output frames at 25 fps represent exactly 0.36 s per command.
    # Recorded end-of-command geometry is held; there is no interpolation.
    for name, azimuth in [('front', 90), ('side', 180)]:
        figure = plt.figure(figsize=(8, 6), dpi=100)
        ax = figure.add_subplot(projection='3d')
        ax.view_init(elev=15, azim=azimuth)
        ax.set_axis_off()
        scene = Poly3DCollection([], linewidth=.2)
        ax.add_collection3d(scene)
        title = figure.text(.02, .97, '', va='top', fontsize=11)
        figure.text(.02, .02, 'Recorded geometry overview: blue sock; tan foot contact shapes.\n'
                    'Camera fits cloth and foot; end poses held for 0.36 s. Supplemental to Unity.', fontsize=9)
        path = args.output / ('overview-' + name + '.mp4')
        process = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', '800x600', '-framerate', '25', '-i', '-',
            '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
        try:
            for row, summary in zip(records, summaries):
                cloth = list(plotted(np.asarray(row['particles_world'])[triangles]))
                foot = [plotted(face) for face in foot_faces(row)]
                points = np.concatenate(cloth + foot)
                low, high = points.min(axis=0), points.max(axis=0)
                center = (low + high) / 2
                span = np.maximum(high - low, .45)
                half = span / 2 + max(float(span.max()) * .04, .02)
                low, high = center - half, center + half
                ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
                ax.set_box_aspect(high - low)
                # One collection sorts cloth and foot faces together for
                # occlusion, rather than drawing either whole object on top.
                scene.set_verts(cloth + foot)
                scene.set_facecolors(['#427aa1'] * len(cloth) + ['#d9b99b'] * len(foot))
                scene.set_edgecolors(['#284f69'] * len(cloth) + ['#9b7d63'] * len(foot))
                error = summary['anchor_error_mm']
                title.set_text(f'{name.title()} | Command {summary["command"]} / {len(records)} | '
                    f'Elapsed physics {summary["command"] * .36:.2f} s\n'
                    f'Attached sides {summary["attached_sides"]} | Pin error '
                    + ('unattached' if error is None else f'{error:.2f} mm')
                    + f' | Stretch / limit {summary["maximum_normalized_stretch"]:.5f}')
                figure.canvas.draw()
                pixels = np.asarray(figure.canvas.buffer_rgba())[..., :3].tobytes()
                for _ in range(9):
                    process.stdin.write(pixels)
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError('Overview encoder failed')
        finally:
            if process.poll() is None:
                process.terminate(); process.wait()
            plt.close(figure)
    (args.output / 'overview-source.json').write_text(json.dumps({
        'source': str(args.diagnostics.resolve()), 'recorded_commands': len(records),
        'source_sha256': hashlib.sha256(args.diagnostics.read_bytes()).hexdigest(),
        'scope': 'Supplemental render of recorded particle surfaces and foot collision shapes; no simulation.',
        'seconds_per_command': .36, 'output_fps': 25, 'held_frames_per_command': 9,
        'camera_directions_world': {'front': '+Z', 'side': '-X', 'up': '+Y'},
        'camera_fit': 'Current cloth and foot contact shapes; minimum axis span 0.45 m, plus padding.',
        'repair_qualification': 'Not assessed by this visualization'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
