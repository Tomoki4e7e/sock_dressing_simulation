"""Projection and prompt selection in the physical head camera image."""
from __future__ import annotations

import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def projected_mesh(vertices, triangles, camera, fov, size=(1280, 960)):
    width, height = size
    xyz = (np.asarray(vertices) - camera['position']) @ Rotation.from_quat(camera['quaternion']).as_matrix()
    focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
    safe = np.maximum(xyz[:, 2], 1e-6)
    xy = np.column_stack([width / 2 + xyz[:, 0] / safe * focal, height / 2 - xyz[:, 1] / safe * focal])
    faces = np.asarray(triangles).reshape(-1, 3)
    faces = faces[np.all(xyz[faces, 2] > .001, axis=1)]
    mask = np.zeros((height, width), dtype='uint8')
    polygons = [np.rint(np.clip(xy[face], -1e5, 1e5)).astype('int32') for face in faces]
    # Mesh faces need a union: fillPoly's even/odd rule for overlapping
    # polygons otherwise punches false holes through front/back faces.
    for polygon in polygons:cv2.fillConvexPoly(mask, polygon, 255)
    return mask > 0


def head_references(meta, root):
    state = json.loads((root / meta['geometry_record']['state_file']).read_text())
    plan = json.loads((root / meta['trajectory']).read_text())
    delta = np.asarray(plan['translation_m'])
    camera, fov = meta['camera_transform'], meta['camera_fov']
    sock = projected_mesh(np.asarray(state['particles_world']) + delta, state['triangles'], camera, fov)
    skin = state['source']['skin_surface']
    foot = projected_mesh(np.asarray(skin['vertices']) + delta, skin['triangles'], camera, fov)
    return dict(sock=sock, foot=foot)


def interior_point(mask):
    if not mask.any():raise ValueError('No foreground for a SAM prompt')
    distance = cv2.distanceTransform(mask.astype('uint8'), cv2.DIST_L2, 5)
    y, x = np.unravel_index(distance.argmax(), mask.shape)
    return [int(x), int(y)]


def head_prompts(rgb, references):
    r, g, b = [rgb[:, :, i].astype(float) for i in range(3)]
    sock, foot = references['sock'], references['foot']
    blue = (b > r * 1.2 + 5) & (b > g * 1.05 + 3) & (r < 150) & sock
    if not blue.any():raise ValueError('Visible sock opening not found in head image')
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blue.astype('uint8'))
    blue = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sock_positive = [interior_point(blue)]
    dark_sock = sock & (rgb.mean(axis=2) < 120)
    if dark_sock.any():sock_positive.append(interior_point(dark_sock))
    expanded = cv2.dilate(sock.astype('uint8'), np.ones((9, 9), dtype='uint8')) > 0
    visible_foot = foot & ~expanded & (rgb.mean(axis=2) > 35) & (rgb.mean(axis=2) < 220)
    if not visible_foot.any():visible_foot = foot & ~expanded
    foot_positive = [interior_point(visible_foot)]
    rows = np.indices(foot.shape)[0];ys = np.where(visible_foot)[0]
    for a, b in ((0., .4), (.6, 1.)):
        band = visible_foot & (rows >= np.quantile(ys, a)) & (rows <= np.quantile(ys, b))
        if band.any():foot_positive.append(interior_point(band))
    outside = interior_point(~(sock | foot))
    sock_negative = foot_positive + [outside]
    white_occluder = sock & (rgb.mean(axis=2) > 180)
    if white_occluder.any():sock_negative.append(interior_point(white_occluder))
    return dict(sock=dict(points=sock_positive + sock_negative, labels=[1]*len(sock_positive)+[0]*len(sock_negative)),
                foot=dict(points=foot_positive + sock_positive + [outside], labels=[1]*len(foot_positive)+[0]*(len(sock_positive)+1)))
