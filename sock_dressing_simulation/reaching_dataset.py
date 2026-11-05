"""Shared-start and spatial diversity contracts for reaching demonstrations."""
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_shared_initial(row, initial):
    """Reject per-condition changes to arms, jaws, torso, or scene translation."""
    for key in ('left_q_rad', 'right_q_rad', 'translation_m'):
        if not np.array_equal(np.asarray(row[key]), np.asarray(initial[key])):
            raise ValueError('Shared initial configuration differs: ' + key)
    for key in ('left_gap_m', 'right_gap_m', 'left_foot_floor_allowance_m'):
        if row[key] != initial[key]:
            raise ValueError('Shared initial configuration differs: ' + key)
    if row['moving_side'] != 'left':
        raise ValueError('This dataset requires a moving left arm')
    if not np.array_equal(row['start_q_rad'], initial['left_q_rad']):
        raise ValueError('Trajectory does not start at the shared left arm posture')
    if row['initial_opening_gap_m'] != initial['left_gap_m']:
        raise ValueError('Trajectory initial jaw gap differs')
    torso = initial['torso_joints_rad']
    if row['torso_joints_rad'] != torso:
        raise ValueError('Shared torso configuration differs')


def verify_dataset_inputs(record, root):
    """Fail closed when the workbook, shared start, state, or samples changed."""
    for item in record.get('dataset_inputs', []):
        if file_sha256(Path(root) / item['file']) != item['sha256']:
            raise ValueError('Dataset input changed: ' + item['file'])
    if record.get('shared_initial_configuration'):
        import yaml
        initial = yaml.safe_load((Path(root) / record['shared_initial_configuration']).read_text())
        assert_shared_initial(record, initial)


def spatial_route_distance(first, second, count=101):
    """RMS TCP separation at equal normalized arc length, ignoring timing."""
    def sample(points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite Nx3 TCP positions')
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if lengths[-1] <= 1e-12:
            raise ValueError('A stationary path has no reaching diversity')
        keep = np.r_[True, np.diff(lengths) > 1e-12]
        return np.column_stack([np.interp(np.linspace(0, 1, count), lengths[keep] / lengths[-1], points[keep, j]) for j in range(3)])
    return float(np.sqrt(np.mean(np.sum((sample(first) - sample(second))**2, axis=1))))


def load_grasp_samples(record, root):
    path = record.get('samples_file', f"artifacts/bulged-cuff/pose-{record['number']:02d}/samples.json")
    return json.loads((Path(root) / path).read_text())


def load_endpoint_samples(probe,state,root):
    """Keep dense endpoint indices tied to the exact material-face samples."""
    if not probe.get('samples_file'):
        from .grasp_geometry import candidate_samples
        return candidate_samples(state)
    path=Path(root)/probe['samples_file']
    if file_sha256(path)!=probe['samples_sha256']:raise ValueError('Endpoint samples changed')
    samples=json.loads(path.read_text());points=np.asarray(state['particles_world']);faces=np.asarray(state['triangles']).reshape(-1,3)
    if not isinstance(samples,list) or not samples:raise ValueError('Empty endpoint samples')
    for index,sample in enumerate(samples):
        weights=np.asarray(sample['weights']);face=int(sample['face']);position=np.asarray(sample['position'])
        if sample['index']!=index or not 0<=face<len(faces):raise ValueError('Endpoint sample index differs')
        if weights.shape!=(3,) or position.shape!=(3,) or not np.isfinite(weights).all() or weights.min()< -1e-5 or abs(weights.sum()-1)>1e-8:
            raise ValueError('Invalid material-face sample')
        if not np.allclose(position,weights@points[faces[face]],rtol=0,atol=1e-9):raise ValueError('Endpoint sample moved off its material face')
    return samples
