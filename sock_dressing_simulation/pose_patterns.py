"""Workbook-driven, held visual sock shapes; no simulated textile calibration."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .preinsert import generate_preinsert, audit_preinsert, audit_preinsert_surface

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def validate_pattern_config(config):
    settings = config.get('pose_pattern')
    if not settings:
        return
    if not isinstance(settings, dict) or set(settings)-{'workbook', 'number', 'amplitude_m', 'above_heel_m', 'signs', 'seed', 'held_reference', 'gravity_drape', 'toe_contact'}:
        raise ValueError('Unknown pose_pattern settings')
    if config.get('circumference_fit') or config.get('preinsert'):
        raise ValueError('pose_pattern requires preinsert: null and no circumference_fit')
    if not settings.get('workbook') or type(settings.get('number', 1)) is not int or settings.get('number', 1) < 1:
        raise ValueError('A workbook and positive integer pattern number are required')
    if type(settings.get('seed', 1729)) is not int or settings.get('seed', 1729) < 0:
        raise ValueError('seed must be a nonnegative integer')
    if 'held_reference' in settings and (not isinstance(settings['held_reference'], str) or not settings['held_reference'].strip()):
        raise ValueError('held_reference must name a saved held-A directory')
    drape = settings.get('gravity_drape')
    toe = settings.get('toe_contact')
    if toe:
        if not isinstance(toe,dict) or set(toe)-{'enabled','band_width_m','support_length_m','support_clearance_m','settle_steps'} or not settings.get('held_reference'):
            raise ValueError('toe_contact requires a held reference and known settings')
        if drape and drape.get('enabled',True) and toe.get('enabled',True):
            raise ValueError('toe_contact and gravity_drape cannot both be enabled')
        values=[toe.get('band_width_m',.030),toe.get('support_length_m',.015),toe.get('support_clearance_m',.008)]
        if not np.isfinite(values).all() or not 0<values[0]<=.05 or not 0<values[1]<=.03 or not 0<values[2]<=.015:
            raise ValueError('Invalid toe_contact support or band dimensions')
        if type(toe.get('settle_steps',300)) is not int or not 100<=toe.get('settle_steps',300)<=2000:
            raise ValueError('toe_contact requires 100–2000 settling steps')
    if drape:
        if not settings.get('held_reference') or not isinstance(drape, dict) or set(drape)-{'enabled', 'pinned_length_m', 'settle_steps'}:
            raise ValueError('gravity_drape requires a held reference and known settings')
        if (type(drape.get('settle_steps', 300)) is not int or not 100 <= drape.get('settle_steps', 300) <= 1000
                or not np.isfinite(drape.get('pinned_length_m', .075)) or not 0 < drape.get('pinned_length_m', .075) < .15):
            raise ValueError('Invalid gravity drape duration or support length')
    amplitude = settings.get('amplitude_m', .015)
    height = settings.get('above_heel_m', .05)
    if not np.isfinite([amplitude, height]).all() or not .01 <= amplitude <= .02 or height <= 0:
        raise ValueError('Invalid visual deformation amplitude or above-heel height')
    signs = settings.get('signs', dict(shift=1, diagonal=1, roll=1))
    if not isinstance(signs, dict) or set(signs) != {'shift', 'diagonal', 'roll'} or any(v not in (-1, 1) for v in signs.values()):
        raise ValueError('Axis signs must be +1 or -1')


@dataclass(frozen=True)
class PosePattern:
    number: int
    sock_id: str
    within_sock: int
    stage: str
    shift_cm: float
    diagonal_deg: float
    opening: str
    wrinkles: str
    fold: str
    pitch_deg: float
    roll_deg: float
    excel_row: int

    def __post_init__(self):
        if self.stage not in 'ABCDE' or len(self.stage) != 1:
            raise ValueError('Unknown opening stage')
        if self.opening not in ('通常', 'つぶれている', '折れ・よれ'):
            raise ValueError('Unknown opening shape')
        if self.wrinkles not in ('ほぼなし', '軽いしわ', 'かかとねじれ', '全体ずれ'):
            raise ValueError('Unknown wrinkle state')
        if self.fold not in ('なし', '内巻き', '外巻き'):
            raise ValueError('Unknown cuff fold')
        if not np.isfinite([self.shift_cm, self.diagonal_deg, self.pitch_deg, self.roll_deg]).all():
            raise ValueError('Nonfinite pattern parameters')


def _sheet_rows(archive, filename, strings):
    result = []
    for row in ET.fromstring(archive.read(filename)).findall('.//s:sheetData/s:row', NS):
        cells = {}
        for cell in row:
            value = cell.find('s:v', NS)
            text = value.text if value is not None else None
            if cell.get('t') == 's' and text is not None:
                text = strings[int(text)]
            elif cell.get('t') == 'inlineStr':
                text = ''.join(cell.find('s:is', NS).itertext())
            cells[''.join(c for c in cell.get('r') if c.isalpha())] = text
        result.append((int(row.get('r')), cells))
    return result


def read_patterns(path):
    """Read cached numeric formula results with stdlib, never execute formulas.

    Reject missing angle caches rather than silently replacing workbook values.
    Handles sheet reordering and rich shared strings; ignores collection notes.
    """
    with zipfile.ZipFile(path) as archive:
        strings = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            strings = [''.join(t.text or '' for t in item.findall('.//s:t', NS))
                       for item in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        for filename in archive.namelist():
            if not filename.startswith('xl/worksheets/') or not filename.endswith('.xml'):
                continue
            rows = _sheet_rows(archive, filename, strings)
            if not any(c.get('A') == '通し番号' and c.get('K') == 'pitch(°)' for _, c in rows):
                continue
            result = []
            for row, cells in rows:
                if not str(cells.get('A', '')).isdigit():
                    continue
                try:
                    result.append(PosePattern(int(cells['A']), cells['B'], int(cells['C']),
                        cells['D'].split(':')[0].strip(), float(cells['E']), float(cells['F']),
                        cells['G'], cells['H'], cells['I'], float(cells['K']), float(cells['M']), row))
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f'Invalid workbook pattern at row {row}: {error}') from error
            if not result or len({p.number for p in result}) != len(result):
                raise ValueError('No patterns or duplicate pattern numbers')
            return result
    raise ValueError('Integrated pattern sheet not found')


def anatomical_frame(source, pitch_deg, roll_deg):
    """World vectors: pitch raises toes from horizontal; positive roll RH toes."""
    forward = np.asarray(source['toe']) - np.asarray(source['ankle'])
    forward[1] = 0
    forward /= np.linalg.norm(forward)
    up = np.array([0., 1., 0.])
    p, r = np.deg2rad([pitch_deg, roll_deg])
    direction = forward*np.cos(p) + up*np.sin(p)
    dorsal = up*np.cos(p) - forward*np.sin(p)
    dorsal = dorsal*np.cos(r) + np.cross(direction, dorsal)*np.sin(r)
    lateral = np.cross(direction, dorsal)
    return direction, dorsal, lateral


def _bump(distance):
    """One compact C2 mound; exactly zero outside its finite support."""
    q = np.maximum(0., 1.-np.asarray(distance))
    return q**4*(5.-4*q)


def _angle_distance(a, b):
    return (a-b+np.pi) % (2*np.pi)-np.pi


def local_features(pattern, amplitude_m, length_m, depth_m, heel_path_m, seed=1729, variant=0):
    """Choose 1-3 isolated events in total, retaining every requested category."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, pattern.number, variant]))
    kinds = []
    if pattern.opening != '通常': kinds.append(pattern.opening)
    if pattern.wrinkles in ('軽いしわ', 'かかとねじれ'): kinds.append(pattern.wrinkles)
    if pattern.fold != 'なし': kinds.append(pattern.fold)
    if not kinds: return []
    count = int(rng.integers(len(kinds), 4))
    choices = kinds+[kinds[int(rng.integers(len(kinds)))] for _ in range(count-len(kinds))]
    features = []
    for kind in choices:
        for _ in range(100):
            angle = float(rng.uniform(-np.pi, np.pi))
            if all(abs(_angle_distance(angle, f['angle_rad'])) > 1.05 for f in features): break
        if kind in ('内巻き', '外巻き'):
            u = 0.
        elif kind in ('つぶれている', '折れ・よれ'):
            u = float(rng.uniform(.008, .025))
        elif kind == 'かかとねじれ':
            u = float(np.clip(depth_m-heel_path_m+rng.uniform(-.015, .015), .025, length_m-.04))
        else:
            u = float(rng.uniform(.035, max(.04, length_m-.045)))
        features.append(dict(kind=kind, angle_rad=angle, u_m=u,
            amplitude_m=float(amplitude_m*rng.uniform(.75, 1.)),
            axial_support_m=float(rng.uniform(.035, .05)),
            angular_support_rad=float(rng.uniform(.70, .95))))
    return features


def _tight_base(source, pattern, clearance_m, above_heel_m, signs, slab_fraction=.65):
    """Fit a polygonal sleeve to each skin section, without a global ellipse."""
    from scipy.spatial import ConvexHull
    from .preinsert import _slab_points
    data = copy.deepcopy(source)
    rest = np.asarray(data['rest_positions'], dtype=float)
    nominal = np.median(np.linalg.norm(rest[:, :2], axis=1))
    rest[:, :2] *= .018/nominal
    data['rest_positions'] = rest.tolist()
    offset = signs['shift']*pattern.shift_cm*.01
    if pattern.wrinkles == '全体ずれ': offset += .015
    positions, report = generate_preinsert(data, pattern.stage, above_heel_m=above_heel_m,
                                            clearance_m=.003, path_offset_m=offset)
    levels = np.unique(np.round(rest[:, 2], 6))
    spacing = float(np.median(np.diff(levels)))*report['axial_initial_scale']
    surfaces = [(np.asarray(s['vertices']), np.asarray(s['triangles']).reshape(-1, 3)) for s in source['surfaces']]
    frames = []
    for index, z in enumerate(levels):
        ids = np.flatnonzero(np.isclose(rest[:, 2], z, atol=1e-6, rtol=0))
        theta = np.arctan2(rest[ids, 1], rest[ids, 0])
        cap = np.linalg.norm(rest[ids, :2], axis=1) < 1e-6
        ring = report['rings'][index]
        center = np.asarray(ring['center_world'])
        relative = positions[ids[~cap]]-center
        lateral = (relative*np.cos(theta[~cap])[:, None]).sum(axis=0)/(np.sum(~cap)*.5*ring['half_width_m'])
        cross = (relative*np.sin(theta[~cap])[:, None]).sum(axis=0)/(np.sum(~cap)*.5*ring['half_height_m'])
        lateral /= np.linalg.norm(lateral); cross /= np.linalg.norm(cross)
        axis = np.cross(cross, lateral); axis /= np.linalg.norm(axis)
        u = float(z-levels[0])*report['axial_initial_scale']
        # Fit to the tilted plane itself, instead of shearing an already tight
        # sleeve through the foot after constructing it.
        sign = 1. if np.dot(cross, source.get('pattern_dorsal', cross)) >= 0 else -1.
        angle = signs['diagonal']*sign*np.deg2rad(pattern.diagonal_deg)*np.exp(-(max(0., u-.04)/.05)**2)
        old_lateral = lateral.copy()
        lateral = np.cos(angle)*lateral+np.sin(angle)*axis
        axis = np.cos(angle)*axis-np.sin(angle)*old_lateral
        samples = np.concatenate([_slab_points(v, f, center, axis, spacing*slab_fraction) for v, f in surfaces])
        directions = np.column_stack([np.cos(theta), np.sin(theta)])
        if len(samples) >= 3:
            local = np.column_stack([(samples-center)@lateral, (samples-center)@cross])
            local = np.unique(np.round(local, 7), axis=0)
            hull = ConvexHull(local)
            shift = local[hull.vertices].mean(axis=0)
            center = center+lateral*shift[0]+cross*shift[1]
            equations = hull.equations
            bounds = -(equations[:, 2]+equations[:, :2]@shift)+clearance_m
            den = directions@equations[:, :2].T
            radii = np.min(np.divide(bounds[None, :], den,
                           out=np.full_like(den, np.inf), where=den>1e-10), axis=1)
            # Compensate the chord of the actor's finite angular sampling.
            radii = radii/np.cos(np.pi/np.sum(~cap))+.0005
        else:
            radii = np.full(len(ids), .018)
        values = center+radii[:, None]*(directions[:, :1]*lateral+directions[:, 1:]*cross)
        values[cap] = center
        positions[ids] = values
        report['rings'][index].update(center_world=center.tolist(), half_width_m=float(np.max(radii*np.abs(np.cos(theta)))),
                                      half_height_m=float(np.max(radii*np.abs(np.sin(theta)))))
        frames.append(dict(ids=ids, cap=cap, theta=theta, u=u, center=center,
                           lateral=lateral, cross=cross, axis=axis, radii=radii))
    report['clearance_m'] = clearance_m
    return positions, report, frames


def _shape(source, pattern, amplitude_m, clearance_m, above_heel_m, signs, seed=1729, variant=0):
    positions, report, frames = _tight_base(source, pattern, clearance_m, above_heel_m, signs,
                                            slab_fraction=.85 if variant >= 6 else .65)
    baseline = positions.copy()
    features = local_features(pattern, amplitude_m, frames[-1]['u'],
                              report['insertion_path_m'], report['heel_path_m'], seed, variant)
    for frame in frames:
        ids, cap, theta = frame['ids'], frame['cap'], frame['theta']
        u, center = frame['u'], frame['center']
        lateral, cross, axis = frame['lateral'], frame['cross'], frame['axis']
        radial = np.zeros(len(ids)); axial = np.zeros(len(ids)); phase = np.zeros(len(ids))
        for feature in features:
            kind = feature['kind']; amount = feature['amplitude_m']
            angular = _bump(abs(_angle_distance(theta, feature['angle_rad']))/feature['angular_support_rad'])
            along = _bump(abs(u-feature['u_m'])/feature['axial_support_m'])
            weight = angular*along
            if kind == '軽いしわ':
                radial += amount*weight
            elif kind == 'つぶれている':
                # A local indentation relative to a raised slack patch. The
                # central valley stays on the skin envelope, not inside it.
                narrow = _bump(abs(_angle_distance(theta, feature['angle_rad']))/(feature['angular_support_rad']*.45))
                radial += amount*(angular-narrow)*along
            elif kind == '折れ・よれ':
                radial += amount*weight
                axial += amount*.3*weight
            elif kind == 'かかとねじれ':
                phase += min(.20, amount/max(.02, float(np.mean(frame['radii']))))*weight
                radial += amount*.7*weight
            else:
                # Turn only the selected rim sector. The reserve needed by an
                # inward fold is local too, never a larger full circumference.
                fold_u = max(0., 1-u/(2*amount))
                if kind == '内巻き':
                    radial += amount*1.25*_bump(max(0., u-.025)/.05)*angular
                    radial -= amount*fold_u**2*angular
                else:
                    radial += amount*fold_u**2*angular
                axial += (-amount*fold_u**2+2*u*fold_u)*angular
        # Twist moves material along the fitted outline, so it does not rotate
        # an ellipse through the skin. Periodic linear interpolation is convex.
        active = ~cap; order = np.argsort(theta[active])
        angles = theta[active][order]
        original = baseline[ids[active]][order]
        target = (theta+phase+np.pi) % (2*np.pi)-np.pi
        loop_angles = np.r_[angles[-1]-2*np.pi, angles, angles[0]+2*np.pi]
        loop = np.vstack([original[-1], original, original[0]])
        values = np.column_stack([np.interp(target, loop_angles, loop[:, j]) for j in range(3)])
        direction = np.cos(target)[:, None]*lateral+np.sin(target)[:, None]*cross
        values += radial[:, None]*direction+axial[:, None]*axis
        values[cap] = center
        positions[ids] = values
    cuff = positions[report['opening_particle_indices']]
    _, _, vh = np.linalg.svd(cuff-cuff.mean(axis=0))
    normal = vh[-1]
    if np.dot(normal, report['opening_inward_normal_world']) < 0: normal *= -1
    perimeter = lambda p: float(np.linalg.norm(p-np.roll(p, 1, axis=0), axis=1).sum())
    # Native actor indices are not guaranteed to follow circumferential order.
    cuff_frame = frames[0]; order = np.argsort(cuff_frame['theta'])
    changed = np.linalg.norm(positions-baseline, axis=1)>1e-7
    report.update(pattern=asdict(pattern), amplitude_m=amplitude_m,
        cuff_center_world=cuff.mean(axis=0).tolist(), opening_inward_normal_world=normal.tolist(),
        opening_shape=pattern.opening, wrinkles=pattern.wrinkles, cuff_fold=pattern.fold,
        baseline_cuff_perimeter_m=perimeter(baseline[cuff_frame['ids']][order]),
        cuff_perimeter_m=perimeter(cuff[order]), local_features=features,
        local_feature_count=len(features), changed_particle_fraction=float(changed.mean()),
        maximum_local_displacement_m=float(np.linalg.norm(positions-baseline, axis=1).max()),
        random_seed=seed, feature_variant=variant, fit_method='skin_section_polygon_3mm_shell',
        motion_mode='held_visual_shape', physical_settling_validated=False,
        sock_identity_affects_geometry=False)
    return positions, report


def generate_pattern(source, pattern, edges, triangles, rest_lengths, *, amplitude_m=.015,
                     above_heel_m=.05, signs=None, seed=1729):
    if not np.isfinite(amplitude_m) or not .01 <= amplitude_m <= .02:
        raise ValueError('Visual deformation amplitude must be between 1 and 2 cm')
    signs = dict(shift=1, diagonal=1, roll=1) if signs is None else dict(signs)
    if set(signs) != {'shift', 'diagonal', 'roll'} or any(v not in (-1, 1) for v in signs.values()):
        raise ValueError('Axis signs must be +1 or -1')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    attempts = []
    # Safety may relocate a patch, but may never inflate the entire garment
    # by centimetres to accommodate a local detail.
    for clearance in (.0025, .0035, .005, .006):
        for variant in range(8):
            positions, report = _shape(source, pattern, amplitude_m, clearance, above_heel_m, signs, seed, variant)
            body = audit_preinsert(positions, source, edges, rest_lengths)
            if not body['geometry_ok']:
                attempts.append(dict(clearance_m=clearance, variant=variant, body=body)); continue
            surface = audit_preinsert_surface(positions, source, edges, triangles)
            attempts.append(dict(clearance_m=clearance, variant=variant, body=body, surface=surface))
            if surface['surface_ok']:
                report.update(geometry_audit=body, surface_audit=surface, attempts=attempts, signs=signs)
                return positions, report
            if not report['local_features']: break
    raise ValueError(f'Pattern {pattern.number}: no nonintersecting tight shape: {attempts}')


def initialize_pattern(environment, pattern, *, amplitude_m=.015, above_heel_m=.05, signs=None, seed=1729, held_reference=None, gravity_drape=None, toe_contact=None):
    """Install an audited shape and hold it explicitly for visual inspection."""
    if held_reference is not None:
        from .held_ankle_poses import initialize_held_ankle_pattern
        return initialize_held_ankle_pattern(environment, pattern, held_reference, signs, gravity_drape=gravity_drape, toe_contact=toe_contact)
    from .circumference_fit import current_pose_hull
    signs = dict(shift=1, diagonal=1, roll=1) if signs is None else signs
    attr = environment.sock_cloth
    if attr is None or environment._cloth_contact_rollout_started:
        raise RuntimeError('Initialize visual patterns before a grasp/contact rollout')
    attr.set_pattern_held(False)
    attr.set_pattern_foot_pose(pattern.pitch_deg, signs['roll']*pattern.roll_deg)
    environment._env.step(simulate=False)
    attr.request_preinsert_source(); attr.request_fit_skin(); attr.request_particles()
    environment._env.step(simulate=False)
    source = copy.deepcopy(attr.data['sock_preinsert_source'])
    source['pattern_dorsal'] = list(attr.data['sock_pattern_pose']['dorsal_direction'])
    source['surfaces'] = []
    for key in ('sock_fit_skin', 'sock_fit_calf_skin'):
        vertices, faces = current_pose_hull(attr.data[key])
        source['surfaces'].append(dict(vertices=vertices.tolist(), triangles=faces.tolist()))
    positions, report = generate_pattern(source, pattern, attr.data['particle_edges'],
        attr.data['particle_surface_triangles'], attr.data['particle_rest_edge_lengths'],
        amplitude_m=amplitude_m, above_heel_m=above_heel_m, signs=signs, seed=seed)
    attr.initialize_preinsert(positions, report['opening_inward_normal_world'])
    environment._env.step(simulate=False)
    attr.set_pattern_held(True); attr.request_particles()
    environment._env.step(simulate=False)
    error = float(np.max(np.linalg.norm(attr.particles()-positions, axis=1)))
    if error > 5e-6 or not attr.data.get('sock_pattern_hold', {}).get('enabled'):
        raise RuntimeError('Visual pattern installation failed native readback')
    pose = copy.deepcopy(attr.data['sock_pattern_pose'])
    measured_pitch = np.degrees(np.arcsin(np.clip(pose['toe_direction'][1], -1, 1)))
    if abs(measured_pitch-pattern.pitch_deg) > .05:
        raise RuntimeError(f'Native foot pitch mismatch: {measured_pitch} vs {pattern.pitch_deg}')
    report.update(native_pose=pose, measured_pitch_deg=float(measured_pitch),
                  native_hold=copy.deepcopy(attr.data['sock_pattern_hold']), maximum_readback_error_m=error)
    environment._preinsert_observation_pending = True
    environment._pose_pattern_report = report
    environment._pose_pattern_source = source
    return report
