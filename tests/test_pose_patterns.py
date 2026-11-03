from dataclasses import replace
from pathlib import Path
import zipfile

import numpy as np
import pytest

from sock_dressing_simulation.pose_patterns import read_patterns, anatomical_frame, _shape
from sock_dressing_simulation.preinsert import generate_preinsert
from test_preinsert import source

WORKBOOK = Path(__file__).resolve().parents[1]/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx'


def test_workbook_uses_final_integrated_angles_and_all_conditions():
    patterns = read_patterns(WORKBOOK)
    assert [p.number for p in patterns] == list(range(1, 51))
    assert {p.sock_id for p in patterns} == {'S01', 'S02', 'S03', 'S04', 'S05'}
    assert {p.stage for p in patterns} == set('ABCDE')
    assert {p.opening for p in patterns} == {'通常', 'つぶれている', '折れ・よれ'}
    assert {p.fold for p in patterns} == {'なし', '内巻き', '外巻き'}
    assert {p.wrinkles for p in patterns} == {'ほぼなし', '軽いしわ', 'かかとねじれ', '全体ずれ'}
    first = patterns[0]
    assert (first.pitch_deg, first.roll_deg, first.shift_cm, first.diagonal_deg) == (50, -30, -2.5, -2)
    assert first.excel_row == 4
    assert patterns[12].pitch_deg == 30


def test_missing_cached_angle_is_rejected(tmp_path):
    target = tmp_path/'missing.xlsx'
    with zipfile.ZipFile(WORKBOOK) as original, zipfile.ZipFile(target, 'w') as modified:
        for name in original.namelist():
            contents = original.read(name)
            if name == 'xl/worksheets/sheet2.xml':
                import xml.etree.ElementTree as ET
                root = ET.fromstring(contents)
                ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                cell = root.find('.//s:c[@r="K4"]', ns)
                cell.remove(cell.find('s:v', ns))
                contents = ET.tostring(root)
            modified.writestr(name, contents)
    with pytest.raises(ValueError, match='row 4'):
        read_patterns(target)


def test_pitch_zero_is_dorsal_up_and_positive_pitch_raises_toes(source):
    forward, dorsal, lateral = anatomical_frame(source, 0, 0)
    assert np.allclose(dorsal, [0, 1, 0])
    assert forward[1] == 0
    tilted, tilted_up, _ = anatomical_frame(source, 40, 0)
    assert np.isclose(tilted[1], np.sin(np.deg2rad(40)))
    assert abs(np.dot(tilted, tilted_up)) < 1e-12
    rolled, rolled_up, _ = anatomical_frame(source, 40, 30)
    assert np.allclose(rolled, tilted)
    assert np.dot(np.cross(tilted_up, rolled_up), tilted) > 0


def test_shift_moves_cuff_on_anatomical_path_without_changing_stage(source):
    _, normal = generate_preinsert(source, 'B')
    _, shifted = generate_preinsert(source, 'B', path_offset_m=.02)
    assert np.isclose(shifted['insertion_path_m']-normal['insertion_path_m'], .02)
    assert shifted['stage'] == normal['stage']


def test_sock_ids_are_labels_without_size_or_material_changes(source):
    p = read_patterns(WORKBOOK)[0]
    a, _ = _shape(source, p, .015, .015, .05, dict(shift=1, diagonal=1, roll=1))
    b, _ = _shape(source, replace(p, sock_id='S05'), .015, .015, .05, dict(shift=1, diagonal=1, roll=1))
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()


def test_unknown_condition_cannot_silently_render_normal():
    with pytest.raises(ValueError, match='opening'):
        replace(read_patterns(WORKBOOK)[0], opening='未定義')


@pytest.mark.parametrize('change', [dict(number=0), dict(number=True), dict(amplitude_m=.03),
                                   dict(above_heel_m=0), dict(signs=dict(shift=0)), dict(unknown=1)])
def test_invalid_pattern_config_fails_before_scene_creation(change):
    from sock_dressing_simulation.pose_patterns import validate_pattern_config
    settings = dict(workbook='patterns.xlsx', number=1)
    settings.update(change)
    with pytest.raises(ValueError): validate_pattern_config(dict(pose_pattern=settings))


def test_visual_and_physical_initialization_cannot_be_mixed():
    from sock_dressing_simulation.pose_patterns import validate_pattern_config
    config = dict(pose_pattern=dict(workbook='patterns.xlsx'), circumference_fit=dict(hold_steps=500))
    with pytest.raises(ValueError): validate_pattern_config(config)


def test_tight_unwrinkled_opening_tracks_skin_size(source):
    # Synthetic foot section is 70 x 50 mm: skin perimeter is 240 mm.
    # A few millimetres of padding must not recreate a 400-500 mm opening.
    p = replace(read_patterns(WORKBOOK)[0], stage='A', shift_cm=0, diagonal_deg=0,
                opening='通常', wrinkles='ほぼなし', fold='なし')
    _, report = _shape(source, p, .015, .0025, .05, dict(shift=1, diagonal=1, roll=1))
    assert .240 < report['baseline_cuff_perimeter_m'] < .280
    assert report['local_feature_count'] == 0
    assert report['changed_particle_fraction'] == 0


def test_defects_have_finite_support_and_repeatable_random_locations(source):
    p = read_patterns(WORKBOOK)[0]
    a, r = _shape(source, p, .015, .0025, .05, dict(shift=1, diagonal=1, roll=1), seed=11)
    b, repeated = _shape(source, p, .015, .0025, .05, dict(shift=1, diagonal=1, roll=1), seed=11)
    c, changed = _shape(source, p, .015, .0025, .05, dict(shift=1, diagonal=1, roll=1), seed=12)
    assert np.array_equal(a, b)
    assert r['local_features'] == repeated['local_features']
    assert not np.array_equal(a, c)
    assert 1 <= r['local_feature_count'] <= 3
    assert 0 < r['changed_particle_fraction'] < .25
    # A single compact bump cannot produce a continuing periodic wave train.
    from sock_dressing_simulation.pose_patterns import _bump
    samples = _bump(np.linspace(0, 2, 101))
    assert samples[0] == 1
    assert np.all(np.diff(samples) <= 0)
    assert np.all(samples[50:] == 0)


def test_each_requested_kind_is_retained_with_at_most_three_events():
    from sock_dressing_simulation.pose_patterns import local_features
    p = replace(read_patterns(WORKBOOK)[0], opening='つぶれている', wrinkles='かかとねじれ', fold='内巻き')
    events = local_features(p, .015, .3, .15, .25)
    assert len(events) == 3
    assert {e['kind'] for e in events} == {'つぶれている', 'かかとねじれ', '内巻き'}


def test_fold_is_local_instead_of_turning_the_entire_rim(source):
    p = replace(read_patterns(WORKBOOK)[0], stage='A', shift_cm=0, diagonal_deg=0,
                opening='通常', wrinkles='ほぼなし', fold='外巻き')
    shape, r = _shape(source, p, .015, .0025, .05, dict(shift=1, diagonal=1, roll=1))
    baseline, _ = _shape(source, replace(p, fold='なし'), .015, .0025, .05, dict(shift=1, diagonal=1, roll=1))
    cuff = r['opening_particle_indices']
    assert np.any(np.linalg.norm(shape[cuff]-baseline[cuff], axis=1) > .001)
    assert np.count_nonzero(np.linalg.norm(shape[cuff]-baseline[cuff], axis=1) < 1e-8) >= 4
    assert r['changed_particle_fraction'] < .25


def test_invalid_random_seed_is_rejected():
    from sock_dressing_simulation.pose_patterns import validate_pattern_config
    with pytest.raises(ValueError, match='seed'):
        validate_pattern_config(dict(pose_pattern=dict(workbook='patterns.xlsx', seed=-1)))
