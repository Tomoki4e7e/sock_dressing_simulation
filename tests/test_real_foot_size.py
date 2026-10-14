import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from adjust_real_foot_size import foot_measurement
from search_real_foot_size import coupled_translation_step


def test_ankle_cut_excludes_wider_lower_leg():
    mask = np.zeros((400, 240), dtype=bool)
    mask[10:100, 20:220] = True
    mask[100:300, 80:160] = True
    measured = foot_measurement(mask, [120, 100], [120, 280])
    assert measured['width_px'] == pytest.approx(80)
    assert measured['length_px'] == pytest.approx(200)
    assert measured['width_length_complete']


def test_cropped_ankle_cannot_pass_as_complete_foot():
    mask = np.zeros((250, 200), dtype=bool)
    mask[:200, 60:140] = True
    measured = foot_measurement(mask, [100, -50], [100, 170])
    assert measured['length_px'] == pytest.approx(250)
    assert not measured['ankle_visible']
    assert measured['outline_truncated']
    assert not measured['width_length_complete']


def test_rotated_axis_measures_transverse_width():
    yy, xx = np.mgrid[:400, :400]
    axis = np.array([.6, .8])
    normal = np.array([.8, -.6])
    points = np.stack([xx - 100., yy - 80.], axis=-1)
    length = points @ axis
    width = points @ normal
    mask = (length >= 0) & (length < 200) & (width >= -40) & (width < 40)
    measured = foot_measurement(mask, [100, 80], np.array([100, 80]) + axis * 180)
    assert measured['length_px'] == pytest.approx(200, abs=1)
    assert measured['width_px'] == pytest.approx(80, abs=1)
    assert measured['width_length_complete']


def test_coupled_height_depth_step_reduces_both_size_errors():
    # A larger Y step increases both dimensions; Z offsets that width change.
    # Independent coordinate clipping would increase width error from .06 to .26.
    jacobian = np.array([[5., -2.5], [5., -5.]])
    error = np.array([.06, .7])
    step = coupled_translation_step(jacobian, error, np.zeros(2))
    assert np.all(np.abs(error + jacobian @ step) < np.abs(error))
    assert np.max(np.abs(step)) <= .08


def test_coupled_step_preserves_direction_at_depth_bound():
    jacobian = np.eye(2)
    origin = np.array([0., .18])
    step = coupled_translation_step(jacobian, np.array([-.05, -.1]), origin) - origin
    assert step == pytest.approx([.01, .02])
