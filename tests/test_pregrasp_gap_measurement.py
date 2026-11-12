"""Known-area controls for the measured finite-jaw cloth clipping."""
import itertools
import numpy as np
import pytest
pytest.importorskip('fcl')
from sock_dressing_simulation.pregrasp_gap_measurement import measure


def jaws():
    return [np.array(list(itertools.product([-.01,.01],ys,[-.01,.01]))) for ys in [(-.015,-.01),(.01,.015)]]


def test_crossing_triangles_are_clipped_to_finite_jaw_footprint():
    # All four vertices lie outside the footprint, but the surface crosses it.
    points=np.array([[-.03,0,-.03],[.03,0,-.03],[.03,0,.03],[-.03,0,.03]])
    result=measure(points,np.array([[0,1,2],[0,2,3]]),np.ones(4)*.02,jaws(),np.array([[[-.03,0,0],[.03,0,0]]]))
    assert result['cloth_surface_area_between_fingers_mm2']==pytest.approx(400,abs=1e-6)
    assert result['material_15_30mm_band_area_between_fingers_mm2']==pytest.approx(400,abs=1e-6)
    assert result['opening_rim_length_between_fingers_mm']==pytest.approx(20,abs=1e-6)


def test_cloth_beyond_the_facing_surfaces_is_not_between_jaws():
    points=np.array([[-.005,.03,-.005],[.005,.03,-.005],[.005,.03,.005],[-.005,.03,.005]])
    result=measure(points,np.array([[0,1,2],[0,2,3]]),np.ones(4)*.02,jaws(),np.array([[[-.03,.03,0],[.03,.03,0]]]))
    assert result['cloth_surface_area_between_fingers_mm2']==0
    assert result['opening_rim_length_between_fingers_mm']==0
    assert not result['cloth_present']


def test_material_band_area_clips_faces_crossing_both_depth_boundaries():
    points=np.array([[-.01,0,-.01],[.01,0,-.01],[.01,0,.01],[-.01,0,.01]])
    result=measure(points,np.array([[0,1,2],[0,2,3]]),np.array([0.,.04,.04,0.]),jaws(),np.array([[[-.01,0,0],[.01,0,0]]]))
    assert result['cloth_surface_area_between_fingers_mm2']==pytest.approx(400,abs=1e-6)
    assert result['material_15_30mm_band_area_between_fingers_mm2']==pytest.approx(150,abs=1e-6)
