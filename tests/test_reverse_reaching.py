"""Regression checks for bounded angular sampling of planned edges."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('fcl')
spec=importlib.util.spec_from_file_location('reverse_planner',Path(__file__).resolve().parents[1]/'scripts/search_reverse_reaching.py');planner=importlib.util.module_from_spec(spec);spec.loader.exec_module(planner)

def test_edge_sampling_includes_endpoints_and_bounds_every_joint_increment():
 a=np.zeros(7);b=np.array([.1,-.2,.005,0,0,0,.023]);q=planner.edge_samples(a,b,.03)
 np.testing.assert_allclose(q[0],a);np.testing.assert_allclose(q[-1],b)
 assert np.max(abs(np.diff(q,axis=0)))<=.03+1e-12

def test_stationary_edge_remains_finite():
 a=np.ones(7);q=planner.edge_samples(a,a,.01)
 assert np.isfinite(q).all()
 np.testing.assert_allclose(q,np.tile(a,(2,1)))
