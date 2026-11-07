"""Numerical counterexamples behind the saved PhysX ray diagnostics."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from audit_compound_rays import ray_interval
from report_compound_contacts import affine_ray_interval,sphere_penetration


@pytest.fixture
def cube_planes():
    return ConvexHull(np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])).equations


def test_ray_entry_must_precede_exit(cube_planes):
    # x requires t >= 2, while y requires t <= 1. An entry-only test
    # incorrectly reports a surface at t=2 for this ray that misses the cube.
    assert ray_interval(cube_planes,np.array([3.,0.,0.]),np.array([-1.,1.,0.])) is None


def test_ray_parallel_to_an_outside_face_is_a_miss(cube_planes):
    assert ray_interval(cube_planes,np.array([3.,2.,0.]),np.array([-1.,0.,0.])) is None
    np.testing.assert_allclose(ray_interval(cube_planes,np.array([3.,0.,0.]),np.array([-1.,0.,0.])),[2.,4.])


def test_ray_inside_and_backward_intersections(cube_planes):
    np.testing.assert_allclose(ray_interval(cube_planes,np.zeros(3),np.array([1.,0.,0.])),[0.,1.])
    assert ray_interval(cube_planes,np.array([3.,0.,0.]),np.array([1.,0.,0.])) is None
    with pytest.raises(ValueError):ray_interval(cube_planes,np.zeros(3),np.zeros(3))


def test_pca_conditioning_preserves_thin_physical_vertices():
    pytest.importorskip('fcl')
    from diagnose_compound_geometry import conditioned_part,vector
    raw=np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]],dtype=float)*[.04,.02,1e-5]
    raw=raw@Rotation.from_euler('xyz',[.4,.2,-.7]).as_matrix().T+[.08,.05,.1]
    source=dict(link='right_arm/link_5',vertices=[vector(p) for p in raw])
    result=conditioned_part(source,17,'pca1')
    read=lambda field:np.array([result[field][a] for a in 'xyz'])
    vertices=np.array([[p[a] for a in 'xyz'] for p in result['vertices']])
    rotation=Rotation.from_quat([result['rotation'][a] for a in 'xyzw']).as_matrix()
    restored=(vertices*read('scale'))@rotation.T+read('position')
    np.testing.assert_allclose(restored,raw.astype(np.float32),atol=2e-8,rtol=0)
    singular=np.linalg.svd(vertices-vertices.mean(0),compute_uv=False)
    assert singular[-1]/singular[0]>.1
    assert result['index']==17


def test_affine_ray_interval_preserves_world_metres_for_nonuniform_scale(cube_planes):
    R=Rotation.from_euler('xyz',[.4,.2,-.7]).as_matrix();linear=R@np.diag([.03,.002,2e-7]);translation=np.array([.5,1.1,.7])
    origin=linear@np.array([3.,0.,0.])+translation;direction=-R[:,0]
    np.testing.assert_allclose(affine_ray_interval(cube_planes,origin,direction,linear,translation),[.06,.12],atol=1e-10)


def test_world_float_vertices_can_collapse_a_valid_thin_convex(cube_planes):
    from scipy.spatial import QhullError
    local=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])
    linear=np.diag([.03,2e-8,.02]);translation=np.array([.5,1.,.7]);world=(local@linear.T+translation).astype(np.float32)
    with pytest.raises(QhullError):ConvexHull(world)
    interval=affine_ray_interval(cube_planes,translation+[0.,2e-7,0.],np.array([0.,-1.,0.]),linear,translation)
    np.testing.assert_allclose(interval,[1.8e-7,2.2e-7],atol=1e-15)


def test_sphere_overlap_depth_uses_euclidean_corner_distance(cube_planes):
    points=np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])
    centers=np.array([[0.,0.,0.],[1.,1.,1.],[1.1,1.1,0.],[2.1,0.,0.]])
    expected=[1.2,.2,.2-np.sqrt(.02),-.1]
    np.testing.assert_allclose(sphere_penetration(points,cube_planes,centers,np.array([.2,.2,.2,1.])),expected,atol=1e-12)


def test_experimental_padding_is_outer_and_has_a_known_displacement_bound():
    pytest.importorskip('fcl')
    from diagnose_compound_geometry import conditioned_part,vector
    points=np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]],dtype=float)*[.02,.01,1e-7]
    points=points.astype(np.float32).astype(float);source=dict(link='right_arm/link_5',vertices=[vector(p) for p in points]);result=conditioned_part(source,17,'pad002')
    vertices=np.array([[p[a] for a in 'xyz'] for p in result['vertices']]);scale=np.array([result['scale'][a] for a in 'xyz']);center=np.array([result['position'][a] for a in 'xyz']);R=Rotation.from_quat([result['rotation'][a] for a in 'xyzw']).as_matrix();physical=(vertices*scale)@R.T+center;planes=ConvexHull(physical).equations
    assert np.max(points@planes[:,:3].T+planes[:,3])<1e-8
    # Support functions of a Minkowski sum differ by at most the offset box's
    # corner radius, even when its hull contains additional vertices.
    rng=np.random.default_rng(42);directions=rng.normal(size=(300,3));directions/=np.linalg.norm(directions,axis=1)[:,None];difference=(physical@directions.T).max(0)-(points@directions.T).max(0)
    assert difference.min()>-1e-8
    assert difference.max()<=result['padding_upper_bound_m']+1e-8
    assert result['padding_upper_bound_m']<1e-4
