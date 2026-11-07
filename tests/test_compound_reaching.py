"""Distance signs and gradients used by the constrained reaching optimizer."""
import numpy as np
import pytest
pytest.importorskip('fcl')
import fcl
from sock_dressing_simulation.compound_reaching import distance_feature,aabb_distance

@pytest.mark.parametrize('offset,expected',[(1.2,.2),(.8,-.2)])
def test_distance_feature_gives_penetration_gradient_to_escape_collision(offset,expected):
    a=fcl.CollisionObject(fcl.Box(1,1,1));b=fcl.CollisionObject(fcl.Box(1,1,1),fcl.Transform(np.array([offset,0.,0.])))
    distance,pa,pb,gradient=distance_feature(a,b)
    assert distance==pytest.approx(expected,abs=1e-7)
    assert gradient[0]==pytest.approx(-1.,abs=1e-7)
    a.setTransform(fcl.Transform(np.array([1e-5,0.,0.])))
    derivative=(distance_feature(a,b)[0]-distance)/1e-5
    assert derivative==pytest.approx(gradient[0],abs=1e-5)


def test_aabb_distance_is_zero_for_overlap_and_euclidean_for_separation():
    a=(np.zeros(3),np.ones(3))
    assert aabb_distance(a,(np.full(3,.5),np.full(3,2.)))==0
    assert aabb_distance(a,(np.array([2.,3.,1.]),np.array([3.,4.,2.])))==pytest.approx(np.sqrt(5))

@pytest.mark.parametrize('offset',[.7,.4])
def test_mesh_dispatch_keeps_feature_normal_in_caller_order(offset):
    from sock_dressing_simulation.arm_reaching import triangle_object
    vertices=np.array([[offset,-2.,-2.],[offset,2.,-2.],[offset,0.,2.]])
    mesh=triangle_object(vertices,[[0,1,2]]);box=fcl.CollisionObject(fcl.Box(1,1,1))
    distance,pa,pb,normal=distance_feature(box,mesh)
    mesh.setTransform(fcl.Transform(np.array([1e-6,0.,0.])))
    derivative=(distance_feature(box,mesh)[0]-distance)/1e-6
    assert derivative==pytest.approx(-normal[0],abs=1e-5)


def test_translating_entire_human_below_floor_is_rejected_even_without_robot_contact():
    from sock_dressing_simulation.compound_reaching import CompoundScene
    checker=CompoundScene.__new__(CompoundScene)
    checker.update=lambda q,gap,delta:None
    checker.pairs=[];checker.floor_top=0.;checker.floor_minima={'human_full_skin':.0234,'support':.43}
    result=checker.slacks(np.zeros(7),.03,[0,-.3,0],details=True)
    assert not result['passed']
    assert result['issues'][0]['pair']==['human_full_skin','RCareWorld/Ground']
    assert result['issues'][0]['slack_m']==pytest.approx(-.2766)
    assert checker.slacks(np.zeros(7),.03,[0,0,0],details=True)['passed']


def test_smooth_profile_preserves_joint_ranges_and_bounds_velocity_and_acceleration():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/validate_compound_trajectory.py'
    spec=importlib.util.spec_from_file_location('compound_trajectory_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    points=np.array([[0.,0,0,0,0,0,0],[.2,-.1,.1,0,0,0,0],[.1,-.2,.15,0,0,0,0],[.4,0,0,0,0,0,0]])
    result=module.smooth_profile(points,np.ones(7),np.ones(7))
    assert np.all(result['q']>=points.min(0)-1e-12)
    assert np.all(result['q']<=points.max(0)+1e-12)
    assert np.allclose(result['velocity'][[0,-1]],0,atol=1e-10)
    assert np.all(np.max(abs(result['velocity']),axis=0)<=result['velocity_bound']+1e-10)
    assert np.all(np.max(abs(result['acceleration']),axis=0)<=result['acceleration_bound']+1e-10)
    assert np.all(result['velocity_bound']<=.1+1e-10)
    assert np.all(result['acceleration_bound']<=.1+1e-10)


def test_floor_clearance_gradient_acts_on_vertical_scene_translation():
    from types import SimpleNamespace
    from sock_dressing_simulation.compound_reaching import CompoundScene
    checker=CompoundScene.__new__(CompoundScene);checker.update=lambda q,gap,delta:None;checker.pairs=[];checker.floor_top=0.;checker.floor_minima={'human':.0234,'support':.43}
    checker.k=SimpleNamespace(arm_parent=np.eye(4),arm_origins=[np.eye(4)]*7,arm_axes=np.tile([0.,0.,1.],(7,1)),A=np.eye(3),b=np.zeros(3))
    values,J=checker.slacks(np.zeros(7),.03,[0,-.3,0],with_jacobian=True)
    expected=np.zeros(10);expected[8]=1.
    np.testing.assert_allclose(J[0],expected)
    np.testing.assert_allclose(J[1],np.zeros(10))


def test_convex_subdivision_preserves_volume_and_parent_containment():
    import importlib.util
    from pathlib import Path
    from scipy.spatial import ConvexHull
    path=Path(__file__).resolve().parents[1]/'scripts/partition_compound_for_cooking.py'
    spec=importlib.util.spec_from_file_location('compound_partition_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(4);points=rng.normal(size=(48,3));points/=np.linalg.norm(points,axis=1)[:,None];points*=np.array([.08,.05,.03]);parent=module.hull(points);children=module.partition(parent,limit=24);planes=ConvexHull(parent.vertices).equations
    assert len(children)>1
    assert all(len(child.vertices)<=24 for child in children)
    assert sum(child.volume for child in children)==pytest.approx(parent.volume,rel=1e-6,abs=1e-12)
    assert all((child.vertices@planes[:,:3].T+planes[:,3]).max()<1e-8 for child in children)


def test_lazy_collision_objects_follow_the_current_transform_after_skipped_queries():
    from types import SimpleNamespace
    from sock_dressing_simulation.compound_reaching import CompoundScene
    from sock_dressing_simulation.grasp_geometry import convex_geometry,HANDEDNESS
    points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    geometry,vertices,_=convex_geometry(points)
    row=dict(link='link',obj=fcl.CollisionObject(geometry),vertices=vertices)
    checker=CompoundScene.__new__(CompoundScene)
    def fk(q,gap,only):
        T=np.eye(4);T[0,3]=q[0];return {'link':T}
    checker.k=SimpleNamespace(fk=fk,A=np.eye(3),b=np.zeros(3));checker.links=['link'];checker.groups={'link':[row]};checker.group_vertices={'link':vertices};checker.group_starts={'link':np.array([0])};checker.transform_epoch=0;checker.obstacles=[]
    checker.update([1.],.04,np.zeros(3));checker.query_object(row)
    np.testing.assert_allclose(row['obj'].getTranslation(),[1.,0.,0.])
    checker.update([2.],.04,np.zeros(3))
    checker.update([3.],.04,np.zeros(3))
    checker.query_object(row)
    np.testing.assert_allclose(row['obj'].getTranslation(),[3.,0.,0.])
    expected=vertices@HANDEDNESS.T+[3.,0.,0.]
    np.testing.assert_allclose(row['bounds'],[expected.min(0),expected.max(0)])
