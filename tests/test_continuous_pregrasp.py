from types import SimpleNamespace
import numpy as np
import pytest
pytest.importorskip('fcl')
from scipy.spatial import ConvexHull
from sock_dressing_simulation.continuous_pregrasp import (SweptPolicy,eroded_vertices,capsule_triangles,
    certify_interval,MotionBounds,state_vector,trajectory_intervals)


def toy_scene(distance):
    def update(x):scene.x=x
    scene=SimpleNamespace(policy=SweptPolicy(),pairs=[('a','b',0.)],motion=np.ones((1,16)),update=update,
        broad_slacks=lambda:np.array([-1.]),slack=lambda i,x,dx,motion_bound=0.:(distance(x[0]),'strict'))
    return scene


def test_swept_check_finds_collision_between_clear_endpoints_and_midpoint():
    scene=toy_scene(lambda x:abs(x+.1)-.03)
    a=np.zeros(16);b=a.copy();a[0]=-.2;b[0]=.2
    assert scene.slack(0,a,a)[0]>0 and scene.slack(0,b,b)[0]>0
    result=certify_interval(scene,a,b)
    assert result['status']=='failed'
    assert result['configuration'][0]==pytest.approx(-.1)


def test_swept_check_proves_a_whole_interval_and_does_not_accept_unresolved():
    a=np.zeros(16);b=a.copy();b[0]=.1
    assert certify_interval(toy_scene(lambda x:1.),a,b)['status']=='certified'
    assert certify_interval(toy_scene(lambda x:1e-8),a,b,maximum_depth=1)['status']=='unresolved'


def test_inward_claw_offset_has_the_requested_surface_depth():
    cube=np.array([[x,y,z]for x in(-.01,.01)for y in(-.01,.01)for z in(-.01,.01)])
    inner=eroded_vertices(cube,.0005)
    np.testing.assert_allclose(inner.min(0),-.0095,atol=1e-12)
    np.testing.assert_allclose(inner.max(0),.0095,atol=1e-12)
    with pytest.raises(ValueError):eroded_vertices(cube,.011)


def test_skin_localization_requires_the_whole_triangle_in_one_capsule():
    v=np.array([[0.,0.,0.],[.01,0.,0.],[0.,.01,0.],[0.,.04,0.]])
    rim=np.array([[[-.02,0.,0.],[.02,0.,0.]]])
    assert capsule_triangles(v,np.array([[0,1,2],[0,1,3]]),rim,.03).tolist()==[True,False]


def test_urdf_motion_bound_covers_both_arms_and_mimic_fingers():
    import sys,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    from search_fixed_torso_starts import context
    from sock_dressing_simulation.grasp_geometry import HANDEDNESS
    path=root/'artifacts/bulged-cuff/pose-10/state.json'
    if not path.exists():pytest.skip('Calibration snapshot unavailable')
    k,base,*_=context(10,json.loads(path.read_text()))
    bounds=MotionBounds(k,base.groups);rng=np.random.default_rng(18)
    a=state_vector([k.q0[f'left_arm/joint_{i}']for i in range(1,8)],k.start,.025,.035)
    b=a+rng.uniform(-.04,.04,16);b[14:]=[.045,.012];middle=(a+b)/2
    base.park_left(middle[:7],middle[14]);base.update(middle[7:14],middle[15],np.zeros(3))
    centres={link:base.group_vertices[link]@base.group_transforms[link].getRotation().T+base.group_transforms[link].getTranslation()for link in base.groups}
    for t in np.linspace(0,1,11):
        q=a+(b-a)*t;base.park_left(q[:7],q[14]);base.update(q[7:14],q[15],np.zeros(3))
        for link in base.groups:
            vertices=base.group_vertices[link]@base.group_transforms[link].getRotation().T+base.group_transforms[link].getTranslation()
            maximum=np.linalg.norm(vertices-centres[link],axis=1).max()
            assert maximum<=bounds.pair(link,'world')@abs(b-a)/2+1e-10,link
    assert bounds.pair('right_arm/link_4','right_arm/link_5')[7]==0.


def test_measured_native_link_transforms_are_not_replaced_by_fixed_torso_fk():
    import sys,json,copy
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    from search_fixed_torso_starts import context
    from native_pregrasp_geometry import native_update
    path=root/'artifacts/bulged-cuff/pose-10/state.json'
    if not path.exists():pytest.skip('Calibration snapshot unavailable')
    k,base,*_,source=context(10,json.loads(path.read_text()))
    links=[dict(link=r['name'],position=r['position'],rotation=r['quaternion'])for r in source['links']]
    scene=SimpleNamespace(k=k,base=base,delta=np.zeros(3),split={},eroded={})
    native_update(scene,np.zeros(16),links)
    old={s:p.copy()for s,p in scene.positions.items()};bounds={n:tuple(v.copy()for v in b)for n,b in base.group_bounds.items()}
    moved=copy.deepcopy(links);shift=np.array([.001,.002,-.003])
    for r in moved:r['position']=(np.array(r['position'])+shift).tolist()
    native_update(scene,np.ones(16),moved)
    for side in ('left','right'):np.testing.assert_allclose(scene.positions[side]-old[side],shift,atol=1e-12)
    for n,b in base.group_bounds.items():np.testing.assert_allclose(b,np.array(bounds[n])+shift,atol=1e-12)
