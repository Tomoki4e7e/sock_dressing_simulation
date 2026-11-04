"""Shape preservation and failure gates for conditioned native registration."""
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial import ConvexHull
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
pytest.importorskip('fcl')
from prepare_registered_compound import condition,tetrahedra
from validate_registered_compound import check_part
from diagnose_compound_geometry import vector
from sock_dressing_simulation.arm_reaching import ROS_UNITY
from sock_dressing_simulation.convex_distance import point_convex_distance


def cube():return np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])


def test_surface_distance_of_a_rotated_very_thin_box():
    from scipy.spatial.transform import Rotation
    R=Rotation.from_euler('xyz',[.3,-.8,.6]).as_matrix();center=np.array([.3,.6,.4]);vertices=(cube()*[.02,1e-7,2e-7])@R.T+center
    points=np.array([[.003,0.,0.],[.003,3e-7,0],[.020001,2e-7,3e-7]])@R.T+center
    np.testing.assert_allclose(point_convex_distance(points,vertices,surface=True),[1e-7,2e-7,np.sqrt(1e-12+2e-14)],atol=2e-14,rtol=1e-7)
    assert point_convex_distance(points[:1],vertices)[0]==0


def test_real_registered_thin_face_counterexample():
    fixture=json.loads((Path(__file__).parent/'fixtures/convex_thin_face_distance.json').read_text())
    distance=point_convex_distance([fixture['point']],fixture['vertices'],surface=True)[0]
    assert 0<distance<1e-6
    assert distance==pytest.approx(fixture['expected_surface_distance_m'],abs=1e-12)


def test_tetrahedral_fallback_preserves_convex_volume_and_containment():
    from scipy.spatial.transform import Rotation
    points=(cube()*[.04,.02,.003])@Rotation.from_euler('xyz',[.2,.7,-.4]).as_matrix().T+[.1,.08,.02];parent=ConvexHull(points);pieces=tetrahedra(points)
    assert sum(ConvexHull(p).volume for p in pieces)==pytest.approx(parent.volume,rel=1e-10)
    assert max((p@parent.equations[:,:3].T+parent.equations[:,3]).max() for p in pieces)<1e-12


def test_minimum_width_has_explicit_small_geometry_change():
    points=cube()*[.02,.01,1e-7]+[.08,.02,.1];shape,physical,bound=condition(points,4,'right_arm/link_5',2e-5)
    assert min(shape['scale'].values())>=2e-5
    assert bound==pytest.approx(np.linalg.norm(points-physical,axis=1).max())
    assert bound<1.1e-5
    assert shape['conditioned']


def test_native_gate_accepts_exact_zero_error_and_rejects_misses():
    points=cube();shape=dict(vertices=[vector(p) for p in points],scale=vector([1,1,1]),position=vector([0,0,0]),rotation=dict(x=0,y=0,z=0,w=1),link='test',parent_index=4)
    row=dict(index=4,matrix=np.eye(4).tolist(),vertices_world=points.tolist(),obi_vertices_world=points.tolist(),convex=True,obi_enabled=True,contact_offset_m=.0001,rays=[dict(hit=True,origin=[3.,0,0],direction=[-1.,0,0],point=[1.,0,0])])
    k=SimpleNamespace(start=np.zeros(7),A=ROS_UNITY,b=np.zeros(3),fk=lambda *a,**kw:{'test':np.eye(4)})
    cooked=dict(cooked_vertex_hull_hausdorff_m=0.,serialized_plane_hull_hausdorff_m=0.,serialized_plane_hull_closed=True)
    assert check_part(row,shape,cooked,k)['passed']
    row['rays'][0]['hit']=False
    assert not check_part(row,shape,cooked,k)['passed']
