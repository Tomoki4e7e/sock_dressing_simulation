import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('fcl')
trimesh=pytest.importorskip('trimesh')

from sock_dressing_simulation.grasp_geometry import RightHand,SkinBody,ConvexFootProxy,make_part,closed_skin,candidate_samples,orientation,HANDEDNESS

ROOT=Path(__file__).resolve().parents[1]
ROS_UNITY=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])


def box_body():
    mesh=trimesh.creation.box(extents=[1,1,1])
    return SkinBody(dict(vertices=mesh.vertices.tolist(),triangles=mesh.faces.tolist()))


def test_solid_containment_cannot_pass_as_no_surface_intersection():
    body=box_body();part=make_part('finger',trimesh.creation.box(extents=[.1,.1,.1]).vertices)
    valid,distance,_,reason=body.inspect([part],np.eye(3),np.zeros(3),.001)
    assert not valid and reason=='contained_in_body' and distance==0
    valid,_,_,reason=body.inspect([part],np.eye(3),np.array([.8,0,0]),.001)
    assert valid and reason=='clear'


def test_clearance_and_intersection_are_distinguished():
    body=box_body();part=make_part('finger',trimesh.creation.box(extents=[.1,.1,.1]).vertices)
    valid,dist,_,reason=body.inspect([part],np.eye(3),np.array([.5505,0,0]),.001)
    assert not valid and reason=='clearance' and np.isclose(dist,.0005)
    valid,_,_,reason=body.inspect([part],np.eye(3),np.array([.54,0,0]),.001)
    assert not valid and reason=='intersection'


def test_continuous_sweep_finds_obstacle_between_clear_endpoints():
    body=box_body();small=np.asarray(trimesh.creation.box(extents=[.1,.1,.1]).vertices)
    for x in [-.8,.8]:
        assert body.inspect([make_part('end',small)],np.eye(3),np.array([x,0,0]),.001)[0]
    swept=make_part('sweep',np.r_[small+[-.8,0,0],small+[.8,0,0]])
    assert not body.inspect([swept],np.eye(3),np.zeros(3),.001)[0]


def test_convex_contact_proxy_is_distinct_from_skin_surface():
    mesh=trimesh.creation.box(extents=[1,1,1]);proxy=ConvexFootProxy([dict(vertices=mesh.vertices)])
    part=make_part('finger',trimesh.creation.box(extents=[.1,.1,.1]).vertices)
    inside,_=proxy.distances([part],np.eye(3),np.zeros(3))
    outside,_=proxy.distances([part],np.eye(3),np.array([.8,0,0]))
    assert inside[0]<=0 and outside[0]>.001


def test_skin_boundary_caps_close_solid_with_consistent_orientation():
    mesh=trimesh.creation.box()
    faces=mesh.faces[np.asarray(mesh.face_normals)[:,2]<.9]
    closed,caps=closed_skin(dict(vertices=mesh.vertices.tolist(),triangles=faces.tolist()))
    assert len(caps)==4 and closed.is_watertight and closed.is_winding_consistent
    body=SkinBody(dict(vertices=mesh.vertices.tolist(),triangles=faces.tolist()))
    assert np.isclose(body.winding([[0,0,0]])[0],1)
    assert body.winding([[2,0,0]])[0]<1e-8


@pytest.fixture(scope='module')
def hand():
    path=ROOT/'assets/generated/dry_airec3_gripper_v2_rcareworld.urdf'
    if not path.exists():pytest.skip('Generated robot assets not available')
    return RightHand(path)


def test_physical_tip_is_used_instead_of_urdf_tip_frame(hand):
    assert np.isclose(hand.physical_tip_x_m-hand.grasp_center_urdf[0],.010,atol=1e-12)
    assert np.isclose(hand.grasp_center_urdf[0],.2646824232339859)
    assert len(hand.bank(.005))==5
    q=hand.joint_for_gap(.050)
    assert 0<q<.09 and np.isclose(q+hand.gap_offset_m,.050)


def test_handedness_matches_real_ros_to_unity_import(hand):
    sample=dict(normal=[-1,0,0],along=[0,0,1])
    rotation=orientation(sample,0,0)
    assert np.isclose(np.linalg.det(rotation),1)
    assert np.allclose(rotation@HANDEDNESS,ROS_UNITY)
    target=ROS_UNITY@hand.grasp_center_urdf
    for name,points in hand.vertices(.025).items():
        world=points@rotation.T+target
        link,raw=hand.meshes[name];matrix=hand.link_transform(link,hand.joint_for_gap(.025))
        expected=(raw@matrix[:3,:3].T+matrix[:3,3])@ROS_UNITY.T
        assert np.allclose(world,expected)


def test_samples_lie_on_actual_cloth_and_mark_inactive_boundary_faces():
    summary=ROOT/'artifacts/toe-contact-final/summary.json'
    if not summary.exists():pytest.skip('Prepared states not available')
    entry=json.loads(summary.read_text())['trials'][0]
    state=json.loads((ROOT/entry['report']).with_name('visual-state.json').read_text())
    samples=candidate_samples(state,8,[0,.015,.025,.030])
    points=np.array(state['particles_world']);faces=np.array(state['triangles']).reshape(-1,3)
    for sample in samples:
        w=np.array(sample['weights']);assert w.min()>-1e-5 and np.isclose(w.sum(),1)
        assert np.allclose(w@points[faces[sample['face']]],sample['position'])
        assert sample['native_contact_face']==(sample['depth_m']<=.025)
        matrix=orientation(sample,60,-30)
        assert np.allclose(matrix.T@matrix,np.eye(3)) and np.linalg.det(matrix)>.99999
