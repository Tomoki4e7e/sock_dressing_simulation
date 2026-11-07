import xml.etree.ElementTree as ET
import numpy as np
import pytest
pytest.importorskip('fcl')
from scipy.spatial.transform import Rotation
from sock_dressing_simulation.arm_reaching import Kinematics,CollisionScene,shape_vertices
from sock_dressing_simulation.grasp_geometry import origin


def test_fk_applies_parent_rotation_before_prismatic_and_mimic_translation():
    k=Kinematics.__new__(Kinematics)
    k.tree=ET.fromstring('''<robot name="test">
      <link name="world"/><link name="a"/><link name="b"/><link name="c"/><link name="d"/>
      <joint name="right_arm/joint_1" type="revolute"><parent link="world"/><child link="a"/><origin xyz="1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="fixed" type="fixed"><parent link="a"/><child link="b"/><origin xyz="2 0 0"/></joint>
      <joint name="finger" type="prismatic"><parent link="b"/><child link="c"/><axis xyz="0 1 0"/></joint>
      <joint name="mimic" type="prismatic"><parent link="c"/><child link="d"/><axis xyz="0 0 1"/><mimic joint="finger" multiplier="0.5"/></joint>
    </robot>''')
    k.joints={j.get('name'):j for j in k.tree.findall('joint')}
    k.parents={j.find('child').get('link'):j for j in k.joints.values()}
    k.origins={n:origin(j.find('origin')) for n,j in k.joints.items()}
    k.axes={n:np.fromstring(j.find('axis').get('xyz'),sep=' ') if j.find('axis') is not None else np.array([1.,0,0]) for n,j in k.joints.items()}
    k.q0={'finger':.4}
    q=np.zeros(7);q[0]=np.pi/2
    T=k.fk(q)['d']
    assert np.allclose(T[:3,3],[.6,2,.2])
    assert np.allclose(T[:3,:3],Rotation.from_euler('z',90,degrees=True).as_matrix())


def test_box_uses_native_center_scale_and_world_rotation():
    points=shape_vertices(dict(kind='BoxCollider',center=[1,0,0],size=[2,4,6],scale=[-2,1,.5],
        rotation=Rotation.from_euler('z',90,degrees=True).as_quat(),position=[3,4,5]))
    assert np.allclose(points.mean(0),[3,2,5])
    assert np.allclose(np.ptp(points,axis=0),[4,4,3])


def test_collision_exclusions_keep_opposing_fingers_and_cross_arm_pairs():
    s=CollisionScene.__new__(CollisionScene)
    s.group={'right_arm/link_5':'r5','left_arm/link_5':'l5','right_arm/link_6':'r6'}
    s.neighbours={frozenset(['r5','r6'])}
    assert s.excluded('right_arm/link_5','right_arm/link_6')
    assert not s.excluded('right_arm/link_5','left_arm/link_5')
    assert not s.excluded('right_gripper/left_finger_link','right_gripper/right_finger_link')
    assert s.excluded('right_gripper/gripper_base','right_gripper/left_finger_base_link')


def test_ik_reports_unreachable_target_error_without_relaxing_joint_bounds():
    k=Kinematics.__new__(Kinematics);k.lower=np.full(7,-.1);k.upper=np.full(7,.1)
    k.pose=lambda q:(np.asarray(q[:3]),Rotation.from_rotvec(q[3:6]).as_matrix())
    result=k.solve(np.array([.2,0,0]),np.eye(3),np.full(7,10.))
    assert np.all(np.asarray(result['q'])<=k.upper+1e-10)
    assert result['position_error_m']>=.09999


def test_analytic_jacobian_handles_reflected_unity_coordinates():
    from types import SimpleNamespace
    from sock_dressing_simulation.arm_reaching import ROS_UNITY
    k=Kinematics.__new__(Kinematics);k.arm_parent=np.eye(4)
    k.arm_origins=[]
    for i in range(7):
        T=np.eye(4);T[:3,3]=[.04,.01,.03];k.arm_origins.append(T)
    k.arm_axes=np.array([[0,0,1],[0,1,0],[1,0,0],[0,1,0],[1,0,0],[0,1,0],[0,0,1]])
    k.arm_tail=np.eye(4);k.arm_tail[:3,3]=[.05,0,0]
    k.hand=SimpleNamespace(grasp_center_urdf=np.array([.2,0,-.04]));k.A=ROS_UNITY;k.b=np.zeros(3)
    q=np.linspace(-.3,.4,7);p,R,J,W=k.fast_pose(q)
    for i in range(7):
        probe=q.copy();probe[i]+=1e-7;position,rotation=k.pose(probe)
        assert np.allclose((position-p)/1e-7,J[:,i],atol=1e-6)
        assert np.allclose(Rotation.from_matrix(rotation@R.T).as_rotvec()/1e-7,W[:,i],atol=1e-6)


def test_model_exclusion_does_not_disable_gripper_base_forearm_pair():
    s=CollisionScene.__new__(CollisionScene)
    s.group={'right_arm/link_5':'r5','right_arm/link_7':'r7','right_gripper/gripper_base':'r7'}
    s.neighbours=set();s.model_exclusions={frozenset(['right_arm/link_5','right_arm/link_7'])}
    assert s.excluded('right_arm/link_5','right_arm/link_7')
    assert not s.excluded('right_arm/link_5','right_gripper/gripper_base')
