from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from sock_dressing_simulation.gravity_reference import GravityTree,ARM_COLUMNS


@pytest.fixture
def pendulum(tmp_path):
    root=ET.Element('robot',name='gravity_test');ET.SubElement(root,'link',name='world')
    for side in ('left','right'):
        parent='world'
        for index in range(1,8):
            child=f'{side}_arm/link_{index}';link=ET.SubElement(root,'link',name=child)
            if index==1:
                inertial=ET.SubElement(link,'inertial');ET.SubElement(inertial,'mass',value='1')
                ET.SubElement(inertial,'origin',xyz='1 0 0');ET.SubElement(inertial,'inertia',ixx='.1',iyy='.1',izz='.1',ixy='0',ixz='0',iyz='0')
            joint=ET.SubElement(root,'joint',name=f'{side}_arm/joint_{index}',type='revolute')
            ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=child)
            ET.SubElement(joint,'axis',xyz='0 1 0' if index==1 else '0 0 1');parent=child
        child=f'{side}_gripper/gripper_base';ET.SubElement(root,'link',name=child)
        joint=ET.SubElement(root,'joint',name=side+'_tool',type='fixed');ET.SubElement(joint,'parent',link=parent);ET.SubElement(joint,'child',link=child)
    path=tmp_path/'pendulum.urdf';ET.ElementTree(root).write(path)
    return GravityTree(path)


def test_gravity_is_holding_torque_with_correct_sign(pendulum):
    q=np.zeros((3,18));q[:,0]=[0.,.3,np.pi/2]
    tau=pendulum.predict(q)
    np.testing.assert_allclose(tau[:,0],-9.81*np.cos(q[:,0]),atol=1e-12)
    np.testing.assert_allclose(tau[:,7],-9.81,atol=1e-12)
    np.testing.assert_allclose(pendulum.predict(q,gravity=(0,0,0)),0.)


def test_torque_matches_independent_potential_energy_difference(pendulum):
    q=np.random.default_rng(21).uniform(-1.,1.,(4,18));actual=pendulum.predict(q);eps=1e-6
    for output,column in enumerate(ARM_COLUMNS):
        plus=q.copy();minus=q.copy();plus[:,column]+=eps;minus[:,column]-=eps
        numeric=(pendulum.potential(plus)-pendulum.potential(minus))/(2*eps)
        np.testing.assert_allclose(actual[:,output],numeric,atol=1e-7)


def test_calibrated_urdf_matches_first_moment_regressor(pendulum,tmp_path):
    q=np.random.default_rng(9).uniform(-1.,1.,(8,18));ng=len(pendulum.groups)
    parameters=np.r_[np.ones(ng),np.zeros(ng*3)];parameters[0]=1.1;parameters[ng:ng+3]=[.004,-.002,.003]
    path=tmp_path/'calibrated.urdf';pendulum.write_calibrated_urdf(path,parameters)
    np.testing.assert_allclose(GravityTree(path).predict(q),pendulum.regressor(q)@parameters,atol=1e-12)


def test_closed_grippers_do_not_qualify_as_released(tmp_path):
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/calibrate_gravity_reference.py'
    spec=importlib.util.spec_from_file_location('gravity_calibration',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    q=np.zeros((30,18));index,_,_=module.terminal_interval(q);assert len(index)==0
    q[20:,7]=.01;q[20:,16]=.01
    index,_,_=module.terminal_interval(q)
    np.testing.assert_array_equal(index,np.arange(20,30))
