import numpy as np
from sock_dressing_simulation.head_camera_profile import projected_mesh


def test_projection_unions_overlapping_front_and_back_faces():
    vertices=np.array([[-.3,-.3,1.],[.3,-.3,1.],[0.,.3,1.]])
    camera=dict(position=[0.,0.,0.],quaternion=[0.,0.,0.,1.])
    first=projected_mesh(vertices,[[0,1,2]],camera,90,size=(100,100))
    overlapping=projected_mesh(vertices,[[0,1,2],[2,1,0]],camera,90,size=(100,100))
    assert first.sum()>300
    assert np.array_equal(first,overlapping)
