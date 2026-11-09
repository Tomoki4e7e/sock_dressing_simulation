import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from sock_dressing_simulation.planning_human_colliders import envelope

@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('height',[.05,.6])
def test_envelope_contains_exact_scaled_rotated_capsule(axis,height):
 scale=np.array([-1.2,.8,1.7]);rot=Rotation.from_euler('xyz',[.4,-.7,1.1]).as_matrix();offset=np.array([.3,-.2,.5]);center_local=np.array([.02,.03,-.01])
 row=dict(name='capsule',kind='CapsuleCollider',direction=axis,radius=.04,height=height,scale=scale.tolist(),center=center_local.tolist(),rotation=Rotation.from_matrix(rot).as_quat().tolist(),position=offset.tolist())
 mesh=envelope(row);hull=ConvexHull(mesh['vertices_world']);n=hull.equations[:,:3];d=hull.equations[:,3]
 radius=.04*max(abs(scale[i]) for i in range(3) if i!=axis);half=max(0,height*abs(scale[axis])/2-radius);center=rot@(center_local*scale)+offset
 # Analytic maximum of every facet inequality over the entire exact capsule.
 support=n@center+half*np.abs(n@rot[:,axis])+radius*np.linalg.norm(n,axis=1)+d
 assert support.max()<=1e-10
 # A coarse bounding box would be unnecessarily restrictive near contact.
 assert hull.volume < 1.15*(4*np.pi*radius**3/3+2*half*np.pi*radius**2)

def test_unknown_shape_fails_closed():
 with pytest.raises(ValueError,match='Unsupported'):
  envelope(dict(name='unknown',kind='TerrainCollider'))
