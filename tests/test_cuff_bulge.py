import numpy as np
from sock_dressing_simulation.cuff_bulge import refine_sections,radial_bulge


def test_refinement_preserves_original_triangle_area_and_orientation():
    rest=np.array([[1.,0,0],[0,1,.04],[-1,0,0]])
    points=np.array([[0.,0,0],[1,3,2],[0,1,1]])
    rr,pp,faces=refine_sections(rest,points,[[0,1,2]])
    assert np.array_equal(pp[:3],points)
    normal=np.cross(points[1]-points[0],points[2]-points[0])
    normals=np.cross(pp[faces[:,1]]-pp[faces[:,0]],pp[faces[:,2]]-pp[faces[:,0]])
    assert np.all(normals@normal>0)
    assert np.isclose(np.linalg.norm(normals,axis=1).sum(),np.linalg.norm(normal),atol=1e-12)
    assert np.isclose(rr[:,2],.015).any() and np.isclose(rr[:,2],.03).any()


def test_radial_increment_peaks_at_15mm_and_keeps_band_boundaries_fixed():
    angles=np.arange(16)*2*np.pi/16
    rest=np.array([[.04*np.cos(a),.04*np.sin(a),z] for z in [0.,.0075,.015,.0225,.03,.04] for a in angles])
    moved,profile=radial_bulge(rest,rest,rest,rest)
    z=rest[:,2];radius=np.linalg.norm(moved[:,:2],axis=1)
    assert np.allclose(radius[z==.015],.045,atol=1e-12)
    assert np.array_equal(moved[(z==0)|(z>=.03)],rest[(z==0)|(z>=.03)])
    assert np.allclose(moved[:,2],rest[:,2])
    assert np.allclose(radius[z==.0075]-.04,.0025,atol=1e-12)


def test_radial_bulge_is_invariant_to_rigid_rotation_and_translation():
    angles=np.arange(8)*2*np.pi/8
    rest=np.array([[.04*np.cos(a),.04*np.sin(a),z] for z in [0.,.015,.03,.04] for a in angles])
    R=np.array([[0.,0,1],[1,0,0],[0,1,0]]);offset=np.array([.2,.7,-.1])
    a,_=radial_bulge(rest,rest,rest,rest)
    b,_=radial_bulge(rest,rest@R.T+offset,rest,rest@R.T+offset)
    assert np.allclose(b,a@R.T+offset,atol=1e-12)
