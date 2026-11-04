import numpy as np
import pytest
pytest.importorskip('fcl');trimesh=pytest.importorskip('trimesh');pytest.importorskip('shapely')
from sock_dressing_simulation.geometric_closure import projected_cloth_area,clip_polygon

def fingers(gap):
    vertices=trimesh.creation.box(extents=[.020,.004,.020]).vertices
    return [vertices+[0,-gap/2-.002,0],vertices+[0,gap/2+.002,0]]

def area(y=0,gap=.001,duplicate=False,offset=0):
    points=np.array([[-.004,y,-.004],[.004,y,-.004],[.004,y,.004],[-.004,y,.004]]);points[:,0]+=offset
    faces=np.array([[0,1,2],[0,2,3]]);faces=np.tile(faces,(2,1)) if duplicate else faces
    return projected_cloth_area(points,faces,fingers(gap),np.zeros(4),np.zeros(4))

def test_exact_area_and_no_duplicate_projection_counting():
    a=area();b=area(duplicate=True)
    assert a['area_m2']==pytest.approx(64e-6)
    assert b['area_m2']==pytest.approx(a['area_m2'])
    assert a['area_fraction']==pytest.approx(1)

def test_screen_projection_cannot_include_cloth_outside_gap_or_pad():
    assert area(y=.005)['area_m2']==0
    assert area(offset=.030)['area_m2']==0
    assert area(y=.00045)['area_m2']==0 # Only 0.05 mm from face; require 0.1 mm.

def test_tilted_cloth_is_clipped_in_three_dimensions_before_projection():
    points=np.array([[-.004,-.004,-.004],[.004,.004,-.004],[.004,.004,.004],[-.004,-.004,.004]])
    r=projected_cloth_area(points,np.array([[0,1,2],[0,2,3]]),fingers(.001),np.zeros(4),np.zeros(4))
    assert r['area_m2']==pytest.approx(.0008*.008)
    assert r['span_x_m']==pytest.approx(.0008)

def test_material_target_and_band_limits_exclude_remote_surface():
    points=np.array([[-.004,0,-.004],[.004,0,-.004],[.004,0,.004],[-.004,0,.004]])
    faces=np.array([[0,1,2],[0,2,3]])
    assert projected_cloth_area(points,faces,fingers(.001),np.full(4,.1),np.zeros(4))['area_m2']==0
    assert projected_cloth_area(points,faces,fingers(.001),np.zeros(4),np.full(4,.1))['area_m2']==0

def test_connected_area_handles_roundoff_cracks_without_filling_real_gaps():
    from shapely.geometry import box
    from sock_dressing_simulation.geometric_closure import connected_groups
    a=box(0,0,.002,.004);b=box(.002+1e-12,0,.004,.004)
    assert connected_groups([a,b],1e-9)==[[0,1]]
    c=box(.002+1e-6,0,.004,.004)
    assert len(connected_groups([a,c],1e-9))==2
    point_touch=box(.002,.004,.004,.008)
    assert len(connected_groups([a,point_touch],1e-9))==2

def test_projection_preserves_holes_instead_of_using_a_convex_hull():
    vertices=[];faces=[]
    for x0,x1,z0,z1 in [(-.004,-.002,-.004,.004),(.002,.004,-.004,.004),(-.002,.002,-.004,-.002),(-.002,.002,.002,.004)]:
        start=len(vertices);vertices.extend([[x0,0,z0],[x1,0,z0],[x1,0,z1],[x0,0,z1]]);faces.extend([[start,start+1,start+2],[start,start+2,start+3]])
    v=np.array(vertices)
    r=projected_cloth_area(v,np.array(faces),fingers(.001),np.zeros(len(v)),np.zeros(len(v)))
    assert r['area_m2']==pytest.approx(48e-6)
    assert len(r['polygons_xz_m'])==1 and len(r['polygons_xz_m'][0]['holes'])==1


def test_actual_sloped_pad_surfaces_bound_the_gap():
    f=fingers(.001)
    for v in f:v[:,1]+=.2*v[:,0]
    points=np.array([[-.004,0,-.004],[.004,0,-.004],[.004,0,.004],[-.004,0,.004]])
    r=projected_cloth_area(points,np.array([[0,1,2],[0,2,3]]),f,np.zeros(4),np.zeros(4))
    assert r['area_m2']==pytest.approx(32e-6)

def test_config_rejects_zero_gap_and_impossible_pad_requirements():
    from pathlib import Path
    import yaml
    from sock_dressing_simulation.geometric_closure import validate_settings
    settings=yaml.safe_load((Path(__file__).resolve().parents[1]/'config/geometric_closure.yaml').read_text())
    validate_settings(settings)
    for key,value in [('closing_gap_m',0),('closing_gap_m',.0001),('minimum_area_fraction',1.1),('minimum_span_m',.020),('topology_tolerance_m',.001)]:
        bad=dict(settings);bad[key]=value
        with pytest.raises(ValueError):validate_settings(bad)

def test_insertion_requires_the_inner_pad_to_face_the_skin():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from reevaluate_geometric_closure import insertion
    from sock_dressing_simulation.grasp_geometry import SkinBody
    class Hand:
        def vertices(self,gap):
            a,b=fingers(gap)
            return {'right_gripper/left_finger_link':a,'right_gripper/right_finger_link':b}
    c=dict(gap_m=.01,rotation_matrix=np.eye(3).tolist(),grasp_center_unity_world_m=[0,0,0])
    sample=dict(normal=[0,1,0])
    for center,expected in [([0,-.55,0],True),([0,.55,0],False)]:
        mesh=trimesh.creation.box(extents=[1,1,1]);mesh.apply_translation(center)
        body=SkinBody(dict(vertices=mesh.vertices.tolist(),triangles=mesh.faces.tolist()))
        assert insertion(Hand(),body,c,sample)['passed']==expected
