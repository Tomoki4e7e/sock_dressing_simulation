import numpy as np
import pytest
pytest.importorskip('fcl')
trimesh=pytest.importorskip('trimesh')
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen,pad_patch
from sock_dressing_simulation.grasp_geometry import make_part


def fingers():
    box=trimesh.creation.box(extents=[.020,.006,.020])
    return [box.vertices+np.array([0,-.008,0]),box.vertices+np.array([0,.008,0])]


def sheet(y=0.,width=.020,x=0.):
    v=np.array([[-width/2,y,-width/2],[width/2,y,-width/2],[width/2,y,width/2],[-width/2,y,width/2]])
    v[:,0]+=x
    return v,np.array([[0,1,2],[0,2,3]],dtype=np.int32)


def test_finite_patch_requires_cloth_between_actual_faces():
    v,f=sheet();r=pad_patch(v,f,fingers());assert r['passed'] and r['grid_hits']==25
    v,f=sheet(y=.006);assert not pad_patch(v,f,fingers())['passed']
    v,f=sheet(width=.001);assert not pad_patch(v,f,fingers())['passed']
    v,f=sheet(x=.030);assert not pad_patch(v,f,fingers())['passed']


def test_cuff_edge_can_have_a_finite_patch_but_a_line_cannot():
    v,f=sheet();v[:,0]+=.010
    assert pad_patch(v,f,fingers())['passed']
    v[:,0]+=.004
    assert not pad_patch(v,f,fingers())['passed']


def test_cloth_intersection_checks_whole_hand_and_swept_body():
    v,f=sheet();screen=ClothScreen(v,f)
    palm=make_part('palm',trimesh.creation.box(extents=[.004,.004,.004]).vertices)
    assert screen.intersections([palm],np.eye(3),np.zeros(3))==['palm']
    assert not screen.intersections([palm],np.eye(3),np.array([0,.02,0]))


def test_photographed_bad_and_good_configurations():
    import json
    from pathlib import Path
    from sock_dressing_simulation.grasp_geometry import RightHand
    root=Path(__file__).resolve().parents[1]
    path=root/'artifacts/grasp-candidate-photos/report.json'
    if not path.exists():pytest.skip('Optional Unity photograph regression data unavailable')
    photos=json.loads(path.read_text());state=json.loads((path.parent/'state.json').read_text())
    prior=json.loads((root/'artifacts/right-grasp-geometry/pose-01/report.json').read_text())
    hand=RightHand(root/prior['settings']['urdf']);screen=ClothScreen(state['particles_world'],state['triangles'])
    for row in photos['candidates']:
        c=row['candidate'];result=screen.inspect(hand,row['gap_m'],np.array(c['rotation_matrix']),np.array(c['grasp_center_unity_world_m']))
        assert result['passed']==row['name'].startswith('D-')
        if not result['passed']:assert result['reason']=='hand_cloth_intersection'
