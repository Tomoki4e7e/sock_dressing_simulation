"""Search resolution may vary; reviewed closure and capture rules must not."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import search_centered_reaching as search
import reevaluate_geometric_closure

@pytest.fixture
def setup(monkeypatch):
    calls=[]
    class Hand:
        maximum_gap_m=.05
        def bank(self,gap,kind='endpoint',**kwargs):
            calls.append((gap,kind,kwargs));return [1]
        def vertices(self,gap):
            return {'right_gripper/left_finger_link':np.zeros((4,3)),'right_gripper/right_finger_link':np.ones((4,3))}
    area=dict(area_m2=5e-6,largest_component_area_m2=5e-6,area_fraction=.1,span_x_m=.001,span_z_m=.005)
    monkeypatch.setattr(search,'projected_cloth_area',lambda *a,**kw:area)
    monkeypatch.setattr(reevaluate_geometric_closure,'insertion',lambda *a,**kw:dict(passed=True))
    skin=SimpleNamespace(inspect=lambda *a:(True,.002,'finger','clear'))
    cloth=SimpleNamespace(intersections=lambda *a:[])
    state=dict(particles_world=np.zeros((3,3)),triangles=[0,1,2],source=dict(rest_positions=np.zeros((3,3))))
    sample=dict(index=0,position=[0,0,0]);settings={n:.001 for n in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')}
    args=(SimpleNamespace(hand=Hand()),state,sample,np.eye(3),skin,cloth,settings,np.zeros(3))
    return args,calls,area


def test_continuous_width_keeps_fixed_closure(setup):
    args,calls,_=setup
    result,reason=search.geometric_check(*args,approach_distance=0,opening_gaps=[.001375])
    assert reason=='accepted' and result['valid_openings'][0]['gap_m']==.001375
    assert (.001,'endpoint',{}) in calls
    assert any(g==.001375 and kind=='closure' and kw['pinch_gap']==.001 for g,kind,kw in calls)

@pytest.mark.parametrize('gaps',[[],[.001],[0],[.050001],[float('nan')],[True]])
def test_invalid_width_rejected(setup,gaps):
    args,_,_=setup
    with pytest.raises(ValueError):search.geometric_check(*args,opening_gaps=gaps)


def test_continuous_width_does_not_reduce_capture_threshold(setup):
    args,_,area=setup;area['area_m2']=3.99e-6
    assert search.geometric_check(*args,opening_gaps=[.001375])==(None,'area')
