import copy
import json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
pytest.importorskip('fcl')
from sock_dressing_simulation.fitted_b_cuff import build_fitted_b_state,qualify_b_state,PROFILE
from sock_dressing_simulation.natural_b_shape import section_perimeters
from sock_dressing_simulation.pose_patterns import read_patterns
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def fitted_b():
    source=ROOT/'artifacts/bulged-cuff/pose-10/state.json'
    if not source.exists():pytest.skip('Native foot geometry is not installed')
    original=json.loads(source.read_text());pattern=next(p for p in read_patterns(ROOT/'artifacts/sock-pose-pattern/sock_pose_patterns.xlsx') if p.number==10)
    state,before,profile=build_fitted_b_state(original,pattern)
    return original,pattern,state,before,profile


def test_B_uses_its_forefoot_and_remaining_length_with_no_A_scale(fitted_b):
    original,_,state,_,_=fitted_b;fit=state['cuff_fit'];e=state['toe_envelope']
    assert fit['passed'] and state['shape_profile']==PROFILE
    assert state['source']==original['source'] and state['triangles']==original['triangles']
    assert fit['wrapped_skin_distance_m']['median']<.008
    assert e['free_sleeve_perimeter_m']==pytest.approx(.9*e['toe_envelope_anchor_perimeter_m'])
    assert e['material_scale_estimate']!=pytest.approx(.54)
    assert not state['placement']['material_rest_lengths_modified']
    assert not fit['physical_cloth_deformation_validated']


def test_anatomical_toes_do_not_become_a_garment_bottleneck(fitted_b):
    _,_,state,before,_=fitted_b;e=state['toe_envelope'];rest=np.array(state['source']['rest_positions'])
    z,c=section_perimeters(before,rest);active=(z>=e['toe_envelope_anchor_depth_m'])&(z<=e['garment_cap_start_m'])
    assert c[active].min()>=.8*e['toe_envelope_anchor_perimeter_m']
    assert state['cuff_fit']['toe_continuity']['minimum_adjacent_perimeter_ratio']>=.85
    assert np.interp(e['anatomical_tip_material_depth_m'],z,c)>.2
    assert c[-1]<c[np.flatnonzero(z<=e['garment_cap_start_m'])[-1]]
    assert state['cuff_fit']['free_distal_downward_alignment']>.8


def test_bulge_is_5mm_after_the_B_specific_shape_with_fixed_band_boundaries(fitted_b):
    _,_,state,before,_=fitted_b;rest=np.array(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min();after=np.array(state['particles_world'])
    peak=np.isclose(depth,.015,atol=1e-8,rtol=0);outside=(depth<=1e-10)|(depth>=.03-1e-10)
    np.testing.assert_allclose(np.linalg.norm(after[peak]-before[peak],axis=1),.005,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(after[outside],before[outside])


def test_regression_rejects_a_thin_toe_even_if_the_rim_is_unchanged(fitted_b):
    _,_,state,before,profile=fitted_b;state=copy.deepcopy(state);before=before.copy();after=np.array(state['particles_world']);rest=np.array(state['source']['rest_positions']);z=rest[:,2]-rest[:,2].min()
    ids=np.isclose(z,.15,atol=1e-6,rtol=0);center=before[ids].mean(0);before[ids]=center+.2*(before[ids]-center);after[ids]=before[ids];state['particles_world']=after.tolist()
    with pytest.raises(ValueError,match="'natural_toe_envelope': False"):qualify_b_state(state,before,profile)


def test_regression_rejects_an_oversized_contact_region(fitted_b):
    _,_,state,before,profile=fitted_b;state=copy.deepcopy(state);before=before.copy();after=np.array(state['particles_world']);rest=np.array(state['source']['rest_positions']);z=rest[:,2]-rest[:,2].min()
    for level in np.unique(z[(z>=.03-1e-8)&(z<=.1+1e-8)]):
        ids=np.isclose(z,level,atol=1e-8,rtol=0);center=before[ids].mean(0);before[ids]=center+2*(before[ids]-center);after[ids]=before[ids]
    state['particles_world']=after.tolist()
    with pytest.raises(ValueError,match="'contact_region_fits_skin': False"):qualify_b_state(state,before,profile)


def test_regression_rejects_changed_peak_or_band_exterior(fitted_b):
    _,_,state,before,profile=fitted_b;state=copy.deepcopy(state);after=np.array(state['particles_world']);rest=np.array(state['source']['rest_positions']);z=rest[:,2]-rest[:,2].min();after[np.isclose(z,.015,atol=1e-8,rtol=0),0]+=.002;state['particles_world']=after.tolist()
    with pytest.raises(ValueError,match="'peak_radius_increment_5mm': False"):qualify_b_state(state,before,profile)


def test_workbook_features_remain_metadata_in_this_reference(fitted_b):
    original,pattern,state,*_=fitted_b;alternative=replace(pattern,shift_cm=3,diagonal_deg=15,opening='つぶれている',wrinkles='全体ずれ',fold='内巻き')
    other,_,_=build_fitted_b_state(original,alternative);np.testing.assert_array_equal(other['particles_world'],state['particles_world'])


def test_endpoint_search_uses_the_new_shape_and_rejects_a_previous_cache(fitted_b,tmp_path):
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    from search_diverse_left_endpoints import run
    initial=ROOT/'config/diverse_left_reaching_shared.yaml';run([10],['B'],tmp_path,initial,seconds=1e-9)
    state=json.loads((tmp_path/'states/pose-10-B.json').read_text());assert state['shape_profile']==PROFILE
    path=tmp_path/'pose-10-B.json';cached=json.loads(path.read_text());cached['shape_profile']='native_B_same_A_material_gravity_tail_5mm_bulge_v4';path.write_text(json.dumps(cached))
    with pytest.raises(ValueError,match='old, unfitted shape'):run([10],['B'],tmp_path,initial,seconds=1e-9)
