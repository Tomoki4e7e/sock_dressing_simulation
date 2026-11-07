import copy
import json

import numpy as np
import pytest

from sock_dressing_simulation.reaching_dataset import (
    assert_shared_initial, file_sha256, spatial_route_distance, verify_dataset_inputs,load_endpoint_samples,
)


def example():
    initial = dict(left_q_rad=list(range(7)), right_q_rad=list(range(7)),
                   translation_m=[-.03, -.12, -.3], left_gap_m=.04, right_gap_m=.04,
                   torso_joints_rad={'torso/joint_1': -.78}, left_foot_floor_allowance_m=.15)
    row = dict(copy.deepcopy(initial), moving_side='left', start_q_rad=initial['left_q_rad'].copy(), initial_opening_gap_m=.04)
    return initial, row


@pytest.mark.parametrize('key', ['left_q_rad', 'right_q_rad', 'translation_m', 'start_q_rad'])
def test_rejects_per_pattern_initial_changes(key):
    initial, row = example()
    assert_shared_initial(row, initial)
    row[key][0] += .001
    with pytest.raises(ValueError):
        assert_shared_initial(row, initial)


def test_rejects_modified_geometry(tmp_path):
    source = tmp_path / 'state.json'; source.write_text('{}')
    record = dict(dataset_inputs=[dict(file='state.json', sha256=file_sha256(source))])
    verify_dataset_inputs(record, tmp_path)
    source.write_text('{"particles_world": []}')
    with pytest.raises(ValueError, match='input changed'):
        verify_dataset_inputs(record, tmp_path)


def test_timing_resampling_does_not_count_as_spatial_diversity():
    straight = [[0, 0, 0], [1, 0, 0]]
    same = [[0, 0, 0], [.01, 0, 0], [.9, 0, 0], [1, 0, 0]]
    assert spatial_route_distance(straight, same) < 1e-14
    assert spatial_route_distance(straight, [[0, 0, 0], [.5, .05, 0], [1, 0, 0]]) > .02


def test_dense_endpoint_samples_are_reused_without_default_resampling(tmp_path):
    state=dict(particles_world=[[0,0,0],[1,0,0],[0,1,0]],triangles=[0,1,2])
    samples=[dict(index=0,face=0,weights=[.2,.3,.5],position=[.3,.5,0]),dict(index=1,face=0,weights=[.1,.7,.2],position=[.7,.2,0])]
    path=tmp_path/'samples.json';path.write_text(json.dumps(samples));probe=dict(samples_file='samples.json',samples_sha256=file_sha256(path))
    assert load_endpoint_samples(probe,state,tmp_path)==samples
    path.write_text('[]')
    with pytest.raises(ValueError,match='samples changed'):load_endpoint_samples(probe,state,tmp_path)


def test_endpoint_samples_must_match_their_frozen_material_face(tmp_path):
    state=dict(particles_world=[[0,0,0],[1,0,0],[0,1,0]],triangles=[0,1,2])
    path=tmp_path/'samples.json';path.write_text(json.dumps([dict(index=0,face=0,weights=[.2,.3,.5],position=[.3,.5,.001])]))
    probe=dict(samples_file='samples.json',samples_sha256=file_sha256(path))
    with pytest.raises(ValueError,match='off its material face'):load_endpoint_samples(probe,state,tmp_path)
