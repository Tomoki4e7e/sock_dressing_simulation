from scripts.check_grasp_feasibility import check_frame


def frame(separation):
    return {'frame': 1, 'particle_edges': [[0, 1], [1, 2]],
            'particle_rest_edge_lengths': [.1, .1], 'opening_particle_indices': [],
            'dressing_qa': {'foot_collision_safety': {'original_grasp_anchors': [
                {'side': 'left', 'particle_indices': [0], 'positions_world': [[0, 0, 0]]},
                {'side': 'right', 'particle_indices': [2], 'positions_world': [[separation, 0, 0]]}]}}}


def test_grasp_slack_is_included_in_necessary_bound():
    result = check_frame(frame(.36), .04)
    assert result['worst_pair']['exact_pin_excess_m'] > 0
    assert not result['infeasible_with_grasp_reach']


def test_graph_path_proves_infeasibility_even_with_grasp_slack():
    result = check_frame(frame(.4), .04)
    assert result['infeasible_with_grasp_reach']
    assert abs(result['worst_pair']['maximum_material_path_m'] - .303) < 1e-12


def test_rim_and_opening_body_have_distinct_fixed_limits():
    data = frame(.4)
    data['opening_particle_indices'] = [0, 1]
    result = check_frame(data, .04)
    assert abs(result['worst_pair']['maximum_material_path_m'] - .3131) < 1e-12
