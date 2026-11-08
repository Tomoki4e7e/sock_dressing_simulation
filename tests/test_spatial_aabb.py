import numpy as np
import pytest

from sock_dressing_simulation.spatial_aabb import candidate_indices


@pytest.mark.parametrize('seed', range(8))
def test_spatial_culling_keeps_every_dense_aabb_candidate_with_mixed_piece_sizes(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-.3, .3, (90, 3)); b = rng.uniform(-.3, .3, (110, 3))
    half_a = rng.uniform(.0001, .025, a.shape)
    half_b = rng.uniform(.0001, .025, b.shape)
    half_b[::17] *= 20  # Large pieces must not vanish among small parts.
    margin = .005
    lo_a, hi_a, lo_b, hi_b = a-half_a, a+half_a, b-half_b, b+half_b
    exact = candidate_indices(lo_a, hi_a, lo_b, hi_b, margin, minimum_product=10**9)
    spatial = candidate_indices(lo_a, hi_a, lo_b, hi_b, margin, minimum_product=0)
    np.testing.assert_array_equal(spatial, exact)


def test_touching_and_exact_margin_survive_spatial_queries():
    lo_a = np.array([[0., 0., 0.], [3., 0., 0.]])
    hi_a = lo_a+1.
    lo_b = np.array([[1., 0., 0.], [4.005, 0., 0.]])
    hi_b = lo_b+1.
    result = candidate_indices(lo_a, hi_a, lo_b, hi_b, .005, minimum_product=0)
    assert 0 in result and 3 in result


def test_per_frame_bucket_cache_keeps_the_same_candidate_pairs():
    rng = np.random.default_rng(12)
    a = rng.normal(size=(200, 3)); b = rng.normal(size=(400, 3))
    cache = {}
    first = candidate_indices(a, a+.3, b, b+.2, .01, cache=cache, key='body', minimum_product=0)
    second = candidate_indices(a, a+.3, b, b+.2, .01, cache=cache, key='body', minimum_product=0)
    np.testing.assert_array_equal(first, second)
