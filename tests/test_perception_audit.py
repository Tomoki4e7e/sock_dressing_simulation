import numpy as np

from sock_dressing_simulation.perception_audit import projection_precision


def test_raster_tolerance_accepts_edge_rounding_but_rejects_a_shifted_object():
    reference = np.zeros((50, 50), bool);reference[10:21, 10:21] = True
    rounding = np.zeros_like(reference);rounding[10:21, 12:23] = True
    shifted = np.zeros_like(reference);shifted[10:21, 16:27] = True
    assert projection_precision(rounding, reference)['tolerant'] == 1.
    assert projection_precision(shifted, reference)['tolerant'] < .8
    assert projection_precision(np.zeros_like(reference), reference)['tolerant'] == 0.
