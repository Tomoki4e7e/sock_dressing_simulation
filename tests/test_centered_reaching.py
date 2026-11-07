"""Task-space invariants for jaw roll and whole-foot centered placement."""
import numpy as np
import pytest
pytest.importorskip('fcl')
from scipy.spatial.transform import Rotation
from sock_dressing_simulation.centered_reaching import roll_tool, placement_bounds


def test_roll_changes_jaw_direction_without_changing_the_approach():
    initial = Rotation.from_euler('xyz', [21, -36, 72], degrees=True).as_matrix()
    turned = roll_tool(initial, 45)
    np.testing.assert_allclose(turned[:, 0], initial[:, 0], atol=1e-14)
    assert turned[:, 1] @ initial[:, 1] == pytest.approx(np.sqrt(.5))
    np.testing.assert_allclose(turned.T @ turned, np.eye(3), atol=1e-14)
    assert np.linalg.det(turned) == pytest.approx(1.)
    flipped = roll_tool(initial, 180)
    np.testing.assert_allclose(flipped[:, 1:], -initial[:, 1:], atol=1e-14)


def test_placement_contains_the_whole_foot_and_keeps_the_other_foot_above_floor():
    # A centered toe alone would permit the wider heel to leave the interval.
    foot = np.array([[-.15, .7, .4], [.06, .7, .4]])
    scene = dict(links=[dict(name='right_gripper/grasping_frame', position=[-.2, .7, .3]),
                        dict(name='left_gripper/grasping_frame', position=[.2, .7, .3])],
                 human_skin=dict(vertices=[[0, .02, 0], [0, 1.5, 0]]),
                 obstacles=[dict(name='Ground', kind='BoxCollider', position=[0, -.05, 0],
                                 rotation=[0, 0, 0, 1], scale=[1, 1, 1], center=[0, 0, 0], size=[4, .1, 4])])
    state = dict(source=dict(surfaces=[dict(vertices=foot)]), particles_world=[[0, .7, .4]])
    lower, upper, _ = placement_bounds(scene, state)
    for shift in [lower[0], upper[0]]:
        assert (foot[:, 0]+shift).min() >= -.19-1e-12
        assert (foot[:, 0]+shift).max() <= .19+1e-12
    assert lower[1] == pytest.approx(-.02)
