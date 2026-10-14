import pytest

from sock_dressing_simulation.quality import physical_gripper_collision_qa, rigid_collision_stop_required


LINKS = ["left_gripper/grasping_frame", "right_gripper/grasping_frame"]


def report(pairs):
    return dict(penetrating_pairs=pairs,
                maximum_penetration_m=max(p["penetration_m"] for p in pairs),
                maximum_enabled_penetration_m=max(p["penetration_m"] for p in pairs),
                maximum_ignored_penetration_m=0.0, ignored_pair_count=0)


def pair(name, depth):
    return dict(robot_collider=name,
                robot_collider_path="world/right_gripper/base_link/grasping_frame/" + name,
                penetration_m=depth, ignored=False)


def test_virtual_trigger_does_not_stop_but_raw_overlap_is_retained():
    raw = report([pair("Empty", .012), pair("finger", .001)])
    qa = physical_gripper_collision_qa(raw, LINKS)
    assert qa["native_report"] == raw
    assert qa["maximum_enabled_penetration_m"] == .001
    assert not rigid_collision_stop_required(qa["maximum_enabled_penetration_m"], .005)
    assert len(qa["excluded_virtual_anchor_overlaps"]) == 1
    assert len(raw["penetrating_pairs"]) == 2


def test_real_finger_contact_still_stops():
    qa = physical_gripper_collision_qa(report([pair("Empty", .02), pair("finger", .012)]), LINKS)
    assert rigid_collision_stop_required(qa["maximum_enabled_penetration_m"], .005)


@pytest.mark.parametrize("maximum", [.03, float("nan")])
def test_incomplete_or_nonfinite_native_summary_remains_fail_closed(maximum):
    raw = report([pair("Empty", .012)])
    raw["maximum_enabled_penetration_m"] = maximum
    qa = physical_gripper_collision_qa(raw, LINKS)
    assert rigid_collision_stop_required(qa["maximum_enabled_penetration_m"], .005)
    assert "excluded_virtual_anchor_overlaps" not in qa


def test_unconfigured_empty_collider_is_not_excluded():
    raw = report([pair("Empty", .012)])
    assert physical_gripper_collision_qa(raw, []) == raw
