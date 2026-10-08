import json
from pathlib import Path

import numpy as np

from sock_dressing_simulation.config import load_config
from sock_dressing_simulation.drape import run_signed_opening_drape
from sock_dressing_simulation.sock_cloth import SceneGeometry


def _geometry(rotation):
    return SceneGeometry(
        opening_center=(0.0, 0.5, 0.0),
        opening_normal=(0.0, 0.0, -1.0),
        opening_outward_normal=(0.0, 0.0, 1.0),
        sock_body_direction=(0.0, -1.0, 0.0),
        sock_body_gravity_alignment=1.0,
        right_toe_position=(0.0, 0.4, 0.2),
        foot_to_opening_plane_m=0.2,
        foot_to_opening_lateral_m=0.0,
        right_leg_raise_degrees=90.0,
        right_knee_flexion_degrees=0.0,
        left_grasp_position=(-0.05, 0.5, 0.0),
        right_grasp_position=(0.05, 0.5, 0.0),
        left_opening_edge=(-0.05, 0.5, 0.0),
        right_opening_edge=(0.05, 0.5, 0.0),
        opening_to_toe_alignment=1.0,
        left_cuff_insertion_depth_m=0.03,
        right_cuff_insertion_depth_m=0.03,
        sock_tip_center=(0.0, 0.2, 0.0),
        opening_target_normal_alignment=1.0,
        opening_ring_target_alignment=1.0,
        signed_opening_span_rotation_degrees=rotation,
    )


def test_signed_drape_records_both_views_without_policy(tmp_path, monkeypatch):
    config = load_config(
        Path("config/autonomous_real_only_opening_reverse_270deg_close.yaml")
    )
    config["assets"]["output_dir"] = str(tmp_path)
    config["inference"]["recording_crop_xywh"] = None
    config["inference"]["tip_drape_wait"]["consecutive_steps"] = 1

    class Environment:
        def __init__(self, _config):
            self.rotation = 0.0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def load(self, *args, **kwargs):
            pass

        def apply_scenario(self, _scenario):
            return {"ok": True}

        def observe(self):
            return {
                "diagnostics": {
                    "scene_geometry": {
                        "sock_tip_center": [0.0, 0.2, 0.0]
                    }
                }
            }

        def release_initial_tip_guidance(self):
            pass

        def arm_opening_body_barrier_predictive_skin(self, _armed):
            pass

        def rotate_grasped_opening_about_span(self, delta):
            self.rotation += delta

        def step_physics(self):
            pass

        def frame_side_camera_on_sock(self, distance, reverse=False):
            return {"distance_m": distance, "reverse": reverse}

        def observe_drape_cameras(self):
            rgb = np.zeros((8, 8, 3), dtype=np.uint8)
            return {
                "geometry": _geometry(self.rotation),
                "overview": {"rgb": rgb},
                "side": {"rgb": rgb},
            }

    monkeypatch.setattr(
        "sock_dressing_simulation.drape._tip_drape_sample",
        lambda *args, **kwargs: {
            "ok": True,
            "sock_tip_world_y_m": 0.2,
        },
    )
    result = run_signed_opening_drape(
        config,
        prepared={"runtime_urdf": str(tmp_path / "robot.urdf")},
        output_root=tmp_path / "episodes",
        rotation_degrees=-180,
        rotation_steps=2,
        settle_steps=2,
        environment_factory=Environment,
    )

    assert result["ok"]
    assert Path(result["overview_video"]).is_file()
    assert Path(result["side_video"]).is_file()
    metadata = json.loads(
        Path(result["episode"], "metadata.json").read_text()
    )
    assert metadata["policy_inference"] is False
    assert metadata["final_geometry"][
        "signed_opening_span_rotation_degrees"
    ] == -180
