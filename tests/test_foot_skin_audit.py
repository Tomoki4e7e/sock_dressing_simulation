from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_foot_skin_containment import validate_pose


def test_skin_audit_rejects_pose_change_instead_of_using_stale_world_vertices():
    reference = {'region':'forefoot', 'reference_position':[0,0,0],
                 'reference_rotation':[0,0,0,1], 'reference_scale':[1,1,1]}
    collider = {'region':'forefoot', 'position':[0,0,0],
                'rotation':[0,0,0,1], 'scale':[1,1,1]}
    row = {'dressing_qa':{'foot_collision_safety':{'foot_collider_geometry':[collider]}}}
    geometry = {'regions':[reference]}
    validate_pose(row, geometry)
    for field in ('position', 'rotation', 'scale'):
        original = collider[field][:]
        collider[field][0] += .001
        with pytest.raises(ValueError, match='pose differs'):
            validate_pose(row, geometry)
        collider[field] = original
