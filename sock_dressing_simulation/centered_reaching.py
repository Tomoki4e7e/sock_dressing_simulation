"""Placement and tool-roll definitions for centered sock reaching experiments."""
import numpy as np
from scipy.spatial.transform import Rotation


def roll_tool(rotation, degrees):
    """Rotate both jaw axes about the unchanged, world-space approach axis."""
    return np.asarray(rotation) @ Rotation.from_rotvec(
        np.array([np.deg2rad(degrees), 0., 0.])
    ).as_matrix()


def placement_bounds(scene, state, horizontal_bound=.8, upward_bound=.3,
                     edge_margin=.01):
    """Keep the entire recorded foot hull within the starting gripper X planes.

    The recorded first contact surface is the foot; the second is the calf.
    These hulls are used only for placement bounds, never as replacement skin
    collision geometry. The floor bound applies to the entire human.
    """
    from .arm_reaching import shape_vertices
    links = {r['name']: r for r in scene['links']}
    span = sorted(links[s + '_gripper/grasping_frame']['position'][0]
                  for s in ('right', 'left'))
    foot = np.asarray(state['source']['surfaces'][0]['vertices'])
    floor = next(r for r in scene['obstacles'] if 'Ground' in r['name'])
    floor_y = shape_vertices(floor)[:, 1].max()
    objects = [np.asarray(scene['human_skin']['vertices']),
               np.asarray(state['particles_world'])]
    objects += [shape_vertices(r) for r in scene['obstacles']
                if 'Ground' not in r['name']]
    minimum_y = min(p[:, 1].min() for p in objects)
    lower = np.array([max(-horizontal_bound, span[0] + edge_margin - foot[:, 0].min()),
                      max(-upward_bound, floor_y - minimum_y), -horizontal_bound])
    upper = np.array([min(horizontal_bound, span[1] - edge_margin - foot[:, 0].max()),
                      upward_bound, horizontal_bound])
    if np.any(lower >= upper):
        raise ValueError('No placement interval contains the foot with the requested margin')
    return lower, upper, dict(
        axis='Unity world X (robot lateral axis in the saved scene)',
        reference='apron starting gripper/grasping_frame positions',
        gripper_x_interval_m=span, edge_margin_m=edge_margin,
        foot_x_interval_before_translation_m=[float(foot[:, 0].min()), float(foot[:, 0].max())],
        foot_geometry='source.surfaces[0]: recorded conservative foot hull',
        translation_lower_m=lower.tolist(), translation_upper_m=upper.tolist())
