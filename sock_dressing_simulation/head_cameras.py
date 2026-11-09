"""Dry-AIREC's two physical See3CAM frames, separate from inspection cameras."""
from __future__ import annotations

import copy
import numpy as np
from PIL import Image


def create_head_cameras(env):
    import pyrcareworld.attributes as attr
    cameras = {'right': env.camera, 'left': env._env.InstanceObject(name='Camera', id=1310, attr_type=attr.CameraAttr)}
    for side, camera in cameras.items():
        link = f'head/see3cam_{side}/camera_color_frame'
        camera.SetParent(env.robot.id, link)
        # Imported camera_color_frame already uses Unity's camera convention.
        # Do not add a viewing offset or use the ROS optical frame directly.
        camera.SetTransform(position=[0., 0., 0.], rotation=[0., 0., 0.], scale=[1., 1., 1.], is_world=False)
    env._env.step(simulate=False)
    env.head_cameras = cameras
    return cameras


def capture_head_cameras(env, fov=92., width=1280, height=960):
    cameras = getattr(env, 'head_cameras', None) or create_head_cameras(env)
    for camera in cameras.values():camera.GetRGB(width, height, float(fov))
    env._env.step(simulate=False)
    images = {side: env._decode(camera.data['rgb'], 'RGB') for side, camera in cameras.items()}
    poses = {side: {key: copy.deepcopy(camera.data[key]) for key in
             ('position', 'rotation', 'quaternion', 'local_position', 'local_rotation') if key in camera.data}
             for side, camera in cameras.items()}
    for side, pose in poses.items():
        if images[side].shape != (height, width, 3):raise ValueError('Invalid head-camera image size')
        if np.linalg.norm(pose.get('local_position', [99., 99., 99.])) > 1e-6:
            raise RuntimeError('Head camera has an artificial translation')
        rotation = np.asarray(pose.get('local_rotation', [99., 99., 99.]))
        if np.max(abs((rotation + 180.) % 360. - 180.)) > 1e-4:
            raise RuntimeError('Head camera has an artificial rotation')
        pose.update(parent_link=f'head/see3cam_{side}/camera_color_frame', mode='robot_link', fov=float(fov))
    baseline = float(np.linalg.norm(np.asarray(poses['left']['position']) - poses['right']['position']))
    if abs(baseline - .084) > 1e-4:raise RuntimeError(f'Incorrect physical head-camera baseline: {baseline}')
    return images, dict(cameras=poses, baseline_m=baseline, width=width, height=height)


def preview_head_cameras(env, out, fov=92.):
    images, report = capture_head_cameras(env, fov)
    for side, image in images.items():Image.fromarray(image).save(out / f'head-{side}.png')
    return report
