"""Lighting-only profiles for head camera data; no scene geometry/physics edits."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image

PROFILES = {
    'baseline': dict(correct_normals=False,ambient=0.,sun_intensity=1.,sun_shadow_strength=1.,shadow_distance=20.,lights=[]),
    'normals_only': dict(correct_normals=True,ambient=0.,sun_intensity=1.,sun_shadow_strength=1.,shadow_distance=20.,lights=[]),
    'head_fill': dict(correct_normals=True,ambient=.12,sun_intensity=.7,sun_shadow_strength=.5,sun_rotation=[55.,180.,0.],shadow_distance=20.,lights=[]),
    'balanced': dict(correct_normals=True,ambient=.2,sun_intensity=.65,sun_shadow_strength=.25,sun_rotation=[55.,180.,0.],shadow_distance=20.,lights=[dict(rotation=[40.,90.,0.],intensity=.25),dict(rotation=[40.,270.,0.],intensity=.25)]),
    'balanced_shadow_off': dict(correct_normals=True,ambient=.2,sun_intensity=.65,sun_shadow_strength=0.,sun_rotation=[55.,180.,0.],shadow_distance=0.,lights=[dict(rotation=[40.,90.,0.],intensity=.25),dict(rotation=[40.,270.,0.],intensity=.25)]),
    'bright_fill': dict(correct_normals=True,ambient=.35,sun_intensity=.85,sun_shadow_strength=.15,sun_rotation=[55.,180.,0.],shadow_distance=20.,lights=[dict(rotation=[40.,90.,0.],intensity=.35),dict(rotation=[40.,270.,0.],intensity=.35)]),
}


def apply_lighting(env, profile):
    spec = PROFILES[profile] if isinstance(profile,str) else profile
    payload=dict(profile=profile if isinstance(profile,str) else 'custom',**spec)
    env.sock_cloth._send_data('ConfigureSockLearningLighting',json.dumps(payload));env._env.step(simulate=False)
    actual=env.sock_cloth.data.get('sock_learning_lighting')
    if not actual or actual.get('profile')!=payload['profile']:raise RuntimeError('Player did not apply the requested lighting profile')
    env.learning_lighting=dict(**payload,actual=actual)
    return env.learning_lighting


def probe_lighting(env, out, fov=92.):
    from .head_cameras import capture_head_cameras
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    env.sock_cloth.request_particles();env._env.step(simulate=False)
    particles=np.array(env.sock_cloth.particles()).copy();joints=np.array(env.robot.data['joint_positions']).copy()
    # A temporary unlit cloth render is an independent visibility reference,
    # not a training segmentation mask. Robot/human occlusion remains visible.
    env.sock_cloth._send_data('SetSockInspectionMaterial',True,0);env._env.step(simulate=False)
    images,poses=capture_head_cameras(env,fov)
    for side,rgb in images.items():Image.fromarray(rgb).save(out/f'visibility-reference-{side}.png')
    env.sock_cloth._send_data('SetSockInspectionMaterial',False,0);env._env.step(simulate=False)
    records=[]
    for name in PROFILES:
        apply_lighting(env,name);folder=out/name;folder.mkdir(exist_ok=True)
        images,poses=capture_head_cameras(env,fov)
        for side,rgb in images.items():Image.fromarray(rgb).save(folder/f'camera_{side}.png')
        env.recording_camera.GetRGB(1280,960,float(fov));env._env.step(simulate=False)
        Image.fromarray(env._decode(env.recording_camera.data['rgb'],'RGB')).save(folder/'camera_overview.png')
        records.append(dict(profile=name,lighting=env.learning_lighting,camera_poses=poses))
    env.sock_cloth.request_particles();env._env.step(simulate=False)
    if not np.array_equal(particles,np.array(env.sock_cloth.particles())):raise RuntimeError('Lighting probe changed cloth geometry')
    if not np.array_equal(joints,np.array(env.robot.data['joint_positions'])):raise RuntimeError('Lighting probe changed robot joints')
    report=dict(records=records,cloth_and_joints_unchanged=True,unlit_reference_scope='Independent visibility QA only; not training masks')
    (out/'lighting-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
