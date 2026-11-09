import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sock_dressing_simulation.reaching_collection import ReachingCollector


def test_camera_cannot_advance_physics(tmp_path):
    source = tmp_path / 'route.json';source.write_text('{}')
    shared = tmp_path / 'shared.yaml';shared.write_text('{}')
    mapping = SimpleNamespace(names=[str(i) for i in range(18)], revolute_action_indices=range(7))
    plan = dict(shared_initial_configuration='shared.yaml', dataset_inputs=[], geometry_record={},
                workbook_pattern={}, applied_opening_stage='A')
    data = dict(inverse_dynamics_available=True, inverse_dynamics_step=100)
    cloth_data = dict(sock_compound_replay=dict(cloth_fixed_steps=0))
    def step(**kwargs):cloth_data['sock_compound_replay']['cloth_fixed_steps'] += 1
    env = SimpleNamespace(robot=SimpleNamespace(data=data, id=1),
        robot_signals=lambda: dict(angle=np.zeros(18), torque=np.ones(18), external_torque=np.zeros(18),
            torque_available=True, torque_source='drive_forces', external_torque_source='measured contact'),
        recording_camera=SimpleNamespace(GetRGB=lambda *args:None),
        sock_cloth=SimpleNamespace(data=cloth_data, _send_data=lambda *args:None),
        _env=SimpleNamespace(step=step))
    with pytest.raises(RuntimeError, match='Camera capture advanced physics'):
        with ReachingCollector(tmp_path / 'episode', mapping, plan, source, tmp_path, {'camera': {'fov': 60}}) as c:
            for i in range(5):
                c.append(env, mapping, dict(step=i, cloth_fixed_steps=10+i, cloth_time_s=(10+i)*.02,
                    stage='reach', stage_time_s=i*.02))
    metadata = json.loads((tmp_path / 'episode/metadata.json').read_text())
    assert not metadata['completed'] and not metadata['reaching_learning_ready']


@pytest.mark.parametrize('image_size', [(1280,960),(640,480)])
def test_stereo_and_overview_are_decoded_before_ephemeral_payloads_disappear(tmp_path, image_size):
    from PIL import Image
    source = tmp_path / 'route.json';source.write_text('{}')
    (tmp_path / 'shared.yaml').write_text('{}')
    mapping = SimpleNamespace(names=[str(i) for i in range(18)], revolute_action_indices=range(7))
    plan = dict(shared_initial_configuration='shared.yaml', dataset_inputs=[], geometry_record={},
                workbook_pattern={}, applied_opening_stage='A')
    class Camera:
        def __init__(self, value, x):
            self.value=value;self.pending=False
            self.data=dict(position=[x,0.,0.],local_position=[0.,0.,0.],local_rotation=[0.,0.,0.])
        def GetRGB(self,width,height,*args):self.pending=True;self.width=width;self.height=height
    right,left,overview=Camera(40,0.),Camera(80,.084),Camera(120,2.)
    def step(**kwargs):
        for camera in (right,left,overview):
            camera.data.pop('rgb',None)
            if camera.pending:camera.data['rgb']=np.full((camera.height,camera.width,3),camera.value,dtype='uint8')
            camera.pending=False
    native=dict(sock_compound_replay=dict(cloth_fixed_steps=0))
    env=SimpleNamespace(head_cameras=dict(right=right,left=left),recording_camera=overview,
        robot=SimpleNamespace(id=1,data=dict(inverse_dynamics_available=True,inverse_dynamics_step=100)),
        robot_signals=lambda:dict(angle=np.zeros(18),torque=np.ones(18),external_torque=np.zeros(18),
            torque_available=True,torque_source='drive_forces',external_torque_source='contact'),
        sock_cloth=SimpleNamespace(data=native,_send_data=lambda *args:None),
        _env=SimpleNamespace(step=step),_decode=lambda payload,mode:payload.copy())
    with ReachingCollector(tmp_path/'episode',mapping,plan,source,tmp_path,{'camera':{'fov':92}},head_cameras=True,image_size=image_size) as c:
        for i in range(5):
            native['sock_compound_replay']['cloth_fixed_steps']=10+i
            c.append(env,mapping,dict(step=i,cloth_fixed_steps=10+i,cloth_time_s=i*.02,stage='reach',stage_time_s=i*.02))
    for name,value in [('right',40),('left',80),('overview',120)]:
        assert np.asarray(Image.open(tmp_path/'episode'/f'camera_{name}/0.png')).mean()==value
    poses=[json.loads(s) for s in (tmp_path/'episode/camera_poses.jsonl').read_text().splitlines()]
    assert poses[0]['baseline_m']==pytest.approx(.084)
    assert poses[0]['step']==5
    assert (poses[0]['width'],poses[0]['height'])==image_size


@pytest.mark.parametrize('passed', [False, True])
def test_contact_episode_preserves_no_closure_and_failed_replay_gate(tmp_path, passed):
    routes = tmp_path / 'routes';routes.mkdir()
    source = routes / 'A-01.json';source.write_text('{}')
    (tmp_path / 'candidate-initial.yaml').write_text('left_foot_floor_allowance_m: 0.15\n')
    mapping = SimpleNamespace(names=[str(i) for i in range(18)], revolute_action_indices=range(7))
    plan = dict(condition={'id': 'A-01'}, translation_m=[0., 0., 0.])
    with ReachingCollector(tmp_path / 'episode', mapping, plan, source, tmp_path,
                           {'camera': {'fov': 92}}, head_cameras=True, contact_only=True) as collector:
        collector.steps = 150
        collector.finish(dict(passed=passed, initial_steps=50, final_steps=200))
    metadata = json.loads((tmp_path / 'episode/metadata.json').read_text())
    assert metadata['completed'] is passed
    assert metadata['replay_passed'] is passed
    assert metadata['physics_steps_unchanged']
    assert metadata['final_closure_performed'] is False
    assert metadata['endpoint_task'] == 'paired_contact_pregrasp'
    assert metadata['shared_initial_configuration'] == 'candidate-initial.yaml'
    assert metadata['left_foot_floor_allowance_m'] == .15
    assert metadata['camera_profile'] == 'head-stereo-zero-offset-v2'
    assert not metadata['learning_ready']  # Raw-data QA is still required.
