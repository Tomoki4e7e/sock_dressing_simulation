from pathlib import Path
import json
import runpy
import numpy as np
import pytest


@pytest.fixture
def replay(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/summarize_foot_contact_replay.py'))
    evaluate = module['evaluate']
    baseline = tmp_path / 'baseline'
    episode = tmp_path / 'episode'
    baseline.mkdir(); episode.mkdir()
    commands = np.zeros((100, 18))
    np.savetxt(baseline / 'applied_action.csv', commands, delimiter=',')
    np.savetxt(episode / 'applied_action.csv', commands, delimiter=',')
    evaluate.__globals__['BASELINE'] = baseline
    safety = dict(contact_safety_armed=True, remaining_swept_intersections=0, remaining_surface_intersections=0,
                  foot_skin_vertex_count=100, uncovered_foot_skin_vertices=0)
    rows = [dict(geometric_maximum_cloth_foot_penetration_m=0,
                 foot_collision_safety=dict(safety)) for _ in range(100)]
    metadata = dict(frames=100, stop_reason='max_steps', dressing_quality_by_frame=rows,
                    task_success=dict(continuous_grasp_ok=True, success=False,
                                      maximum_obi_cloth_foot_penetration_m=.5))
    path = episode / 'metadata.json'
    def assess():
        path.write_text(json.dumps(metadata))
        return evaluate(path)
    return metadata, commands, episode, assess


def test_contact_safety_and_original_task_success_are_separate(replay):
    _, _, _, assess = replay
    result = assess()
    assert result['passed']
    assert not result['task_success']
    assert result['maximum_original_obi_penetration_m'] == .5


@pytest.mark.parametrize('key', ['remaining_swept_intersections', 'remaining_surface_intersections'])
def test_crossing_at_one_frame_fails_despite_zero_endpoint_penetration(replay, key):
    metadata, _, _, assess = replay
    metadata['dressing_quality_by_frame'][47]['foot_collision_safety'][key] = 1
    assert not assess()['passed']


def test_missing_diagnostics_fail_closed(replay):
    metadata, _, _, assess = replay
    del metadata['dressing_quality_by_frame'][47]['foot_collision_safety']
    assert not assess()['passed']


def test_changed_command_is_not_accepted_as_fixed_motion(replay):
    _, commands, episode, assess = replay
    commands[47, 0] = .001
    np.savetxt(episode / 'applied_action.csv', commands, delimiter=',')
    assert not assess()['checks']['exact_commands']


def test_early_stop_does_not_pass(replay):
    metadata, _, _, assess = replay
    metadata['frames'] = 99
    metadata['stop_reason'] = 'error'
    assert not assess()['passed']


def test_renderer_replay_uses_native_camera_depth_contract():
    module = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/replay_foot_contact.py'))
    depth = np.arange(16, dtype=np.uint8).reshape(4,4)
    sock = np.zeros((4,4), dtype=bool); sock[0,:] = True
    leg = np.zeros((4,4), dtype=bool); leg[3,:] = True
    result = module['renderer_measurements']({'camera':{'camera_depth':depth}},
                                            {'sock':sock, 'leg':leg})
    np.testing.assert_array_equal(result.depth, depth)
    np.testing.assert_array_equal(result.sock_depth, np.where(sock,depth,0))
    assert result.quality['source'] == 'renderer_measurements_for_fixed_motion_replay'


def test_parallel_replays_generate_shared_assets_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import time
    module=runpy.run_path('scripts/replay_foot_contact.py')
    prepare=module['prepared_replay_assets'];calls=[]
    prepare.__globals__['scenario_from_config']=lambda *_args,**_kwargs:SimpleNamespace(
        to_metadata=lambda:dict(sock_mesh=dict(length_m=.3)))
    def generate(config,scenario):
        calls.append(config)
        time.sleep(.03)
        output=Path(config['assets']['output_dir'])
        paths={name:str(output/(name+'.txt')) for name in ['urdf','runtime_urdf','sock_obj']}
        for name,path in paths.items():Path(path).write_text(name)
        return paths
    prepare.__globals__['_prepare']=generate
    directories=[tmp_path/name for name in ['first','second','third']]
    for directory in directories:directory.mkdir()
    config={'assets':{'output_dir':'unused','torobo_ros':'fixed_robot'}}
    with ThreadPoolExecutor(max_workers=3) as executor:
        results=list(executor.map(lambda directory:prepare(config,directory),directories))
    assert len(calls)==1 and results[0]==results[1]==results[2]
    Path(results[0]['runtime_urdf']).write_text('corrupted')
    with pytest.raises(RuntimeError,match='Immutable replay asset changed'):
        prepare(config,directories[0])
    assert len(calls)==1
