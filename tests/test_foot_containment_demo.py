import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from foot_containment_demo import closed_faces, inside_points, opening_crossing, FootViewRecorder, finalize


def open_box():
    vertices = [[0,0,0],[1,0,0],[1,0,1],[0,0,1],[0,1,0],[1,1,0],[1,1,1],[0,1,1]]
    faces = [[0,2,1],[0,3,2],[0,1,5],[0,5,4],[1,2,6],[1,6,5],
             [2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    return dict(frame=0,particles_world=vertices,particle_surface_triangles=faces,
                opening_particle_indices=[4,5,6,7],scene_geometry={'right_toe_position':[.5,1.2,.5]})


def test_closed_cuff_disambiguates_open_mesh_and_keeps_geometry():
    row = open_box(); original = np.array(row['particles_world'])
    vertices,faces,cap = closed_faces(row)
    inside,ambiguous = inside_points([[.5,.5,.5],[.5,1.2,.5],[2,.5,.5]],vertices[faces])
    assert inside.tolist()==[True,False,False]
    assert not ambiguous.any()
    assert len(cap)==2
    np.testing.assert_array_equal(row['particles_world'],original)


def test_invalid_boundary_does_not_manufacture_containment():
    row=open_box();row['opening_particle_indices']=[4,5,6]
    with pytest.raises(ValueError):closed_faces(row)


def test_moving_aperture_crosses_stationary_toe():
    first=open_box();second=open_box();second['frame']=1
    second['particles_world']=(np.array(second['particles_world'])+[0,.5,0]).tolist()
    crossing=opening_crossing(first,second)
    assert crossing is not None
    assert crossing['fraction']==pytest.approx(.4,abs=.003)
    assert opening_crossing(first,first) is None


def test_extra_views_use_only_non_simulating_steps_and_assert_clock(tmp_path):
    clock={'fixed_steps':18,'simulated_time_s':.36}
    steps=[];requests=[]
    sock=SimpleNamespace(data={'dressing_qa':{'foot_collision_safety':{'cloth_physics_clock':clock}}},
        request_dressing_qa=lambda:requests.append(True))
    env=SimpleNamespace(_env=SimpleNamespace(step=lambda **kw:steps.append(kw)),sock_cloth=sock,
        _decode=lambda *_args:np.zeros((2,2,3),dtype=np.uint8))
    recorder=FootViewRecorder(env,tmp_path)
    recorder.cameras={name:SimpleNamespace(data={'rgb':b''},GetRGB=lambda *_args:None)
                      for name in ('top','bottom')}
    def communicate(**kw):
        steps.append(kw)
        if len(steps)==2:
            for camera in recorder.cameras.values():camera.data.clear()
    env._env.step=communicate
    cloth={'dressing_qa':{'foot_collision_safety':{'cloth_physics_clock':dict(clock)}}}
    result=recorder.capture(cloth)
    assert len(result['images'])==2
    assert steps==[{'simulate':False},{'simulate':False}]
    for camera in recorder.cameras.values():camera.data={'rgb':b''}
    sock.data['dressing_qa']['foot_collision_safety']['cloth_physics_clock']={'fixed_steps':19}
    with pytest.raises(RuntimeError,match='advanced cloth physics'):recorder.capture(cloth)


def test_replay_initializes_views_from_current_cloth_observation(tmp_path):
    # Execute the real nested rollout hook with a fake backend; the surrounding
    # replay source-copy loop also has a variable named data containing bytes.
    source=Path('scripts/replay_foot_contact.py').read_text()
    hook=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef)
              and n.name=='begin_cloth_contact_rollout')
    klass=ast.ClassDef(name='ReplayEnvironment',bases=[ast.Name(id='Base',ctx=ast.Load())],
                       keywords=[],body=[hook],decorator_list=[])
    tree=ast.fix_missing_locations(ast.Module(body=[klass],type_ignores=[]))
    received=[];steps=[]
    clock={'externally_stepped':True,'fixed_steps':0}
    observation={'dressing_qa':{'foot_collision_safety':{'cloth_physics_clock':clock}},'particles':[]}
    namespace={'Base':type('Base',(),{'begin_cloth_contact_rollout':lambda self:None}),
               'args':SimpleNamespace(foot_views=True),'initial_clock':[None],
               'foot_views':[None],'data':b'compiled_source','directory':tmp_path,
               'FootViewRecorder':lambda *_args:SimpleNamespace(initialize=lambda d:received.append(d)),
               'yaml':SimpleNamespace(safe_dump=lambda *_args,**_kw:'{}')}
    exec(compile(tree,'rollout-hook','exec'),namespace)
    env=namespace['ReplayEnvironment']()
    env.sock_cloth=SimpleNamespace(data=observation,request_dressing_qa=lambda:None,
        request_particles=lambda:None,request_scene_geometry=lambda:None)
    env._env=SimpleNamespace(step=lambda **kw:steps.append(kw))
    env._rollout_material_report=None;env.config={}
    env.begin_cloth_contact_rollout()
    assert received==[observation]
    assert steps==[{'simulate':False}]


@pytest.mark.parametrize('run_mode',('replay','autonomous','user_stopped_replay'))
def test_final_report_checks_containment_sync_and_real_video_frame_counts(tmp_path, run_mode):
    from sock_dressing_simulation.demo import _open_video, _write_video_frame
    samples=[[x,y,z] for x in (.3,.7) for y in (.3,.7) for z in (.3,.7)]
    rows=[];sync=[]
    for frame in range(3):
        row=open_box();row['frame']=frame;row['scene_geometry']['right_toe_position']=[.5,.5,.5]
        clock={'fixed_steps':18*(frame+1),'simulated_time_s':.36*(frame+1)}
        row['dressing_qa']={'anatomical_foot_conformity':{'valid':True,'sections':[
            {'name':name,'samples_world':samples,'containment_ratio':1}
            for name in ('toes','forefoot','heel')]},
            'geometric_maximum_cloth_foot_penetration_m':.001,
            'foot_collision_safety':{'maximum_original_grasp_anchor_error_m':.001,
                'remaining_surface_intersections':0,'remaining_swept_intersections':0,
                'remaining_swept_surface_intersections':0,'cloth_physics_clock':clock}}
        row['stretch']={'available':True,'edge_classes':{'body':{'maximum_stretch':1.1,'edge_count':8}}}
        rows.append(row);sync.append({'frame':frame,'clock_before':clock,'clock_after':clock})
    initial=open_box();initial['frame']=-1;initial['scene_geometry']['right_toe_position']=[.5,.5,.5]
    initial['particles_world']=(np.asarray(initial['particles_world'])+[0,-1,0]).tolist()
    (tmp_path/'foot-initial-state.json').write_text(json.dumps(initial))
    (tmp_path/'live_frame_diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'foot-view-sync.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sync))
    metadata={'replay_command_maximum_error':0,'stop_reason':'max_steps',
              'rollout_initial_cloth_physics_clock':{'fixed_steps':0}}
    if run_mode=='autonomous':
        metadata.update(mode='samdamsarnn_closed_loop',policy_source='SAMDAMSARNNPolicy',
            frames_replayed=0,frames_inferred=3,
            reference_action_projection={'path':None,'blend':0,'cartesian_pull_m':0})
    elif run_mode=='user_stopped_replay':
        metadata.update(stop_reason='user_requested_stop',user_requested_prefix_complete=True,
                        effective_requested_commands=3)
    metadata_path=tmp_path/'metadata.json';metadata_path.write_text(json.dumps(metadata))
    for name in ('demo','side_demo','top_demo','bottom_demo'):
        image=np.zeros((48,64,3),dtype=np.uint8)
        video=_open_video(tmp_path/(name+'.mp4'),image.shape,5)
        for frame in range(3):_write_video_frame(video,image,frame)
        video.release()
    finalize(tmp_path,metadata_path,3)
    result=json.loads((tmp_path/'foot-containment.json').read_text())
    assert result['foot_containment_confirmed_by_geometry']
    assert result['first_three_command_containment_completed_at']==3
    assert result['final_foot_contained']
    assert not result['ankle_calf_pull_up_required']
    assert result['visual_review_complete'] is False
    assert result['recording_complete']
    assert result['command_source_verified']
    assert result['original_commands_unchanged'] is (None if run_mode=='autonomous' else True)
    assert (tmp_path/'top-bottom-demo.mp4').is_file()
    assert (tmp_path/'foot_containment-bottom.png').is_file()
