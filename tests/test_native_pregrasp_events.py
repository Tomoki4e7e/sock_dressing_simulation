"""Ensure sub-10µm native events obey the accepted localized-contact policy."""
import json
from pathlib import Path
import sys
import pytest
pytest.importorskip('fcl')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_pregrasp_native_contact_events import run

@pytest.mark.parametrize('distance,link,collider,separation,point_x,expected',[
    (.0051,'left_finger_link','right_leg_actual_skin_rigid',-5e-7,.1,False),
    (.0049,'left_finger_link','right_leg_actual_skin_rigid',-5e-7,.1,True),
    (.0049,'gripper_base','right_leg_actual_skin_rigid',-5e-7,.1,False),
    (.0049,'left_finger_link','right_leg_actual_skin_rigid',-5e-7,.15,False),
    (.0049,'left_finger_link','right_leg_actual_skin_rigid',-.00051,.1,False),
    (.0049,'left_finger_link','Chair',-5e-7,.1,False),
    (.0051,'left_finger_link','right_leg_actual_skin_rigid',-5e-8,.1,True),
])
def test_native_event_recheck_covers_window_part_region_and_intrusion(tmp_path,distance,link,collider,separation,point_x,expected):
    out=tmp_path/'out';source=tmp_path/'source';(source/'routes').mkdir(parents=True)
    folder=out/'drives/T/native-route-initial';folder.mkdir(parents=True)
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(particles_world=[[0,0,0],[.01,0,0],[0,.01,0]],triangles=[0,1,2],source=dict(rest_positions=[[0,0,0],[.01,0,0],[0,.01,0]]))))
    (source/'candidate-initial.yaml').write_text('translation_m: [0.1, 0.2, 0.3]\n')
    (source/'routes/T.json').write_text(json.dumps(dict(condition=dict(state=str(state)),terminal=dict(left=dict(tcp_world_m=[.1,.2,.3])))))
    trace=folder/'motion-trace.jsonl';trace.write_text(json.dumps(dict(time_s=1.,stage='approach',left_tcp_world_m=[.1+distance,.2,.3],contacts=[dict(a_link='left_gripper/'+link,b_link='',a='claw',b=collider,separation_m=separation,point=[point_x,.2,.3])]))+'\n')
    (folder/'replay-result.json').write_text('{}')
    run((trace,out,source))
    result=json.loads((out/'native-contact-event-audit/T.json').read_text())
    assert result['passed']is expected
