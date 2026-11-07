"""Regression checks for the rigid translation experiment's world frame."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('translation_search',Path(__file__).resolve().parents[1]/'scripts/search_translated_reaching.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_translation_moves_support_skin_cloth_but_preserves_robot_and_floor():
    scene=dict(human_skin=dict(vertices=[[1.,2.,3.]],triangles=[0,0,0]),particles_world=[[2.,3.,4.]],robot_colliders=[dict(link='fixed',position=[0,0,0])],obstacles=[dict(name='foot',kind='MeshCollider',vertices_world=[[1.,2.,3.]]),dict(name='support',kind='BoxCollider',position=[0.,.5,0.]),dict(name='RCareWorld/Ground',kind='BoxCollider',position=[0.,0.,0.])])
    original=copy.deepcopy(scene);delta=np.array([.2,-.1,.3]);moved=module.translated(scene,delta,scene['particles_world'])
    assert scene==original
    assert moved['robot_colliders']==scene['robot_colliders']
    assert moved['obstacles'][2]==scene['obstacles'][2]
    np.testing.assert_allclose(np.array(moved['human_skin']['vertices'])-delta,scene['human_skin']['vertices'])
    np.testing.assert_allclose(np.array(moved['obstacles'][0]['vertices_world'])-delta,scene['obstacles'][0]['vertices_world'])
    np.testing.assert_allclose(np.array(moved['obstacles'][1]['position'])-delta,scene['obstacles'][1]['position'])
    np.testing.assert_allclose(np.array(moved['particles_world'])-np.array(moved['human_skin']['vertices']),np.array(scene['particles_world'])-np.array(scene['human_skin']['vertices']))
