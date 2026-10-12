import copy
import importlib.util
import json
import hashlib
from pathlib import Path

import pytest


@pytest.mark.parametrize('edge_class,stretch,expected', [
    ('body_body', 1.51, True), ('body_body', 1.52, False),
    ('opening_body', 1.112, False), ('pin_body', 1.516, False),
    ('opening_rim', 2.019, True), ('opening_rim', 2.021, False),
    ('body_body', float('nan'), False), ('body_body', None, False),
])
def test_repair_admission_checks_early_frames_and_each_material_class(tmp_path, monkeypatch, edge_class, stretch, expected):
    spec = importlib.util.spec_from_file_location('validate_coupled_pins',
        Path(__file__).resolve().parents[1] / 'scripts/validate_coupled_pins.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.runpy, 'run_path', lambda _: {
        'assess': lambda *_: {'checks': {'original_checks': True}}})
    frame = {'stretch': {'edge_classes': {edge_class: {'maximum_stretch': stretch, 'edge_count': 1},
        'pin_pin': {'maximum_stretch': None, 'edge_count': 0}}}}
    good = copy.deepcopy(frame)
    good['stretch']['edge_classes'][edge_class]['maximum_stretch'] = 1.0
    safety = {'foot_collision_safety': {'maximum_grasp_offset_drift_m': 0,
        'coupled_zero_compliance_pins': True,
        'coupled_cuff_face_projection': True,
        'coupled_pin_projection_order': 'pins_then_material_then_face_contact'}}
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'cloth_quality_by_frame': [frame] + [good] * 10,
        'dressing_quality_by_frame': [safety] * 11}))
    result = module.evaluate(metadata, 11)
    assert result['repair_validation_passed'] is expected
    assert bool(result['all_frame_stretch_violations']) is not expected
    if not expected:
        assert result['all_frame_stretch_violations'][0]['frame'] == 0


@pytest.mark.parametrize('damage', [None, 'missing_manifest', 'modified_assembly'])
def test_replay_requires_a_complete_unchanged_build(tmp_path, damage):
    spec = importlib.util.spec_from_file_location('validate_coupled_pins',
        Path(__file__).resolve().parents[1] / 'scripts/validate_coupled_pins.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    player = tmp_path / 'Player.x86_64'
    player.write_bytes(b'fixture'); player.chmod(0o755)
    archive = tmp_path / 'runtime_sources'; archive.mkdir()
    compiled = tmp_path / 'Player_Data/Managed'; compiled.mkdir(parents=True)
    manifest = {}
    for name in ('RCareWorld.dll', 'Obi.dll'):
        (compiled / name).write_bytes(name.encode())
        manifest['compiled_' + name] = hashlib.sha256(name.encode()).hexdigest()
    (archive / 'SockClothAttr.cs').write_text('fixture source')
    manifest['RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockClothAttr.cs'] = hashlib.sha256(b'fixture source').hexdigest()
    (archive / 'manifest.json').write_text(json.dumps(manifest))
    if damage == 'missing_manifest':
        (archive / 'manifest.json').unlink()
    elif damage == 'modified_assembly':
        (compiled / 'RCareWorld.dll').write_bytes(b'other build')
    if damage:
        with pytest.raises(RuntimeError): module.validate_player(player)
    else:
        assert module.validate_player(player)
