import importlib.util
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import collect_contact_pregrasp as collection


@pytest.mark.parametrize('kind',['exact','supplementary'])
def test_collection_requires_certificates_bound_to_route_and_initial_bytes(tmp_path,monkeypatch,kind):
    monkeypatch.setattr(collection,'ROOT',tmp_path)
    source=tmp_path/'project/source';(source/'routes').mkdir(parents=True)
    route=source/'routes/A-01.json';route.write_text('{}')
    initial=source/'candidate-initial.yaml';initial.write_text('left_gap_m: 0.04\n')
    binding=dict(route_sha256=collection.sha256(route),initial_sha256=collection.sha256(initial))
    proof=tmp_path/'project/skin-far-verification'
    (proof/'endpoint-audit').mkdir(parents=True)
    (proof/'endpoint-audit/A-01.json').write_text(json.dumps(dict(passed=True,**binding)))
    if kind=='exact':
        (proof/'continuous').mkdir()
        (proof/'continuous/A-01.json').write_text(json.dumps(dict(passed=True,binding=binding)))
    else:
        (proof/'continuous-shoulder-bound').mkdir()
        (proof/'continuous-shoulder-bound/A-01.json').write_text(json.dumps(dict(passed=True,route_sha256=binding['route_sha256'])))
        (source.parent/'result-summary.json').write_text(json.dumps(dict(source_initial_sha256=binding['initial_sha256'])))
    assert set(collection.planning_proofs(route,initial))=={'continuous','endpoint'}
    route.write_text('{"repaired":true}')
    with pytest.raises(ValueError,match='No passed bound'):collection.planning_proofs(route,initial)
    route.write_text('{}');initial.write_text('left_gap_m: 0.03\n')
    with pytest.raises(ValueError,match='No passed bound'):collection.planning_proofs(route,initial)
