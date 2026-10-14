import json
from pathlib import Path
import sys

import numpy as np
import pytest

from sock_dressing_simulation.config import load_config

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_foot_contact import apply_total_cloth_mass, validate_total_cloth_mass


def test_50g_is_total_mass_and_changes_no_other_material(tmp_path):
    base=load_config(Path('config/opening_widen_w95.yaml'))
    config=load_config(Path('config/foot_containment_autonomous_50g.yaml'))
    snapshot=tmp_path/'state.json'
    snapshot.write_text(json.dumps({'positions':[{}]*801}))
    apply_total_cloth_mass(config,snapshot)
    assert config['obi']['expected']['particle_mass_kg']*801==pytest.approx(.05)
    before=dict(base['obi']['expected']); after=dict(config['obi']['expected'])
    before.pop('particle_mass_kg'); after.pop('particle_mass_kg')
    assert before==after
    # Native uses a float32 inverse mass and reads its reciprocal back.
    requested=config['obi']['expected']['particle_mass_kg']
    measured=float(np.float32(1)/np.float32(1/requested))
    result=validate_total_cloth_mass(config,801,{'particle_mass_kg':measured})
    assert result['ok'] and result['actual_total_mass_kg']==pytest.approx(.05,rel=1e-6)
    assert base['obi']['expected']['particle_mass_kg']==.005


def test_mass_readback_rejects_wrong_count_or_original_mass(tmp_path):
    config=load_config(Path('config/foot_containment_autonomous_50g.yaml'))
    snapshot=tmp_path/'state.json'; snapshot.write_text(json.dumps({'positions':[{}]*801}))
    apply_total_cloth_mass(config,snapshot)
    with pytest.raises(RuntimeError):
        validate_total_cloth_mass(config,801,{'particle_mass_kg':.005})
    with pytest.raises(RuntimeError):
        validate_total_cloth_mass(config,800,{'particle_mass_kg':.05/800})


@pytest.mark.parametrize('total',(0,-.05,float('nan'),float('inf')))
def test_invalid_total_mass_is_rejected(tmp_path,total):
    config=load_config(Path('config/foot_containment_autonomous_50g.yaml'))
    config['obi']['total_mass_kg']=total
    snapshot=tmp_path/'state.json'; snapshot.write_text(json.dumps({'positions':[{}]*801}))
    with pytest.raises(ValueError): apply_total_cloth_mass(config,snapshot)
