import copy
from types import SimpleNamespace
import numpy as np
import pytest

from sock_dressing_simulation.pose_patterns import validate_pattern_config
from sock_dressing_simulation.toe_contact import select_toe_support, assess_toe_settling


def native_samples():
    return [dict(physics_model='native_obi_cloth_toe_support',mesh_vertices_are_followers=False,
        physics_steps=10*i,physics_time_s=.2*i,fixed_particles=20,native_mesh_finite_mass_particles=781,
        band_fixed_particles=0,right_finger_colliders_enabled=4,outside_band_contacts=0,
        gravity_world=[0,-9.81,0],maximum_fixed_error_m=0,maximum_free_speed_m_s=.001,
        maximum_band_speed_m_s=.001,band_positions_world=[[.01,.5,.7],[.02,.5,.7]]) for i in range(6)]


def test_native_acceptance_and_failures():
    rows=native_samples()
    assert all(assess_toe_settling(rows,801).values())
    for field,value,check in [
        ('mesh_vertices_are_followers',True,'native_dynamic_cloth'),
        ('native_mesh_finite_mass_particles',0,'all_other_particles_dynamic'),
        ('band_fixed_particles',1,'free_grasp_band'),
        ('outside_band_contacts',1,'no_outside_band_contacts'),
        ('right_finger_colliders_enabled',0,'right_finger_colliders'),
        ('maximum_free_speed_m_s',.2,'free_cloth_settled'),
    ]:
        changed=copy.deepcopy(rows);changed[-1][field]=value
        assert not assess_toe_settling(changed,801)[check]


def test_contact_band_cannot_be_declared_stable_while_drifting():
    rows=native_samples();rows[-1]['band_positions_world'][0][0]+=.01
    assert not assess_toe_settling(rows,801)['band_settled']


def test_small_velocity_jitter_requires_measured_millimetre_stationarity():
    rows=native_samples();rows[-1]['maximum_free_speed_m_s']=.08
    assert not assess_toe_settling(rows,801)['free_cloth_settled']
    rows[-1].update(whole_cloth_recent_window_s=1.,whole_cloth_recent_excursion_m=.0005)
    assert assess_toe_settling(rows,801)['free_cloth_settled']
    rows[-1]['whole_cloth_recent_excursion_m']=.002
    assert not assess_toe_settling(rows,801)['free_cloth_settled']


def test_hybrid_tail_is_reported_separately_and_cannot_replace_dynamic_band():
    rows=native_samples()
    for row in rows:
        row.update(physics_model='native_obi_band_with_dynamic_section_tail',
            mesh_vertices_are_followers=True,band_vertices_are_followers=False,
            fixed_particles=64,native_mesh_finite_mass_particles=128,
            dynamic_tail_follower_particles=609,dynamic_tail_sections=20,
            band_particle_indices=list(range(96)),band_native_finite_particles=96)
    assert all(assess_toe_settling(rows,801).values())
    rows[-1]['band_native_finite_particles']=0
    assert not assess_toe_settling(rows,801)['native_dynamic_cloth']


def test_contact_rollout_preserves_validated_local_contact_model():
    from sock_dressing_simulation.environment import SockDressingEnv
    env=SockDressingEnv.__new__(SockDressingEnv)
    env._pose_pattern_report=dict(toe_contact=dict(passed=True))
    env.sock_cloth=SimpleNamespace(data=dict(sock_toe_contact=dict(active=True,
        band_particle_indices=[0,1,2],band_native_finite_particles=3,band_fixed_particles=0,outside_band_contacts=0)))
    env.begin_cloth_contact_rollout()
    assert env._cloth_contact_rollout_started
    env.sock_cloth.data['sock_toe_contact']['band_fixed_particles']=1
    with pytest.raises(RuntimeError,match='dynamic, validated'):
        env.begin_cloth_contact_rollout()
    env._pose_pattern_report=dict(native_hold=dict(enabled=True))
    with pytest.raises(RuntimeError,match='visual pattern is held'):
        env.begin_cloth_contact_rollout()


def test_support_uses_anatomical_toe_and_never_fixes_grasp_band():
    # A box-shaped foot with its toe bone proximal to the actual skin tip.
    foot=np.array([[x,y,z] for x in (-.02,.02) for y in (-.02,.02) for z in (0,.1)])
    points=np.array([[.022,y,z] for z in (.02,.04,.09,.095) for y in (-.005,0,.005)])
    rest=np.zeros_like(points);rest[:,2]=np.repeat([0,.025,.05,.0625],3)
    source=dict(rest_positions=rest,ankle=[0,0,0],toe=[0,0,.04],surfaces=[dict(vertices=foot)])
    # Include a free hanging tail so the support is a limited fraction.
    points=np.r_[points,np.tile([0,0,.3],(30,1))]
    source['rest_positions']=np.r_[rest,np.tile([0,0,.2],(30,1))]
    support=select_toe_support(points,source)
    ids=support['particle_indices']
    assert len(ids)==6
    assert np.all(np.asarray(source['rest_positions'])[ids,2]>.03)
    assert np.all(points[ids,2]>=.085)
    source['surfaces'][0]['vertices']=foot+[0,0,1]
    with pytest.raises(ValueError,match='No small toe'):
        select_toe_support(points,source)


def test_native_and_section_chain_cannot_both_run():
    settings=dict(workbook='test.xlsx',held_reference='held/A',toe_contact=dict(enabled=True))
    validate_pattern_config(dict(pose_pattern=settings))
    settings['gravity_drape']=dict(enabled=True)
    with pytest.raises(ValueError,match='cannot both'):
        validate_pattern_config(dict(pose_pattern=settings))


@pytest.mark.parametrize('values',[dict(band_width_m=float('nan')),dict(support_length_m=.1),dict(settle_steps=0)])
def test_invalid_toe_settings_rejected(values):
    with pytest.raises(ValueError,match='toe_contact'):
        validate_pattern_config(dict(pose_pattern=dict(workbook='x',held_reference='saved/A',toe_contact=values)))
