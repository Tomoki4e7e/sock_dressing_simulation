import copy
import numpy as np
import pytest

from sock_dressing_simulation.circumference_fit import current_pose_hull, assess_fit_hold


def stable_rows():
    return [dict(physics_time_s=2+i*.1,physics_steps=100+i*5,maximum_speed_m_s=.001,
        maximum_foot_penetration_m=0,cuff_force_median_n=.08,hoop_force_max_n=.2,
        finite_mass_particle_count=801,particle_count=801,grasped_particles=0,near_foot_particles=50,total_mass_kg=.05,
        contact_substep_starts=800+i*40,contact_substep_ends=800+i*40,
        gravity=[0,-9.81,0],position_pins_or_geometric_holds=False,native_surface_collisions=True,
        cuff_observed_speed_m_s=.001,cuff_particles_world=[[.04*np.cos(t),.5+i*.0001,.8+.04*np.sin(t)] for t in np.arange(32)*2*np.pi/32],
        force_measurement='native_distance_lambda_over_substep_dt_squared',cuff_center_world=[0,.5+i*.0001,.8])
        for i in range(11)]


def test_stability_requires_actual_physics_and_unpinned_contact():
    result=assess_fit_hold(stable_rows(),reference_center=[0,.5,.8])
    assert result['passed']
    assert result['duration_s']==pytest.approx(1)
    assert result['cuff_drift_m']==pytest.approx(.001)


@pytest.mark.parametrize('change',[
    dict(gravity=[0,0,0]),dict(finite_mass_particle_count=800),dict(grasped_particles=1),
    dict(near_foot_particles=0),dict(contact_substep_ends=0),dict(cuff_observed_speed_m_s=.1),
    dict(maximum_foot_penetration_m=.002),dict(cuff_force_median_n=0),
    dict(cuff_force_median_n=.8),dict(hoop_force_max_n=10),dict(cuff_force_median_n=float('nan')),
    dict(position_pins_or_geometric_holds=True),dict(force_measurement='unmeasured_proxy'),dict(total_mass_kg=.0001),
    dict(physics_time_s=3.,physics_steps=150),
])
def test_bad_retention_cannot_be_reported_stable(change):
    rows=stable_rows()
    for r in rows:r.update(change)
    assert not assess_fit_hold(rows)['passed']


def test_large_drift_or_detachment_cannot_be_hidden_by_small_velocity():
    rows=stable_rows()
    rows[-1]['cuff_center_world'][0]=.006
    assert not assess_fit_hold(rows)['passed']
    rows=stable_rows()
    assert not assess_fit_hold(rows,reference_center=[.1,.5,.8])['passed']
    del rows[0]['cuff_force_median_n']
    assert not assess_fit_hold(rows)['passed']


def test_current_pose_hull_covers_all_skin_support_with_outward_closed_faces():
    pytest.importorskip('scipy.spatial')
    skin=np.array([[x,y,z] for x in (-.04,.04) for y in (-.025,.025) for z in (0,.2)])
    vertices,faces=current_pose_hull(skin)
    a,b,c=(vertices[faces[:,i]] for i in range(3))
    normals=np.cross(b-a,c-a);normals/=np.linalg.norm(normals,axis=1)[:,None]
    assert np.max(skin@normals.T-np.sum(normals*a,axis=1)) < -1e-5
    edges={}
    for triangle in faces:
        for i in range(3):
            edge=tuple(sorted((triangle[i],triangle[(i+1)%3])))
            edges[edge]=edges.get(edge,0)+1
    assert set(edges.values())=={2}


def test_stationary_cuff_center_does_not_hide_ring_deformation():
    rows=stable_rows()
    rows[-1]['cuff_particles_world'][0][0]+=.005
    assert not assess_fit_hold(rows)['passed']


def test_free_sock_tail_can_move_while_the_contact_band_is_stationary():
    rows=stable_rows()
    for r in rows:r['maximum_speed_m_s']=.5
    assert assess_fit_hold(rows)['passed']


def test_shrink_schedule_never_overshoots_and_includes_small_final_change():
    from sock_dressing_simulation.circumference_fit import circumference_schedule
    assert circumference_schedule(.995,.01)==pytest.approx([.995])
    schedule=circumference_schedule(.485,.01)
    assert schedule[0]==pytest.approx(.99)
    assert schedule[-1]==pytest.approx(.485)
    assert np.all(np.diff(schedule)<0)
    assert np.max(-np.diff([1]+schedule))<=.010000001


def test_instantaneous_or_missing_preinsert_fit_config_is_rejected():
    from sock_dressing_simulation.circumference_fit import validate_fit_config
    with pytest.raises(ValueError,match='preinsert'):
        validate_fit_config(dict(circumference_fit=dict(hold_steps=250)))
    with pytest.raises(ValueError,match='gravity hold'):
        validate_fit_config(dict(preinsert=dict(stage='A'),circumference_fit=dict(hold_steps=0)))


def test_saved_pose_check_ignores_hull_order_but_rejects_real_pose_change():
    from sock_dressing_simulation.circumference_fit import equivalent_hulls
    points=np.array([[x,y,z] for x in (0,.1) for y in (0,.08) for z in (0,.25)])
    assert equivalent_hulls(points,points[::-1])
    assert equivalent_hulls(points,points+1e-6)
    assert not equivalent_hulls(points,points+[.001,0,0])
