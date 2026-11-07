"""Geometry and acceptance gates for physical, unpinned circumference fitting."""
from __future__ import annotations

import copy
import numpy as np


def circumference_schedule(minimum_scale, step):
    if not np.isfinite([minimum_scale,step]).all() or not .4<=minimum_scale<1 or not 0<step<=.05:
        raise ValueError('Invalid circumference scale schedule')
    count=int(np.ceil((1-minimum_scale)/step-1e-10))
    return [max(float(minimum_scale),float(1-step*i)) for i in range(1,count+1)]


def validate_fit_config(config):
    settings=config.get('circumference_fit')
    if not settings:
        return
    if not config.get('preinsert'):
        raise ValueError('circumference_fit requires a preinsert stage')
    if not isinstance(settings,dict) or set(settings)-{'minimum_scale','scale_step','settle_steps','hold_steps','target_cuff_force_n'}:
        raise ValueError('Unknown circumference_fit settings')
    circumference_schedule(float(settings.get('minimum_scale',.48)),float(settings.get('scale_step',.01)))
    if settings.get('settle_steps',5)<1 or settings.get('hold_steps',250)<55:
        raise ValueError('Circumference fitting requires relaxation and a measured gravity hold')
    force=float(settings.get('target_cuff_force_n',.08))
    if not np.isfinite(force) or force<=0:
        raise ValueError('Target cuff force must be finite and positive')


def current_pose_hull(skin, padding_m=.0003):
    from scipy.spatial import ConvexHull
    points=np.asarray(skin,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():
        raise ValueError('Skin support must contain finite Nx3 vertices')
    if not np.isfinite(padding_m) or padding_m<0:
        raise ValueError('Hull padding must be finite and nonnegative')
    cloud=(points[:,None,:]+np.r_[np.eye(3),-np.eye(3)][None,:,:]*padding_m).reshape(-1,3)
    # Remove sub-micron slivers before Unity casts the hull to float32.
    cloud=np.unique(np.round(cloud/.000002)*.000002,axis=0)
    vertices=cloud[ConvexHull(cloud).vertices]
    hull=ConvexHull(vertices)
    triangles=hull.simplices.copy()
    for face,equation in zip(triangles,hull.equations):
        a,b,c=vertices[face]
        if np.dot(np.cross(b-a,c-a),equation[:3])<0:face[1],face[2]=face[2],face[1]
    return vertices,triangles


def assess_fit_hold(rows, *, reference_center=None, minimum_duration_s=1., maximum_drift_m=.005,
                    maximum_speed_m_s=.05, maximum_penetration_m=.001, tension_range_n=(.01,.5),
                    maximum_cuff_point_excursion_m=.005):
    """Fail closed: still pictures, missing force data, pins and zero gravity fail."""
    if len(rows)<2:
        return dict(passed=False,checks=dict(sufficient_samples=False),cuff_drift_m=None)
    def finite(r,k):
        try:return bool(np.isfinite(float(r[k])))
        except (KeyError,TypeError,ValueError):return False
    keys=('physics_time_s','physics_steps','maximum_speed_m_s','cuff_observed_speed_m_s','maximum_foot_penetration_m',
          'cuff_force_median_n','hoop_force_max_n','finite_mass_particle_count','particle_count',
          'grasped_particles','near_foot_particles','contact_substep_starts','contact_substep_ends','total_mass_kg')
    if not all(all(finite(r,k) for k in keys) for r in rows):
        return dict(passed=False,checks=dict(measured_finite_data=False),cuff_drift_m=None)
    centers=np.asarray([r.get('cuff_center_world',[np.nan]*3) for r in rows],dtype=float)
    if centers.shape!=(len(rows),3) or not np.isfinite(centers).all():
        return dict(passed=False,checks=dict(measured_centers=False),cuff_drift_m=None)
    drift=float(np.max(np.linalg.norm(centers-centers[0],axis=1)))
    cuff_points=np.asarray([r.get('cuff_particles_world',[]) for r in rows],dtype=float)
    if cuff_points.ndim!=3 or cuff_points.shape[0]!=len(rows) or cuff_points.shape[1]<8 or cuff_points.shape[2]!=3 or not np.isfinite(cuff_points).all():
        return dict(passed=False,checks=dict(measured_cuff_shape=False),cuff_drift_m=drift)
    excursion=float(np.max(np.linalg.norm(cuff_points-cuff_points[0],axis=2)))
    times=np.array([r['physics_time_s'] for r in rows]);steps=np.array([r['physics_steps'] for r in rows])
    checks=dict(
        simulated_duration=bool(times[-1]-times[0]>=minimum_duration_s-1e-5 and np.all(np.diff(times)>0) and np.all(np.diff(steps)>0)),
        gravity_enabled=all(np.shape(r.get('gravity'))==(3,) and np.allclose(r['gravity'],[0,-9.81,0]) for r in rows),
        unpinned_finite_mass=all(r['particle_count']>0 and r['grasped_particles']==0 and r['finite_mass_particle_count']==r['particle_count'] and r.get('position_pins_or_geometric_holds') is False for r in rows),
        physical_mass=all(np.isclose(r['total_mass_kg'],.05,atol=1e-5,rtol=0) for r in rows),
        actual_contact_solver=all(r['contact_substep_starts']>0 and r['contact_substep_ends']==r['contact_substep_starts'] and r.get('native_surface_collisions') is True for r in rows),
        measured_tension=all(r.get('force_measurement')=='native_distance_lambda_over_substep_dt_squared' for r in rows),
        foot_contact=all(r['near_foot_particles']>=8 for r in rows),
        cuff_drift=drift<=maximum_drift_m,
        low_cuff_motion=all(r['cuff_observed_speed_m_s']<=maximum_speed_m_s for r in rows) and excursion<=maximum_cuff_point_excursion_m,
        no_excess_penetration=all(r['maximum_foot_penetration_m']<=maximum_penetration_m for r in rows),
        moderate_tension=all(tension_range_n[0]<=r['cuff_force_median_n']<=tension_range_n[1] and r['hoop_force_max_n']<=2 for r in rows),
    )
    if reference_center is not None:
        checks['foot_retained_from_start']=float(np.max(np.linalg.norm(centers-np.asarray(reference_center),axis=1)))<=.02
    return dict(passed=all(checks.values()),checks=checks,cuff_drift_m=drift,cuff_point_excursion_m=excursion,duration_s=float(times[-1]-times[0]))


def begin_fit(environment, *, stage_contact=False):
    """Install a hull of this human pose, then calibrate native garment rest size."""
    attr=environment.sock_cloth
    attr.request_fit_skin();environment._env.step(simulate=False)
    skin=attr.data.get('sock_fit_skin')
    if skin is None:
        raise RuntimeError('Player does not support circumference fitting; rebuild it')
    vertices,triangles=current_pose_hull(skin)
    calf_vertices,calf_triangles=current_pose_hull(attr.data['sock_fit_calf_skin'])
    attr.install_fit_hull(vertices,triangles,calf_vertices,calf_triangles);environment._env.step(simulate=False)
    hull=attr.data.get('sock_fit_hull',{})
    if not hull.get('ok') or hull.get('uncovered')!=0 or hull.get('calf_uncovered')!=0:
        raise RuntimeError('Current-pose foot hull does not cover the skin')
    if stage_contact:
        attr._send_data('BeginSockHeldStageCircumferenceFit')
    else:
        attr.begin_circumference_fit()
    environment._env.step(simulate=False)
    if not attr.data.get('sock_circumference_fit',{}).get('active'):
        raise RuntimeError('Native circumference fitting initialization failed')
    result=dict(vertices=vertices.tolist(),triangles=triangles.ravel().tolist(),native=hull)
    result['calf_surface']=dict(vertices=calf_vertices.tolist(),triangles=calf_triangles.ravel().tolist())
    return result


def read_fit(environment):
    import copy
    attr=environment.sock_cloth
    attr.request_circumference_fit();attr.request_particles();attr.request_particle_velocities()
    environment._env.step(simulate=False)
    row=copy.deepcopy(attr.data['sock_circumference_fit'])
    ids=np.asarray(attr.data['opening_particle_indices'],dtype=int)
    points=attr.particles()[ids]
    velocities=attr.particle_velocities()[ids]
    row['cuff_particles_world']=points.tolist()
    row['cuff_maximum_native_speed_m_s']=float(np.linalg.norm(velocities,axis=1).max())
    previous=getattr(environment,'_circumference_fit_observation',None)
    dt=row['physics_time_s']-previous[0] if previous is not None else 0
    row['cuff_observed_speed_m_s']=(float(np.linalg.norm(points-previous[1],axis=1).max()/dt) if dt>1e-8 else 0.)
    environment._circumference_fit_observation=(row['physics_time_s'],points.copy())
    return row


def run_fit(environment, *, minimum_scale=.48, scale_step=.01, settle_steps=5,
            hold_steps=250, target_cuff_force_n=.08, on_sample=None, on_phase=None, stage_contact=False):
    """Contract by rest circumference, then verify finite-mass gravitational hold.

    Temporary zero gravity during shape relaxation is explicit. Acceptance only
    uses subsequent +time samples under full gravity with no supports or pins.
    """
    if (not np.isfinite([minimum_scale,scale_step,target_cuff_force_n]).all()
            or not .4<=minimum_scale<1 or not 0<scale_step<=.05
            or settle_steps<1 or hold_steps<55 or target_cuff_force_n<=0):
        raise ValueError('Invalid circumference fitting schedule')
    if environment._cloth_contact_rollout_started:
        raise RuntimeError('Fit before starting the gripper contact rollout')
    hull=begin_fit(environment, stage_contact=True) if stage_contact else begin_fit(environment)
    rows=[]
    def sample(phase):
        row=read_fit(environment);row['phase']=phase;rows.append(row)
        if on_sample is not None:on_sample(row)
        return row
    initial=sample('initial')
    if on_phase is not None:on_phase('initial',initial)
    chosen=1.
    for scale in circumference_schedule(minimum_scale,scale_step):
        chosen=float(max(minimum_scale,scale))
        if stage_contact:
            environment.sock_cloth._send_data('SetSockHeldStageCircumferenceScale', chosen)
        else:
            environment.sock_cloth.set_circumference_scale(chosen)
        environment._env.step(simulate=False)
        environment.advance_physics(settle_steps)
        row=sample('contract')
        if row['near_foot_particles']>=8 and row['cuff_force_median_n']>=target_cuff_force_n:
            break
    contracted=row
    if on_phase is not None:on_phase('contracted',contracted)
    def holding_sample(row):
        rows.append(row)
        if on_sample is not None:on_sample(row)
    held=hold_fit(environment,steps=hold_steps,reference_center=contracted['cuff_center_world'],on_sample=holding_sample)
    hold=held['telemetry'];assessment=held['assessment']
    if on_phase is not None:on_phase('held',hold[-1])
    # Apply the same independent surface gate to library initialization as CLI
    # exports: particle contact alone cannot detect every coarse-mesh fold.
    from .preinsert import audit_preinsert, audit_preinsert_surface
    attr=environment.sock_cloth
    attr.request_particles();environment._env.step(simulate=False)
    surfaces=dict(surfaces=[hull]+([hull['calf_surface']] if hull.get('calf_surface') else []))
    geometry_audit=audit_preinsert(attr.particles(),surfaces,
                                  attr.data['particle_edges'],attr.data['particle_rest_edge_lengths'])
    surface_audit=audit_preinsert_surface(attr.particles(),surfaces,
                                         attr.data['particle_edges'],attr.data['particle_surface_triangles'])
    environment._preinsert_observation_pending=True
    return dict(assessment=assessment,
                passed=assessment['passed'] and geometry_audit['geometry_ok'] and surface_audit['surface_ok'],
                geometry_audit=geometry_audit,surface_audit=surface_audit,stage=environment.config['preinsert']['stage'],
                initial=initial,contracted=contracted,final=hold[-1],rest_circumference_scale=chosen,
                foot_hull=hull,telemetry=rows,preparation_gravity_disabled=True,
                gravity_hold_duration_s=hold[-1]['physics_time_s']-contracted['physics_time_s'])


def hold_fit(environment, *, steps=250, reference_center=None, on_sample=None):
    if steps<55:
        raise ValueError('At least 55 fixed steps are required for a measured hold')
    environment.sock_cloth.set_fit_gravity(True);environment._env.step(simulate=False)
    rows=[]
    for first in range(0,steps,5):
        environment.advance_physics(min(5,steps-first))
        row=read_fit(environment);row['phase']='gravity_hold';rows.append(row)
        if on_sample is not None:on_sample(row)
    return dict(assessment=assess_fit_hold(rows[-11:],reference_center=reference_center),telemetry=rows)


def restore_fit(environment, directory):
    """Restore calibrated rest geometry as well as the settled native state."""
    import json
    from pathlib import Path
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    if not report.get('passed'):
        raise ValueError('Refusing to restore a fit that failed its physical hold test')
    attr=environment.sock_cloth
    if attr.data.get('sock_circumference_fit',{}).get('active'):
        raise ValueError('Disable automatic circumference_fit when preparing a scene for saved-state restoration')
    attr.restore_prepared_state((directory/'initial-prepared-state.json').read_text().strip())
    environment._env.step(simulate=False)
    stage_contact = report.get('fit_settings', {}).get('stage_contact', False)
    hull=begin_fit(environment, stage_contact=True) if stage_contact else begin_fit(environment)
    old_hull=json.loads((directory/'foot-collision-hull.json').read_text())
    if not equivalent_hulls(hull['vertices'],old_hull['vertices']):
        raise ValueError('Saved fit belongs to a different human pose')
    if old_hull.get('calf_surface') and not equivalent_hulls(hull['calf_surface']['vertices'],old_hull['calf_surface']['vertices']):
        raise ValueError('Saved fit belongs to a different calf pose')
    if stage_contact:
        attr._send_data('SetSockHeldStageCircumferenceScale', report['rest_circumference_scale'])
    else:
        attr.set_circumference_scale(report['rest_circumference_scale'])
    environment._env.step(simulate=False)
    attr.restore_prepared_state((directory/'fitted-prepared-state.json').read_text().strip())
    attr.request_particles();environment._env.step(simulate=False)
    restored=attr.data.get('prepared_state_restore_report',{})
    expected=json.loads((directory/'fitted-geometry.json').read_text())['particles_world']
    if not restored.get('ok') or not np.allclose(attr.particles(),expected,atol=5e-6,rtol=0):
        raise RuntimeError('Settled fit restore failed native position readback')
    attr.set_fit_gravity(True);environment._env.step(simulate=False)
    environment._circumference_fit_observation=None
    environment._preinsert_observation_pending=True
    return dict(restored=True,rest_circumference_scale=report['rest_circumference_scale'],native=restored)


def equivalent_hulls(first,second,tolerance_m=5e-6):
    """Compare convex bodies independently of vertex ordering/triangulation."""
    from scipy.spatial import ConvexHull
    a,b=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    for points,other in [(a,b),(b,a)]:
        planes=ConvexHull(other).equations
        if np.max(points@planes[:,:3].T+planes[:,3])>tolerance_m:return False
    return True
