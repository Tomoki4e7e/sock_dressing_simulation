"""Native cloth preparation with a small anatomical toe support patch."""
from __future__ import annotations

import copy
from collections import deque
from pathlib import Path

import numpy as np


def select_toe_support(points, source, *, band_width_m=.030, support_length_m=.015,
                       support_clearance_m=.008):
    """Find the actual foot's distal cap, not the end of the hanging sock.

    The toe bone is proximal to the visible toe end. Use the current foot skin
    hull and reject supports in the grasp band rather than pinning the cuff.
    """
    from scipy.spatial import ConvexHull
    points = np.asarray(points, dtype=float)
    rest = np.asarray(source['rest_positions'], dtype=float)
    ankle = np.asarray(source['ankle'], dtype=float)
    forward = np.asarray(source['toe'], dtype=float)-ankle
    forward /= np.linalg.norm(forward)
    foot = np.asarray(source['surfaces'][0]['vertices'], dtype=float)
    tip = float(np.max((foot-ankle)@forward))
    axial = (points-ankle)@forward
    planes = ConvexHull(foot).equations
    signed = np.max(points@planes[:, :3].T+planes[:, 3], axis=1)
    material = rest[:, 2]-rest[:, 2].min()
    support = np.flatnonzero((axial >= tip-support_length_m) & (axial <= tip+support_clearance_m)
        & (material > band_width_m+1e-6) & (signed <= support_clearance_m) & (signed >= -.001))
    if not 3 <= len(support) <= len(points)//5:
        raise ValueError(f'No small toe support patch outside grasp band: {len(support)} particles')
    return dict(particle_indices=support.tolist(), foot_tip_axial_m=tip,
        support_length_m=support_length_m, support_clearance_m=support_clearance_m,
        material_min_m=float(material[support].min()), material_max_m=float(material[support].max()))


def assess_toe_settling(samples, particle_count):
    if len(samples)<6:
        return dict(sufficient_samples=False)
    final=samples[-1]
    band=np.asarray([s['band_positions_world'] for s in samples[-6:]], dtype=float)
    finite=band.ndim==3 and band.shape[2]==3 and np.isfinite(band).all()
    excursion=float(np.linalg.norm(band-band[0],axis=2).max()) if finite else float('inf')
    hybrid=final.get('physics_model')=='native_obi_band_with_dynamic_section_tail'
    return dict(native_dynamic_cloth=(final.get('physics_model')=='native_obi_cloth_toe_support'
            and final.get('mesh_vertices_are_followers') is False) or (hybrid
            and final.get('band_vertices_are_followers') is False
            and final.get('band_native_finite_particles')==len(final['band_particle_indices'])
            and final.get('dynamic_tail_sections',0)>0),
        limited_toe_support=3<=final['fixed_particles']<=particle_count//5,
        all_other_particles_dynamic=final['native_mesh_finite_mass_particles']+final.get('dynamic_tail_follower_particles',0)==particle_count-final['fixed_particles'],
        free_grasp_band=final['band_fixed_particles']==0,
        right_finger_colliders=final['right_finger_colliders_enabled']>0,
        no_outside_band_contacts=final['outside_band_contacts']==0,
        gravity=bool(np.allclose(final['gravity_world'], [0,-9.81,0])),
        time_advanced=final['physics_steps']>samples[0]['physics_steps'] and final['physics_time_s']>samples[0]['physics_time_s'],
        support_stable=final['maximum_fixed_error_m']<5e-6,
        band_settled=finite and excursion<.005 and final['maximum_band_speed_m_s']<.05,
        free_cloth_settled=final['maximum_free_speed_m_s']<.05 or (
            final['maximum_free_speed_m_s']<.1
            and final.get('whole_cloth_recent_window_s',0)>=1.-1e-5
            and final.get('whole_cloth_recent_excursion_m',float('inf'))<.002))


def apply_toe_contact(environment, reference, source, report, settings):
    from .held_ankle_poses import audit_actual_skin, _perimeter
    attr=environment.sock_cloth
    attr.request_particles();environment._env.step(simulate=False)
    support=select_toe_support(attr.particles(), source,
        band_width_m=float(settings.get('band_width_m',.030)),
        support_length_m=float(settings.get('support_length_m',.015)),
        support_clearance_m=float(settings.get('support_clearance_m',.008)))
    attr.set_pattern_held(False);environment._env.step(simulate=False)
    foot,calf=source['surfaces']
    attr.install_fit_hull(foot['vertices'],foot['triangles'],calf['vertices'],calf['triangles'])
    environment._env.step(simulate=False)
    state=(Path(reference['directory'])/'fitted-prepared-state.json').read_text()
    attr._send_data('BeginSockToeContact',state,support['particle_indices'],float(settings.get('band_width_m',.030)))
    environment._env.step(simulate=False)
    initial=copy.deepcopy(attr.data.get('sock_toe_contact',{}))
    if not initial.get('active'):
        raise RuntimeError('BeginSockToeContact returned no active report; inspect Player errors or rebuild it')
    skin_enabled=settings.get('rigid_skin_contact',True)
    if skin_enabled:
        scene=environment.config['scene']
        policy=2 if scene.get('ignore_robot_human_rigid_collisions',True) else (1 if scene.get('ignore_non_gripper_robot_human_rigid_collisions',False) else 0)
        attr._send_data('InstallSockRigidSkinContact',int(environment.config['assets']['robot_id']),policy)
        environment._env.step(simulate=False)
        if not attr.data.get('sock_rigid_skin_contact',{}).get('active'):
            raise RuntimeError('Actual skin rigid contact requires a Player built with SockRigidSkinContact.cs')
    samples=[]
    recent=deque(maxlen=6)
    steps=int(settings.get('settle_steps',300))
    for first in range(0,steps,10):
        environment.advance_physics(min(10,steps-first))
        attr._send_data('GetSockToeContact');attr.request_particles();environment._env.step(simulate=False)
        sample=copy.deepcopy(attr.data['sock_toe_contact'])
        recent.append((sample['physics_time_s'],attr.particles().copy()))
        sample['whole_cloth_recent_window_s']=recent[-1][0]-recent[0][0]
        sample['whole_cloth_recent_excursion_m']=float(max(np.linalg.norm(p-recent[0][1],axis=1).max() for _,p in recent))
        samples.append(sample)
        if (first+10)%50==0:
            row=samples[-1]
            print(f'toe contact {first+10}/{steps}: free_speed={row["maximum_free_speed_m_s"]:.4g} m/s '
                  f'band_speed={row["maximum_band_speed_m_s"]:.4g} m/s contacts={row["native_right_finger_contacts"]}',flush=True)
    attr.request_particles();environment._env.step(simulate=False)
    points=attr.particles()
    body,surface=audit_actual_skin(points,source['skin_surface'],attr.data['particle_edges'],attr.data['particle_surface_triangles'])
    boundary_trials=[]
    if samples[-1].get('physics_model')=='native_obi_band_with_dynamic_section_tail':
        boundary_trials.append(dict(blend_intervals=4,body=body,surface=surface))
        if body['geometry_ok'] and not surface['surface_ok']:
            for blend in (2.,3.,6.,8.):
                attr._send_data('SetSockPatternDrapeBoundaryBlend',blend);environment._env.step(simulate=False)
                environment.advance_physics(5)
                attr._send_data('GetSockToeContact');attr.request_particles();environment._env.step(simulate=False)
                samples.append(copy.deepcopy(attr.data['sock_toe_contact']))
                points=attr.particles()
                body,surface=audit_actual_skin(points,source['skin_surface'],attr.data['particle_edges'],attr.data['particle_surface_triangles'])
                boundary_trials.append(dict(blend_intervals=blend,body=body,surface=surface))
                if body['geometry_ok'] and surface['surface_ok']:break
    checks=assess_toe_settling(samples,len(points))
    checks.update(no_skin_intersection=body['geometry_ok'],no_self_intersection=surface['surface_ok'])
    if skin_enabled:
        attr._send_data('GetSockRigidSkinContact');environment._env.step(simulate=False)
        skin=copy.deepcopy(attr.data.get('sock_rigid_skin_contact',{}))
        checks['rigid_skin_matches']=assess_rigid_skin(skin,source['skin_surface'])
        report['rigid_skin_contact']=skin
        source['rigid_collision_surface']=dict(vertices=skin.get('vertices_world',[]),triangles=skin.get('triangles',[]),source='actual_baked_skin_triangle_mesh')
    perimeter,_=_perimeter(points,source['rest_positions'])
    support['seed_particle_indices']=support['particle_indices']
    support['particle_indices']=samples[-1]['toe_particle_indices']
    material=np.asarray(source['rest_positions'])[:,2]
    material-=material.min()
    support['material_min_m']=float(material[support['particle_indices']].min())
    support['material_max_m']=float(material[support['particle_indices']].max())
    result=dict(passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
                initial=initial,final=samples[-1],samples=samples,support=support,
                boundary_trials=boundary_trials,
                native_finger_contact_response_validated=False)
    report.update(motion_mode=samples[-1]['physics_model'],toe_contact=result,
                  native_hold=dict(mode='small_anatomical_toe_patch',enabled=True,
                      fixed_particles=samples[-1]['fixed_particles'],free_particles=samples[-1]['free_particles']),
                  geometry_audit=body,surface_audit=surface,cuff_perimeter_m=perimeter,
                  physical_settling_validated=result['passed'],
                  validated_physics_model=samples[-1]['physics_model'] if result['passed'] else None,
                  reference_rigid_checks_apply_to='before_native_settling')
    environment._toe_contact_debug=dict(report=report,points=points.tolist(),source=source,
        edges=attr.data['particle_edges'],triangles=attr.data['particle_surface_triangles'])
    if not result['passed']:
        raise RuntimeError(f'Native toe contact settling failed: {result["checks"]}')
    return report


def assess_rigid_skin(report, reference):
    """Check native collider readback against the skin used by geometric search."""
    vertices=np.asarray(report.get('vertices_world',[]),dtype=float)
    expected=np.asarray(reference['vertices'],dtype=float)
    return bool(report.get('active') and report.get('enabled')
        and not report.get('convex',True) and not report.get('trigger',True)
        and not report.get('obi_collider_added',True)
        and report.get('old_proxy_colliders_disabled',0)>0
        and report.get('old_proxy_colliders_still_enabled',1)==0
        and report.get('maximum_skin_vertex_error_m',1)<5e-6
        and report.get('triangles_match_baked_skin')
        and (report.get('collision_policy',2)==2 or (report.get('right_gripper_pairs_enabled',0)>0 and report.get('right_gripper_pairs_ignored',1)==0))
        and vertices.shape==expected.shape and np.allclose(vertices,expected,atol=5e-6,rtol=0)
        and np.array_equal(report.get('triangles',[]),reference['triangles']))
