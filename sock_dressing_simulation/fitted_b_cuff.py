"""B-specific frozen sock references with a continuous anatomical toe envelope.

Geometry is fitted to the native posed foot and continued beyond its tip into a
flatter empty sleeve. This is an explicit reference-shape construction, not a
claim of calibrated cloth deformation or gravitational equilibrium.
"""
import copy
from dataclasses import asdict
import numpy as np
import trimesh
from .held_ankle_poses import audit_actual_skin
from .preinsert import generate_preinsert
from .cuff_bulge import radial_bulge
from .natural_b_shape import natural_b_baseline,audit_toe_continuity

PROFILE='B_anatomical_continuous_toe_envelope_frozen_reference_v6'


def opening_perimeter(points,rest):
    ids=np.flatnonzero(np.isclose(rest[:,2],rest[:,2].min(),atol=1e-6,rtol=0))
    ids=ids[np.argsort(np.arctan2(rest[ids,1],rest[ids,0]))]
    return float(np.linalg.norm(points[ids]-np.roll(points[ids],1,axis=0),axis=1).sum())


def qualify_b_state(state,before,profile):
    state=copy.deepcopy(state);source=state['source'];rest=np.asarray(source['rest_positions'])
    points=np.asarray(state['particles_world']);before=np.asarray(before);depth=rest[:,2]-rest[:,2].min()
    envelope=state['toe_envelope'];continuity=audit_toe_continuity(before,rest,envelope)
    rim=np.isclose(depth,0.,atol=1e-8,rtol=0);peak=np.isclose(depth,.015,atol=1e-8,rtol=0)
    outside=(depth<=1e-10)|(depth>=.030-1e-10)
    center=np.array(profile['reference_center_world']);axis=np.array(profile['reference_axis_world'])
    def radius(p):
        v=p-center;return np.linalg.norm(v-axis*np.sum(v*axis,axis=1)[:,None],axis=1)
    increment=radius(points)-radius(before)
    skin=source['skin_surface'];mesh=trimesh.Trimesh(skin['vertices'],np.array(skin['triangles']).reshape(-1,3),process=False)
    _,distances,_=trimesh.proximity.closest_point_naive(mesh,points[rim])
    wrapped=(depth>=.03-1e-8)&(depth<=envelope['toe_envelope_anchor_depth_m']+1e-8)
    _,wrapped_distances,_=trimesh.proximity.closest_point_naive(mesh,before[wrapped])
    body,surface=audit_actual_skin(points,skin,state['edges'],state['triangles'])
    old,_=generate_preinsert(source,'B',clearance_m=.015)
    old_perimeter=opening_perimeter(old,rest);perimeter=opening_perimeter(points,rest)
    levels=np.unique(np.round(depth,6));centers=np.array([before[np.isclose(depth,z,atol=1e-6,rtol=0)].mean(0) for z in levels])
    root=int(np.flatnonzero(levels>=envelope['free_sleeve_bend_start_m']-1e-6)[0])
    distal=centers[-1]-centers[-3];distal_alignment=float(-distal[1]/np.linalg.norm(distal));tip_below=float(centers[root,1]-centers[-1,1])
    checks=dict(B_specific_forefoot_envelope=envelope['fit_method']=='B_forefoot_envelope_continued_over_anatomical_toe',
        natural_toe_envelope=continuity['passed'],free_end_points_down=distal_alignment>.8,free_tip_below_toe=tip_below>.02,
        opening_reduced=perimeter<.9*old_perimeter,opening_skin_median_within_8mm=bool(np.median(distances)<=.008),
        contact_region_fits_skin=bool(np.median(wrapped_distances)<=.008 and np.quantile(wrapped_distances,.95)<=.012),
        peak_section_present=bool(peak.any()),peak_radius_increment_5mm=bool(peak.any() and np.allclose(increment[peak],.005,atol=1e-12,rtol=0)),
        maximum_radius_increment_5mm=bool(np.isclose(increment.max(),.005,atol=1e-12,rtol=0)),
        outside_30mm_band_unchanged=bool(np.array_equal(points[outside],before[outside])),
        no_skin_intersection=body['geometry_ok'],no_self_intersection=surface['surface_ok'])
    fit=dict(passed=all(checks.values()),checks=checks,fit_clearance_m=envelope['clearance_m'],
        old_ellipse_opening_perimeter_m=old_perimeter,fitted_opening_perimeter_m=perimeter,
        opening_reduction_fraction=1-perimeter/old_perimeter,
        body_fit_method='actual_skin_contact_region_with_continuous_B_toe_envelope',
        opening_skin_distance_m=dict(minimum=float(distances.min()),median=float(np.median(distances)),maximum=float(distances.max())),
        wrapped_skin_distance_m=dict(median=float(np.median(wrapped_distances)),percentile95=float(np.quantile(wrapped_distances,.95)),maximum=float(wrapped_distances.max())),
        peak_radius_increment_min_m=float(increment[peak].min()),peak_radius_increment_max_m=float(increment[peak].max()),
        toe_continuity=continuity,toe_envelope=envelope,free_tip_below_toe_m=tip_below,free_distal_downward_alignment=distal_alignment,
        body=body,surface=surface,physical_cloth_deformation_validated=False)
    state.update(cuff_fit=fit,shape_profile=PROFILE,placement=dict(stage='B',fit_method=fit['body_fit_method'],
        cuff_center_world=points[rim].mean(axis=0).tolist(),opening_inward_normal_world=state['opening_normal'],
        physical_settling_validated=False,material_rest_lengths_modified=False))
    if not fit['passed']:raise ValueError('B shape qualification failed: '+str(checks))
    return state


def build_fitted_b_state(original,pattern,clearances=(.0025,.0035,.005,.006)):
    """Fit B's own foot and garment outline; never reuse A's 54% scale."""
    attempts=[]
    for clearance in clearances:
        before,envelope,_=natural_b_baseline(original['source'],pattern,clearance)
        rest=np.asarray(original['source']['rest_positions'])
        after,profile=radial_bulge(rest,before,rest,before,width=.030,peak=.005)
        state=copy.deepcopy(original)
        for key in ('native_preparation','cuff_fit','bulge','placement'):state.pop(key,None)
        state.update(mode='frozen_B_anatomical_continuous_toe_envelope',pattern=asdict(pattern),
            applied_opening_stage='B',shape_profile=PROFILE,toe_envelope=envelope,particles_world=after.tolist(),
            opening_normal=envelope['opening_inward_normal_world'],scene_number=original.get('scene_number',original['pattern']['number']),
            settings=dict(generator='natural_b_baseline',shape_model='B_forefoot_envelope',physical_cloth_deformation_validated=False),
            bulge=dict(width_m=.030,center_m=.015,peak_radius_increment_m=.005,applied_after_B_specific_shape=True,fixed_geometry_only=True))
        try:state=qualify_b_state(state,before,profile)
        except ValueError as error:
            attempts.append(dict(clearance_m=clearance,passed=False,reason=str(error)));continue
        attempts.append(dict(clearance_m=clearance,passed=True));state['cuff_fit']['exploration']=attempts
        return state,before,profile
    raise ValueError('No natural B reference passed: '+str(attempts))
