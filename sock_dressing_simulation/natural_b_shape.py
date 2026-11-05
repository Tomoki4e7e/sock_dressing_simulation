"""Stage-B toe envelope: continuous garment sections across the foot tip.

The skin's distal cap is not the garment's cap. With cloth extending beyond the
toes, shrinking every ring to that skin cap produces an artificial bottleneck.
Keep a forefoot-sized envelope, relax it gradually into a flatter empty sleeve,
and round the garment's own closed end. These are explicit shape assumptions,
not calibrated textile mechanics.
"""
from dataclasses import replace
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from .pose_patterns import _tight_base


def smoothstep(value):
    q=np.clip(value,0.,1.)
    return q**3*(10-15*q+6*q*q)


def section_perimeters(points,rest):
    points=np.asarray(points);rest=np.asarray(rest)
    depth=rest[:,2]-rest[:,2].min();levels=np.unique(np.round(depth,6))
    values=[]
    for level in levels:
        ids=np.flatnonzero(np.isclose(depth,level,atol=1e-6,rtol=0)&(np.linalg.norm(rest[:,:2],axis=1)>1e-7))
        ids=ids[np.argsort(np.arctan2(rest[ids,1],rest[ids,0]))]
        ring=points[ids]
        values.append(float(np.linalg.norm(ring-np.roll(ring,1,axis=0),axis=1).sum()))
    return levels,np.array(values)


def natural_b_baseline(source,pattern,clearance_m=.0025,free_perimeter_ratio=.9,flatness=.2):
    """Derive the toe/free sleeve from B's own covered forefoot section."""
    if not .85<=free_perimeter_ratio<=1 or not .15<=flatness<=.7:
        raise ValueError('Natural B envelope parameters outside bounded exploration')
    normal=replace(pattern,stage='B',shift_cm=0.,diagonal_deg=0.,opening='通常',wrinkles='ほぼなし',fold='なし')
    points,placement,frames=_tight_base(source,normal,clearance_m,.05,dict(shift=1,diagonal=1,roll=1),section_surfaces=[source['skin_surface']])
    rest=np.asarray(source['rest_positions']);depth=rest[:,2]-rest[:,2].min();length=float(depth.max())
    levels,circumferences=section_perimeters(points,rest)
    # Start the envelope before the skin narrows into its distal cap. The
    # anatomical insertion depth changes with B's ankle pose.
    target=max(.05,placement['insertion_path_m']-.045)
    index=int(np.flatnonzero(levels<=target+1e-8)[-1]);anchor=levels[index]
    anchor_frame=frames[index];ids=anchor_frame['ids'];theta=anchor_frame['theta']
    anchor_values=points[ids]-anchor_frame['center']
    outline=np.c_[anchor_values@anchor_frame['lateral'],anchor_values@anchor_frame['cross']]
    order=np.argsort(theta);angles=theta[order];outline=outline[order]
    forefoot_perimeter=float(circumferences[index]);free_perimeter=free_perimeter_ratio*forefoot_perimeter
    transition_end=min(length-.05,anchor+.10)
    cap_start=length-.025
    bend_index=int(np.flatnonzero(levels>=placement['insertion_path_m']-1e-8)[0])
    bend_start=levels[bend_index];bend_frame=frames[bend_index]
    outward=-bend_frame['axis'];down=np.array([0.,-1.,0.])
    bend_axis=np.cross(outward,down);bend_axis/=np.linalg.norm(bend_axis)
    turn_angle=float(np.arccos(np.clip(outward@down,-1.,1.)))
    bend_radius=float(np.clip(.26*(length-bend_start),.025,.045))
    perpendicular=np.cross(bend_axis,outward)
    for frame,level in zip(frames,levels):
        if level<=anchor+1e-8:continue
        ids=frame['ids'];theta=frame['theta'];active=~frame['cap']
        loop_angles=np.r_[angles[-1]-2*np.pi,angles,angles[0]+2*np.pi]
        loop=np.vstack([outline[-1],outline,outline[0]])
        reference=np.column_stack([np.interp(theta,loop_angles,loop[:,i]) for i in range(2)])
        reference/=forefoot_perimeter
        ellipse=np.c_[np.cos(theta),flatness*np.sin(theta)]
        sorted_ids=np.argsort(theta[active]);ellipse_ring=ellipse[active][sorted_ids]
        ellipse/=np.linalg.norm(ellipse_ring-np.roll(ellipse_ring,1,axis=0),axis=1).sum()
        blend=float(smoothstep((level-anchor)/(transition_end-anchor)))
        local=(1-blend)*reference+blend*ellipse
        ring=local[active][sorted_ids]
        local/=np.linalg.norm(ring-np.roll(ring,1,axis=0),axis=1).sum()
        perimeter=(1-blend)*forefoot_perimeter+blend*free_perimeter
        # Taper only at the garment's closed end, not at the anatomical toes.
        cap_fraction=np.clip((level-cap_start)/(length-cap_start),0.,1.)
        perimeter*=np.sqrt(1-(.94*cap_fraction)**2)
        local*=perimeter
        if level<=placement['insertion_path_m']:
            # A smooth forefoot template must still enclose B's actual local
            # skin section. Add only the necessary toe-sector reserve instead
            # of inflating the cuff and whole garment to clear one toe corner.
            original=points[ids]-frame['center']
            skin_outline=np.c_[original@frame['lateral'],original@frame['cross']]
            hull=ConvexHull(np.vstack([local[active],skin_outline[active]]))
            directions=np.c_[np.cos(theta),np.sin(theta)]
            den=directions@hull.equations[:,:2].T
            bounds=-hull.equations[:,2]
            radii=np.min(np.divide(bounds[None,:],den,out=np.full_like(den,np.inf),where=den>1e-10),axis=1)
            local=directions*(radii/np.cos(np.pi/active.sum())+.0003)[:,None]
        if abs(level-bend_start)<1e-6:
            extent=max(abs(local[:,0]*(perpendicular@bend_frame['lateral'])+local[:,1]*(perpendicular@bend_frame['cross'])))
            bend_radius=max(bend_radius,1.2*float(extent)+.003)
        center=frame['center'];lateral=frame['lateral'];cross=frame['cross']
        if level>bend_start:
            distance=level-bend_start;angle=min(turn_angle,distance/bend_radius)
            center=bend_frame['center']+bend_radius*(np.sin(angle)*outward+(1-np.cos(angle))*perpendicular)
            center+=max(0.,distance-bend_radius*turn_angle)*down
            rotation=Rotation.from_rotvec(bend_axis*angle).as_matrix()
            lateral=rotation@bend_frame['lateral'];cross=rotation@bend_frame['cross']
        values=center+local[:,:1]*lateral+local[:,1:]*cross
        values[frame['cap']]=center;points[ids]=values
        frame.update(center=center,lateral=lateral,cross=cross)
    for ring,frame in zip(placement['rings'],frames):
        values=points[frame['ids']];center=values.mean(axis=0);relative=values-center
        ring.update(center_world=center.tolist(),half_width_m=float(np.max(abs(relative@frame['lateral']))),half_height_m=float(np.max(abs(relative@frame['cross']))))
    placement.update(cuff_center_world=points[frames[0]['ids']].mean(axis=0).tolist(),opening_shape='actual_skin_section_polygon',
        toe_envelope_anchor_depth_m=float(anchor),toe_envelope_anchor_perimeter_m=forefoot_perimeter,
        free_sleeve_perimeter_m=free_perimeter,free_perimeter_ratio=free_perimeter_ratio,free_sleeve_flatness=flatness,
        toe_transition_end_m=float(transition_end),garment_cap_start_m=float(cap_start),
        anatomical_tip_material_depth_m=placement['insertion_path_m'],
        free_sleeve_bend_start_m=float(bend_start),free_sleeve_bend_radius_m=bend_radius,
        free_sleeve_bend_model='smooth continuation around anatomical toe toward gravity',
        material_scale_estimate=free_perimeter/(2*np.pi*float(np.median(np.linalg.norm(rest[np.linalg.norm(rest[:,:2],axis=1)>1e-7,:2],axis=1)))),
        fit_method='B_forefoot_envelope_continued_over_anatomical_toe')
    return points,placement,frames


def audit_toe_continuity(points,rest,envelope):
    levels,circumferences=section_perimeters(points,rest)
    anchor=envelope['toe_envelope_anchor_depth_m'];end=envelope['garment_cap_start_m']
    active=(levels>=anchor-1e-6)&(levels<=end+1e-6)
    values=circumferences[active];ratios=np.minimum(values[1:]/values[:-1],values[:-1]/values[1:])
    reference=envelope['toe_envelope_anchor_perimeter_m']
    checks=dict(no_anatomical_toe_bottleneck=bool(values.min()>=.80*reference),
        no_abrupt_section_change=bool(ratios.min()>=.85),
        no_sleeve_oversizing=bool(values.max()<=1.08*reference))
    return dict(passed=all(checks.values()),checks=checks,depth_m=levels.tolist(),perimeter_m=circumferences.tolist(),
        minimum_toe_to_forefoot_perimeter_ratio=float(values.min()/reference),
        minimum_adjacent_perimeter_ratio=float(ratios.min()))

