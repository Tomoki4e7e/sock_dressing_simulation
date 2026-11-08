"""Source binding, exact interpolation, and the unchanged A grasp evaluator."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from search_centered_reaching import REGISTERED_MANIFEST,geometric_check
from sock_dressing_simulation.common_opening_approach import load_state,sha256
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics
from sock_dressing_simulation.grasp_geometry import SkinBody
from sock_dressing_simulation.grasp_cloth_geometry import ClothScreen
from sock_dressing_simulation.geometric_closure import material_distances,projected_cloth_area
from validate_compound_trajectory import smooth_profile


def verify_source(path,initial,state):
    record=json.loads(Path(path).read_text())
    if not record['passed'] or record['initial_sha256']!=sha256(initial) or record['condition']['state_sha256']!=sha256(state) or record['geometry_manifest_sha256']!=sha256(REGISTERED_MANIFEST):
        raise ValueError('Unqualified or stale immutable source')
    if Path(state).resolve()!=(ROOT/record['condition']['state']).resolve():raise ValueError('Wrong state path')
    selected=yaml.safe_load(Path(initial).read_text());side=record['side'];other='left' if side=='right' else 'right'
    for a,b in [(record['start_q_rad'],selected[side+'_q_rad']),(record['other_arm_q_rad'],selected[other+'_q_rad']),(record['translation_m'],selected['translation_m'])]:np.testing.assert_array_equal(a,b)
    return record


def quintic(t):
    return 10*t**3-15*t**4+6*t**5,30*t**2-60*t**3+30*t**4


def make_profile(route,model):
    duration=route['duration_s']
    if 'knots_q_rad' in route:
        knots=np.array(route['knots_q_rad']);durations=np.array(route['segment_durations_s']);ends=np.cumsum(durations)
        np.testing.assert_allclose(ends[-1],duration,atol=1e-10,rtol=0)
        def evaluate(t):
            i=min(int(np.searchsorted(ends,t,side='right')),len(durations)-1)
            tau=np.clip((t-(ends[i-1] if i else 0))/durations[i],0,1);u,du=quintic(tau);dq=knots[i+1]-knots[i]
            return knots[i]+dq*u,dq*du/durations[i]
    else:
        p=smooth_profile(route['raw_q_rad'],model.velocity,model.acceleration)
        np.testing.assert_allclose(p['q'],route['q_rad'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(p['duration'],duration,atol=1e-9,rtol=0)
        def evaluate(t):
            u,du=quintic(np.clip(t/duration,0,1));return p['curve'](u),p['curve'](u,1)*du/duration
    actual=np.array([evaluate(t)[0] for t in route['times_s']]);np.testing.assert_allclose(actual,route['q_rad'],atol=1e-10,rtol=0)
    return dict(duration=duration,evaluate=evaluate)


def motion_metrics(trace,initial,last,model,route,dt,fixed_drift,result,completed):
    native,tcp,R,actual,gap=last
    qs=np.array([r['actual_q_rad'] for r in trace]);gaps=np.array([r['actual_gap_m'] for r in trace]);v=np.diff(qs,axis=0)/dt;a=np.diff(v,axis=0)/dt
    value=dict(steps=len(trace),completed=bool(completed),endpoint_position_error_m=float(np.linalg.norm(tcp-route['target']['tcp_target_m'])),endpoint_orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(R@np.array(route['target']['rotation']).T).magnitude())),maximum_fixed_drift_m=fixed_drift,
        actual_hard_joint_limits_passed=bool(np.isfinite(qs).all() and np.all(qs>=model.lower-1e-8) and np.all(qs<=model.upper+1e-8)),minimum_actual_joint_margin_deg=float(np.rad2deg(np.minimum(qs-model.lower,model.upper-qs)).min()),maximum_velocity_fraction=float(np.max(abs(v)/model.velocity)) if len(v) else 0.,maximum_acceleration_fraction=float(np.max(abs(a)/model.acceleration)) if len(a) else 0.,maximum_gap_velocity_m_s=float(abs(np.diff(gaps)/dt).max()) if len(gaps)>1 else 0.,initial_clock=initial['cloth_time_s'],final_clock=native['cloth_time_s'],initial_steps=initial['cloth_fixed_steps'],final_steps=native['cloth_fixed_steps'])
    value['passed']=bool(completed and not result['unexpected_contacts'] and not result['actual_configuration_failures'] and value['endpoint_position_error_m']<=1e-4 and value['endpoint_orientation_error_deg']<=.5 and fixed_drift<2e-5 and value['actual_hard_joint_limits_passed'] and value['maximum_velocity_fraction']<=1 and value['maximum_acceleration_fraction']<=1 and value['maximum_gap_velocity_m_s']<=.02 and value['final_steps']-value['initial_steps']==len(trace))
    return value


def evaluate_grasp(record,state,k,checker,right,left,gap,delta):
    """Same geometry and thresholds as report_photo_grasp_adjustment.actual_recheck."""
    state=load_state(ROOT,record['condition']) if state is None else state
    side=record['side'];right_gap=gap if side=='right' else .04;left_gap=gap if side=='left' else .04
    checker.park_left(left,left_gap);checker.closure_side=side
    model=ControlledArmKinematics(k,side,right,right_gap);p,R=model.pose(left if side=='left' else right)
    sample=dict(record['route']['target']['sample'],position=(p-delta).tolist())
    faces=np.array(state['triangles']).reshape(-1,3);rest=np.array(state['source']['rest_positions']);dist=material_distances(rest,faces,faces[sample['face']],sample['weights'])
    settings=yaml.safe_load((ROOT/'config/geometric_closure.yaml').read_text());skin=SkinBody(state['source']['skin_surface']);cloth=ClothScreen(state['particles_world'],state['triangles'])
    geometry,reason=geometric_check(model,state,sample,R,skin,cloth,settings,dist,approach_distance=0.)
    fingers=[g for n,g in model.hand.vertices(gap).items() if n.endswith(('/left_finger_link','/right_finger_link'))];depth=rest[:,2]-rest[:,2].min()
    area=projected_cloth_area((np.array(state['particles_world'])-(p-delta))@R,faces,fingers,dist,depth,**{n:settings[n] for n in ('half_extent_m','edge_clearance_m','material_radius_m','band_width_m','topology_tolerance_m')})
    clear,distance,limiting,why=skin.inspect(model.hand.bank(gap),R,p-delta,.001)
    body=checker.slacks(right,right_gap,delta,allow_cloth_overlap=True,details=True)
    passed=bool(geometry and clear and body['passed'] and min(area['area_m2'],area['largest_component_area_m2'])>=4e-6 and area['area_fraction']>=.05 and min(area['span_x_m'],area['span_z_m'])>=.0005)
    return dict(passed=passed,moving_side=side,geometric_check_reason=reason,actual_closed_gap_m=gap,actual_projection=area,closed_skin_clearance_m=distance,limiting_part=limiting.replace('right_gripper/',side+'_gripper/'),closed_skin_reason=why,whole_scene_closed_check=body,criteria_file='config/geometric_closure.yaml',criteria_sha256=sha256(ROOT/'config/geometric_closure.yaml'),actual_right_q_rad=list(right),actual_left_q_rad=list(left),scope='Previous A geometric capture rule, frozen cloth; no friction or retention test')


def threshold_check(checker,right,gap,delta,closing):
    """Conservative fast gate; exact old slacks resolves every rejection.

    The fast query uses the same pair margins and exclusions. It can only
    shortcut a clear result, and the triangle shoulder guard uses its actual
    registered pair margin. A failed/touching fast test falls back to the old
    distance-feature evaluator, retaining the A tolerance at zero separation.
    """
    original=checker.pairs
    if closing:
        prefix=checker.closure_side+'_gripper/'
        checker.pairs=[(a,b,m) for a,b,m in original if not (b=='frozen_cloth' and a.startswith(prefix))]
    checker.use_spatial_aabb=True
    try:
        clear=checker.is_clear(right,gap,delta)
        if clear:
            clear=all(checker.shoulder_distance(right,side)>=original[i][2]-1e-9 for side,i in checker.shoulder_pairs.items())
    finally:checker.pairs=original
    if clear:return dict(passed=True,minimum_slack_m=None,issues=[],method='same_policy_threshold_gate')
    result=checker.slacks(right,gap,delta,allow_cloth_overlap=closing,details=True)
    result['method']='original_A_distance_features'
    return result
