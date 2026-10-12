#!/usr/bin/env python3
"""Verify autonomous command/video provenance and report motion acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def verify(directory, regression_path, real_mesh_path=None):
    metadata_files = list(directory.glob('data*/train/*/metadata.json'))
    if len(metadata_files) != 1:
        raise RuntimeError('Expected exactly one autonomous episode')
    metadata_path = metadata_files[0]
    metadata = json.loads(metadata_path.read_text())
    live = rows(directory / 'live_frame_diagnostics.jsonl')
    sync = rows(directory / 'physics-video-sync.jsonl')
    motion = rows(directory / 'grasp-substeps.jsonl')
    initial_clock=metadata.get('rollout_initial_cloth_physics_clock',{})
    motion=[r for r in motion if r['fixed_step']>=initial_clock.get('fixed_steps',0)]
    regressions = json.loads(regression_path.read_text())
    real_mesh_path=real_mesh_path or directory.parent.parent/'paired-real-mesh-probe.json'
    mesh_rows=json.loads(real_mesh_path.read_text()) if real_mesh_path.exists() else []
    real_mesh={}
    for row in mesh_rows:
        late=row['samples'][-40:]
        real_mesh[row['self_contact_geometry']]=dict(
            self_collision=row['self_collision'],surface_collisions=row['surface_collisions'],
            mean_speed_p95_m_s=float(np.mean([s['speed_p95_m_s'] for s in late])),
            mean_kinetic_energy_j=float(np.mean([s['kinetic_energy_j'] for s in late])),
            center_height_range_m=float(np.ptp([s['center_height_m'] for s in late])),
            samples=len(row['samples']))
    comparisons = {}
    for scenario in sorted({r['scenario'] for r in regressions}):
        baseline = next(r for r in regressions if r['scenario'] == scenario and not r['interpolated'])
        fixed = next(r for r in regressions if r['scenario'] == scenario and r['interpolated'])
        reductions = {}
        for key in ('relative_speed_p95_m_s', 'stop_relative_energy_j'):
            reductions[key] = None if baseline[key] < 1e-12 else 1 - fixed[key] / baseline[key]
        comparisons[scenario] = dict(baseline=baseline, fixed=fixed, reductions=reductions,
            reduction_gate_pass=all(v is not None and v >= .5 for v in reductions.values()))
    checks = {
        '30_live_inferences': metadata.get('frames_inferred') == 30 and metadata.get('frames') == 30,
        'no_replayed_commands': metadata.get('frames_replayed') == 0 and metadata.get('replay_source') is None,
        'completed': metadata.get('stop_reason') == 'max_steps' and len(live) == 30,
        'model_policy': metadata.get('policy_source') == 'SAMDAMSARNNPolicy',
        'no_reference_blend': metadata.get('reference_action_projection', {}).get('blend') == 0,
        'no_cartesian_assistance': metadata.get('reference_action_projection', {}).get('cartesian_pull_m') == 0,
        'no_policy_catch_up': not metadata.get('policy_catch_up', {}).get('enabled', False),
        'capture_preserved_time': bool(sync) and all(r['clock_before'] == r['clock_after'] for r in sync),
        'every_fixed_step_recorded': bool(sync) and all(
            b['clock_before']['fixed_steps'] == a['clock_before']['fixed_steps'] + 1
            for a, b in zip(sync, sync[1:])),
        '50_percent_reproduction_reduction': comparisons.get('anchor_collision', {}).get('reduction_gate_pass', False),
    }
    scenarios = {'stationary', 'constant', 'accelerated', 'asymmetric', 'release',
                 'reversals', 'anchor_collision', 'rotation', 'gravity_hang', 'gravity_release'}
    checks['complete_regression_suite'] = len(regressions) == 20 and all(
        sum(r['scenario'] == scenario and r['interpolated'] == mode for r in regressions) == 1
        for scenario in scenarios for mode in (False, True))
    settling_path=directory/'settling/validation.json'
    settling=json.loads(settling_path.read_text()) if settling_path.exists() else {}
    checks['physically_settled_initial_state']=settling.get('passed',False) and settling.get('policy_commands')==0
    old_mesh=real_mesh.get('surface_simplices',{})
    fixed_mesh=real_mesh.get('vertices',{})
    checks['real_mesh_stationary_stability']=bool(fixed_mesh) and (
        fixed_mesh['self_collision'] and fixed_mesh['surface_collisions'] and fixed_mesh['samples']==120 and
        fixed_mesh['mean_speed_p95_m_s']<=.08 and fixed_mesh['center_height_range_m']<=.002)
    checks['real_mesh_50_percent_reduction']=bool(old_mesh) and bool(fixed_mesh) and all(
        fixed_mesh[k]<=old_mesh[k]*.5 for k in ('mean_speed_p95_m_s','mean_kinetic_energy_j'))
    checks['gravity_hanging'] = all(
        r['final_center_height_m'] < r['initial_center_height_m'] - .05
        for r in regressions if r['interpolated'] and r['scenario'].startswith('gravity_')
    ) and sum(r['interpolated'] and r['scenario'].startswith('gravity_') for r in regressions) == 2
    checks['gravity_release'] = any(
        r['interpolated'] and r['scenario'] == 'gravity_release'
        and r['release_mean_velocity_error_m_s'] < .005 for r in regressions)
    checks['regression_pin_accuracy'] = all(
        r['maximum_pin_error_m'] <= .001 for r in regressions if r['interpolated'])
    checks['regression_pin_velocity'] = any(
        r['interpolated'] and r['scenario'] == 'constant'
        and r['constant_pin_velocity_error_m_s'] <= .01 for r in regressions)
    initial = json.loads((directory / 'foot-initial-state.json').read_text())
    def toe_distance(row):
        geometry = row['scene_geometry']
        return float(np.linalg.norm(np.asarray(geometry['opening_center']) - np.asarray(geometry['right_toe_position'])))
    reaching = dict(initial_distance_m=toe_distance(initial),
                    final_distance_m=toe_distance(live[-1]) if live else None)
    checks['reached_toward_toe'] = bool(live) and reaching['final_distance_m'] < reaching['initial_distance_m'] - .001
    violations = []
    for row in live:
        safety = row['dressing_qa']['foot_collision_safety']
        for key in ('remaining_surface_intersections', 'remaining_swept_intersections', 'remaining_swept_surface_intersections'):
            if safety.get(key) != 0:
                violations.append(dict(frame=row['frame'], check=key, value=safety.get(key)))
        penetration = row['dressing_qa'].get('geometric_maximum_cloth_foot_penetration_m')
        if penetration is None or penetration > .00200001:
            violations.append(dict(frame=row['frame'], check='penetration', value=penetration))
        if safety.get('maximum_grasp_offset_drift_m', float('inf')) > 1e-6:
            violations.append(dict(frame=row['frame'], check='binding_offset_changed'))
        anchor_error = safety.get('maximum_original_grasp_anchor_error_m', float('inf'))
        anchor_limit = .001
        if not np.isfinite(anchor_error) or anchor_error > anchor_limit:
            violations.append(dict(frame=row['frame'], check='original_anchor_error', value=anchor_error))
        for name, edge in row['stretch'].get('edge_classes', {}).items():
            if edge.get('edge_count') == 0:
                continue
            limit = {'opening_rim': 2., 'opening_body': 1.1}.get(name, 1.5)
            if edge.get('maximum_stretch', float('inf')) > limit * 1.01:
                violations.append(dict(frame=row['frame'], check='stretch_'+name, value=edge.get('maximum_stretch')))
        grasps = row.get('grasp_state', [])
        if len(grasps) != 2 or any(len(g.get('particle_indices', [])) != 5 for g in grasps):
            violations.append(dict(frame=row['frame'], check='expected_ten_grasp_particles'))
        for grasp in grasps:
            if not grasp.get('attached'):
                violations.append(dict(frame=row['frame'], check='grasp_released', side=grasp.get('side')))
    checks['geometry_and_grasp_safe'] = len(live) == 30 and not violations
    quality = metadata.get('cloth_quality_by_frame', [])
    checks['distal_cloth_following'] = len(quality) == 30 and all(q.get('following_ok', False) for q in quality)
    checks['finite_motion_diagnostics'] = bool(motion) and all(
        all(key in r and np.isfinite(r[key]) for key in (
            'center_of_mass_height_m', 'free_mean_vertical_velocity_m_s',
            'free_maximum_upward_velocity_m_s', 'kinetic_energy_j', 'pin_error_m',
            'pin_velocity_error_m_s')) for r in motion)
    checks['anchors_fixed_to_real_grippers']=bool(motion) and all(
        r.get('grasp_frame_local_position_drift_m',float('inf'))<=1e-6 and
        r.get('grasp_frame_local_rotation_drift_deg',float('inf'))<=.001 for r in motion)
    checks['preparation_templates_released']=len(live)==30 and all(
        r['dressing_qa']['foot_collision_safety'].get('opening_rim_shape_weight')==0 and
        r['dressing_qa']['foot_collision_safety'].get('opening_body_barrier_effective_weight')==0 for r in live)
    checks['self_contacts_and_external_surfaces_preserved']=len(live)==30 and all(
        r['dressing_qa']['foot_collision_safety'].get('self_contact_geometry')=='vertices' and
        r['dressing_qa']['foot_collision_safety'].get('external_contact_geometry')=='surface_simplices' and
        r['dressing_qa']['foot_collision_safety'].get('native_collision_points',0)>0 and
        r['dressing_qa']['foot_collision_safety'].get('native_collision_triangles',0)>0 for r in live)
    checks['convex_solids_and_scoped_grasp_contacts']=len(live)==30 and all(
        r['dressing_qa']['foot_collision_safety'].get('native_convex_solid_collider_count',0)>0 and
        r['dressing_qa']['foot_collision_safety'].get('grasp_patch_contact_exclusion')=='inserted_cuff_against_own_jaws' and
        r['dressing_qa']['foot_collision_safety'].get('diagnostic_ablation_mode')=='none' for r in live)
    checks['no_bulk_cloth_translation']=len(live)==30 and all(
        r['dressing_qa']['foot_collision_safety'].get('bulk_grasp_transport') is False for r in live)
    videos = {}
    paths = [directory / 'side_physics_50fps.mp4', directory / 'side_demo.mp4',
             directory / 'top_demo.mp4', directory / 'bottom_demo.mp4',
             metadata_path.parent / 'demo.mp4', metadata_path.parent / 'inference_camera.mp4']
    for path in paths:
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS); decoded = 0
        while capture.read()[0]:
            decoded += 1
        capture.release()
        expected = len(sync) if path.name == 'side_physics_50fps.mp4' else 30
        videos[path.name] = dict(path=str(path), fps=fps, decoded_frames=decoded,
            expected_frames=expected, valid=decoded == expected and decoded > 0,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None)
    checks['all_videos_decodable_and_complete'] = all(v['valid'] for v in videos.values())
    checks['physics_video_50_fps'] = abs(videos['side_physics_50fps.mp4']['fps'] - 50) < 1e-6
    diagnostics = {}
    if motion:
        for key in ('anchor_speed_m_s', 'pin_error_m', 'pin_velocity_error_m_s',
                    'free_maximum_upward_velocity_m_s', 'free_mean_vertical_velocity_m_s',
                    'center_of_mass_height_m', 'kinetic_energy_j', 'relative_speed_p95_m_s'):
            values = [r[key] for r in motion if key in r]
            diagnostics[key] = dict(minimum=min(values), maximum=max(values)) if values else None
    report = dict(checks=checks, passed=all(checks.values()), violations=violations,
        comparisons=comparisons, videos=videos, metadata=str(metadata_path),
        motion_substeps=len(motion), dressing_success=metadata.get('task_success'),
        regression_source=str(regression_path), reaching=reaching, motion_diagnostics=diagnostics,
        settling_validation=str(settling_path))
    report['real_mesh_comparison']=real_mesh
    report['real_mesh_source']=str(real_mesh_path)
    (directory / 'grasp-following-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    failed = [key for key, passed in checks.items() if not passed]
    reproduction = comparisons['anchor_collision']['reductions']
    anchor_errors = [r['dressing_qa']['foot_collision_safety']['maximum_original_grasp_anchor_error_m'] for r in live]
    (directory / 'REPORT.md').write_text(
        '# pin追従・30step自律デモ検証\n\n'
        + ('全検査合格。' if report['passed'] else '未合格の検査: ' + ', '.join(failed))
        + '\n\nnative pinと補助pin補正を同じサブステップ位置に整合させた。仮想アンカー球の衝突無効化を維持した。'
        + '材料制限前の繰り返し追加収縮を除き、布全体の強制移送を無効にした。摩擦・減衰・質量・把持オフセットを維持する。'
        + '\n\n同一初期状態・同一入力の実Obi再現試験: 相対速度95パーセンタイル '
        + f"{100*reproduction['relative_speed_p95_m_s']:.2f}%低減、停止後の相対運動エネルギー "
        + f"{100*reproduction['stop_relative_energy_j']:.2f}%低減。"
        + f"\n\n自律推論 {metadata.get('frames_inferred')} 回、指令再生 {metadata.get('frames_replayed')} 回、"
        + f"完了記録 {len(live)} step。停止理由: `{metadata.get('stop_reason')}`。"
        + f"\n\n元アンカーからの最大誤差: {1000*max(anchor_errors, default=0):.3f} mm。"
        + f"\n\n開口中心からつま先までの距離: {1000*reaching['initial_distance_m']:.2f} mm → {1000*reaching['final_distance_m']:.2f} mm。"
        + '\n\n自己接触は粒子同士、足・グリッパーとの外部接触は三角形面で解く。'
        + '凸メッシュは閉じた凸形状として扱い、最近傍の接触点と外向きの法線で解く。'
        + '把持した挿入部分だけを自身の指との接触から除外し、解除時に復元する。'
        + '操作中のアンカーのグリッパー相対姿勢を保持し、準備用の矩形・平面拘束を解除した。'
        + '重力・既存減衰による静止待ちは settling/ に別記録し、30step推論区間に含めない。'
        + '\n\n50 fpsの物理時間側面映像: [side_physics_50fps.mp4](side_physics_50fps.mp4)。'
        + '正面・側面・上下・推論カメラの動画は検証JSONにパス、再生確認フレーム数、SHA256を保存。'
        + '\n\nrendererの物体IDを使う既存の観測設定では、接近による面積増加を同じ物体として扱う。'
        + '空・全画面・同一マスク、時間連続性、重なりの検査と、SAM2モードの面積事前分布による検査を保持した。'
        + '\n\n詳細は [grasp-following-validation.json](grasp-following-validation.json)、'
        + 'grasp-substeps.jsonl、physics-video-sync.jsonl を参照。'
        + '\n\n着衣成功は追従改善と別に評価する。停止後エネルギーの比較値は、40個の固定時刻における相対運動エネルギーの和。\n')
    if motion:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        times = np.array([r['time_s'] for r in motion]); times -= times[0]
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for field, label in (('raw_maximum_speed_m_s', 'Native max'),
                             ('final_maximum_speed_m_s', 'Accepted max'),
                             ('relative_speed_p95_m_s', 'Relative p95'), ('anchor_speed_m_s', 'Grasp speed')):
            axes[0].plot(times, [r[field] for r in motion], label=label)
        axes[0].set_ylabel('Speed (m/s)'); axes[0].legend()
        axes[1].plot(times, [r['relative_kinetic_energy_j'] for r in motion])
        axes[1].set_ylabel('Relative energy (J)'); axes[1].set_xlabel('Physics time (s)')
        figure.tight_layout(); figure.savefig(directory / 'grasp-motion.png'); plt.close(figure)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--regressions', type=Path, required=True)
    parser.add_argument('--real-mesh-regressions',type=Path)
    args = parser.parse_args()
    result = verify(args.directory, args.regressions, args.real_mesh_regressions)
    print(json.dumps({'passed': result['passed'], 'checks': result['checks']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
