#!/usr/bin/env python3
"""Collect both-arm opening routes, audit shared frames, and export timed CSVs."""
from __future__ import annotations

import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/common-opening-mpl')
import argparse
import csv
import html
import json
from pathlib import Path
import sys
import shutil

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from search_centered_reaching import write, URDF, ROBOT, REGISTERED_MANIFEST
from search_translated_reaching import load
from apron_reaching_common import overlay_robot
from sock_dressing_simulation.arm_reaching import Kinematics, ARM
from sock_dressing_simulation.fixed_torso_reaching import LEFT_ARM
from sock_dressing_simulation.controlled_arm_reaching import ControlledArmKinematics
from sock_dressing_simulation.common_opening_approach import inventory, sha256, load_state
from replay_controlled_photo_grasp import jaw_profile


def model_pair(initial):
    native = overlay_robot(load(10), json.loads(ROBOT.read_text()))
    k = Kinematics(URDF, native)
    k.q0.update(zip(LEFT_ARM, initial['left_q_rad']))
    k.q0['left_gripper/finger_joint'] = k.hand.joint_for_gap(initial['left_gap_m'])
    return k, {side: ControlledArmKinematics(k, side, initial['right_q_rad'], initial['right_gap_m'])
               for side in ('right', 'left')}


def fixed_frame_audit(result, initial, model):
    side = result['side']
    other = 'left' if side == 'right' else 'right'
    names = ['base_link']+[f'torso/link_{i}' for i in range(1, 4)]+[f'head/link_{i}' for i in range(1, 4)]
    names += [f'{other}_arm/link_{i}' for i in range(1, 8)]+[f'{other}_gripper/gripper_base']
    route = result['route']
    q = np.asarray(route['q_rad'])
    expected = model.fk(initial[side+'_q_rad'], initial[side+'_gap_m'], only=names)
    position, angle = 0., 0.
    for value in q[np.unique([0, len(q)//2, len(q)-1])]:
        frames = model.fk(value, route['opening_gap_m'], only=names)
        for name in names:
            position = max(position, float(np.linalg.norm(frames[name][:3, 3]-expected[name][:3, 3])))
            angle = max(angle, float(Rotation.from_matrix(frames[name][:3, :3]@expected[name][:3, :3].T).magnitude()))
    start_error = float(np.max(abs(q[0]-initial[side+'_q_rad'])))
    other_error = float(np.max(abs(np.asarray(result['other_arm_q_rad'])-initial[other+'_q_rad'])))
    translation_error = float(np.max(abs(np.asarray(result['translation_m'])-initial['translation_m'])))
    return dict(passed=max(position, angle, start_error, other_error, translation_error) < 1e-12,
        maximum_fixed_frame_position_error_m=position, maximum_fixed_frame_orientation_error_rad=angle,
        maximum_start_joint_error_rad=start_error, maximum_other_arm_joint_error_rad=other_error,
        maximum_human_translation_error_m=translation_error,
        checked_links=names, checked_route_indices=np.unique([0, len(q)//2, len(q)-1]).tolist())


def export_csv(path, result, initial, model):
    side = result['side']; route = result['route']
    q = np.asarray(route['q_rad']); times = np.asarray(route['times_s'])
    gap = route['opening_gap_m']; initial_gap = initial[side+'_gap_m']
    preopen = 0. if abs(gap-initial_gap) < 1e-10 else max(2., 1.875*abs(gap-initial_gap)/.01)
    if preopen:
        jaw_t = np.linspace(0., preopen, int(np.ceil(preopen/.05))+1)[:-1]
        widths, _ = jaw_profile(jaw_t, preopen, initial_gap, gap)
        q = np.vstack([np.repeat(q[:1], len(jaw_t), axis=0), q])
        times = np.r_[jaw_t, times+preopen]
        widths = np.r_[widths, np.full(len(route['q_rad']), gap)]
    else:
        widths = np.full(len(q), gap)
    right = q if side == 'right' else np.repeat([initial['right_q_rad']], len(q), axis=0)
    left = q if side == 'left' else np.repeat([initial['left_q_rad']], len(q), axis=0)
    right_gap = widths if side == 'right' else np.full(len(q), initial['right_gap_m'])
    left_gap = widths if side == 'left' else np.full(len(q), initial['left_gap_m'])
    tcp = np.asarray([model.pose(value)[0] for value in q])
    values = np.c_[times, right, left, right_gap, left_gap, tcp]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_s']+[name+'_rad' for name in ARM+LEFT_ARM]
                        +['right_gap_m', 'left_gap_m', 'tcp_x_m', 'tcp_y_m', 'tcp_z_m'])
        writer.writerows(values)
    assert np.array_equal(right[0], np.asarray(initial['right_q_rad']))
    assert np.array_equal(left[0], np.asarray(initial['left_q_rad']))
    return dict(file=str(path.relative_to(ROOT)), sha256=sha256(path), samples=len(times),
                preopen_duration_s=preopen, total_duration_s=float(times[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'artifacts/common-opening-approach')
    parser.add_argument('--merge-passed', action='store_true', help='Collect successful same-start variant searches')
    args = parser.parse_args(); out = args.root.resolve()
    initial = yaml.safe_load((out/'initial.yaml').read_text())
    initial_sha = sha256(out/'initial.yaml')
    states = inventory(ROOT)
    k, models = model_pair(initial)
    rows, accepted = [], []
    for state in states:
        for side in ('right', 'left'):
            path = out/'routes'/side/f'{state["id"]}.json'
            if args.merge_passed:
                current = json.loads(path.read_text()) if path.exists() else {}
                if not current.get('passed'):
                    for donor in sorted(out.glob(f'variant-*/routes/{side}/{state["id"]}.json')):
                        value = json.loads(donor.read_text())
                        if (value.get('passed') and value['initial_sha256'] == initial_sha
                                and value['condition']['state_sha256'] == state['state_sha256']
                                and value['geometry_manifest_sha256'] == sha256(REGISTERED_MANIFEST)):
                            if path.exists():
                                archive = out/'search-attempts'/side/f'{state["id"]}-before-merge.json'
                                archive.parent.mkdir(parents=True, exist_ok=True)
                                if not archive.exists(): shutil.copy2(path, archive)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            write(path, value)
                            break
            row = dict(id=state['id'], stage=state['stage'], number=state['number'],
                pitch_deg=state['pitch_deg'], roll_deg=state['roll_deg'], side=side,
                passed=False, status='pending', trajectory=str(path.relative_to(ROOT)))
            if path.exists():
                result = json.loads(path.read_text())
                if result['initial_sha256'] != initial_sha or result['condition']['state_sha256'] != state['state_sha256']:
                    raise ValueError(f'Stale route: {path}')
                if result['geometry_manifest_sha256'] != sha256(REGISTERED_MANIFEST):
                    raise ValueError(f'Stale collision geometry: {path}')
                row.update(status='passed' if result['passed'] else 'search_exhausted', counts=result['counts'])
                if result['passed']:
                    audit = fixed_frame_audit(result, initial, models[side])
                    if not audit['passed'] or not result['route']['validation']['passed']:
                        raise ValueError(f'Shared pose/frame validation failed: {path}')
                    csv_record = export_csv(out/'csv'/side/f'{state["id"]}.csv', result, initial, models[side])
                    row.update(passed=True, trajectory_sha256=sha256(path), validation=result['route']['validation'],
                        fixed_frame_audit=audit, csv=csv_record, target=result['route']['target'],
                        motion_duration_s=result['route']['duration_s'])
                    accepted.append(result)
            rows.append(row)
    counts = {side: sum(r['passed'] and r['side'] == side for r in rows) for side in ('right', 'left')}
    stage_counts = {s: {side: sum(r['passed'] and r['stage'] == s and r['side'] == side for r in rows)
                       for side in ('right', 'left')} for s in 'ABCDE'}
    passed = all(r['passed'] for r in rows)
    report = dict(all_passed=passed, expected_states=70, expected_routes=140,
        counts=counts, stage_counts=stage_counts, initial_configuration=str((out/'initial.yaml').relative_to(ROOT)),
        initial_sha256=initial_sha, reference=json.loads((out/'reference.json').read_text()),
        initial_tcp_world_m={s: models[s].pose(initial[s+'_q_rad'])[0].tolist() for s in ('right', 'left')},
        geometry_manifest_sha256=sha256(REGISTERED_MANIFEST), kinematic_calibration=k.calibration,
        scope='Fixed human, torso, head and base; one arm approaches while the other holds the same shared initial pose. Frozen cloth. Opening approaches only; closure, capture and physical replay are not validated.',
        continuous_collision_certified=False, learning_ready=False, rows=rows)
    native_path = out/'native-start/native-initial/result.json'
    if native_path.exists():
        native = json.loads(native_path.read_text())
        report['native_initial'] = dict(file=str(native_path.relative_to(ROOT)), sha256=sha256(native_path),
            passed=native['passed'], maximum_fixed_frame_position_error_m=native['maximum_fixed_frame_position_error_m'],
            maximum_planned_link_position_error_m=native['maximum_planned_link_position_error_m'],
            human_error_m=native['human_error_m'], cloth_error_m=native['cloth_error_m'])
    # The archived candidate YAML remains immutable because route hashes bind
    # it. Publish a separate verified configuration once all routes pass.
    if passed:
        verified = dict(initial, reachability_verified=True, verified_conditions_per_arm=70,
            verified_route_count=140, verification_summary=str((out/'summary.json').relative_to(ROOT)),
            source_candidate_configuration=str((out/'initial.yaml').relative_to(ROOT)),
            source_candidate_sha256=initial_sha)
        (out/'verified-initial.yaml').write_text(yaml.safe_dump(verified, allow_unicode=True, sort_keys=False))
        report['verified_initial_configuration'] = str((out/'verified-initial.yaml').relative_to(ROOT))
    write(out/'summary.json', report)
    fields = ['id', 'stage', 'number', 'pitch_deg', 'roll_deg', 'side', 'passed', 'status', 'motion_duration_s', 'trajectory']
    with (out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    if accepted:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_subplot(121, projection='3d'); planar = fig.add_subplot(122)
        palette = dict(left='#207bb5', right='#df7928')
        foot_state = load_state(ROOT, next(r for r in states if r['id'] == 'A-10'))
        foot = np.asarray(foot_state['source']['surfaces'][0]['vertices'])+initial['translation_m']
        ax.scatter(foot[:, 0], foot[:, 2], foot[:, 1], s=1, color='#777777', alpha=.12)
        planar.scatter(foot[:, 0], foot[:, 2], s=2, color='#bbbbbb', alpha=.25)
        for result in accepted:
            xyz = np.asarray(result['route']['tcp_world_m'])
            color = palette[result['side']]
            ax.plot(xyz[:, 0], xyz[:, 2], xyz[:, 1], color=color, alpha=.35, linewidth=.6)
            planar.plot(xyz[:, 0], xyz[:, 2], color=color, alpha=.35, linewidth=.6)
        for side, model in models.items():
            p = model.pose(initial[side+'_q_rad'])[0]
            ax.scatter([p[0]], [p[2]], [p[1]], s=70, marker='*', color=palette[side], label=f'{side}: {counts[side]}/70')
            planar.scatter([p[0]], [p[2]], s=100, marker='*', color=palette[side])
        ax.set_xlabel('World X [m]'); ax.set_ylabel('World Z [m]'); ax.set_zlabel('World Y [m]')
        ax.set_title('Shared starts and accepted TCP paths'); ax.legend()
        planar.set_xlabel('World X [m]'); planar.set_ylabel('World Z [m]'); planar.set_title('Top view')
        planar.axis('equal'); planar.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/'overview.png', dpi=160); plt.close(fig)
    def local_link(path, title):
        return f'<a href="{html.escape(os.path.relpath(ROOT/path, out))}">{html.escape(title)}</a>'
    cells = []
    for row in rows:
        links = local_link(row['trajectory'], 'JSON') if (ROOT/row['trajectory']).exists() else ''
        if row.get('csv'): links += ' '+local_link(row['csv']['file'], 'CSV')
        cells.append('<tr>'+''.join(f'<td>{html.escape(str(row[k]))}</td>' for k in ('id','pitch_deg','roll_deg','side'))
            +f'<td>{"合格" if row["passed"] else "探索継続" if row["status"] == "pending" else "未達"}</td>'
            +f'<td>{row.get("motion_duration_s", 0):.2f}</td><td>{links}</td></tr>')
    document = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>70形状の共通初期姿勢と接近IK軌道</title>
<style>body{{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 16px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:7px;border:1px solid #ccc}}img{{width:100%}}</style>
<h1>70形状の共通初期姿勢と接近IK軌道</h1><p>右腕 {counts['right']}/70、左腕 {counts['left']}/70。全条件合格：{passed}。</p>
<p>左腕は既存Aの最後の通過点、右腕は足の矢状面に対する反対側を探索中心とする。同じ初期姿勢から片腕ずつ接近し、もう一方は共通初期姿勢に保持する。人体・胴体・頭部・台座の配置は固定。</p>
<p>目標は開口帯（開口縁から材料長0〜15mm）の表面に対し、指先中央面が2mm手前に来る姿勢。TCPから指先までの10mmを補正。開口への接近IKを検査し、閉鎖把持や摩擦保持、動力学再生は含めない。</p>
<p><a href="{'verified-initial.yaml' if passed else 'initial.yaml'}">共通初期姿勢 YAML</a> / <a href="reference.json">基準通過点・鏡映面</a> / <a href="summary.csv">対応表 CSV</a> / <a href="summary.json">検査結果 JSON</a></p>
<p>{'<a href="ik-trajectories-70-states.zip">140軌道のJSON・CSV一式 ZIP</a>' if passed else ''}</p>
<p>Unityによる共通初期姿勢の復元：{report.get('native_initial', {}).get('passed', '未実施')}。軌道全体の動力学再生は未実施。</p>
<p><a href="native-start/native-initial/result.json">Unity初期姿勢の検査</a></p>
<img src="native-start/native-initial/oblique.png" alt="Unityで復元した共通初期姿勢">
<p>経路CSVは時刻・右腕7関節・左腕7関節・左右開き幅・TCP位置（rad / m / s）。開き幅を変える場合は、初期位置での開閉区間を含む。位置IK誤差0.1mm以下、関節余裕5°以上、既存の実形状と登録済み衝突形状で検査。</p>
<img src="overview.png" alt="共通初期姿勢と接近軌道"><table><tr><th>条件</th><th>pitch°</th><th>roll°</th><th>腕</th><th>結果</th><th>接近時間s</th><th>軌道</th></tr>{''.join(cells)}</table></html>'''
    (out/'report.html').write_text(document)
    print('ROUTES', counts, 'all_passed', passed, flush=True)


if __name__ == '__main__':
    main()
