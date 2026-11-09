"""Audit the opt-in frozen full-skin contact model against saved planning geometry."""
import gzip
import json
import numpy as np


def install(env, source_scene, delta, out):
    env.sock_cloth._send_data('InstallSockExactHumanContact', env.robot.id)
    env._env.step(2, simulate=False)
    report = env.sock_cloth.data.get('sock_exact_human_contact')
    if report is None:
        # Reports are transient; explicitly request a single response.
        env.sock_cloth._send_data('GetSockExactHumanContact')
        env._env.step(simulate=False)
        report = env.sock_cloth.data['sock_exact_human_contact']
    points = np.asarray(report['vertices_world'])
    reference = np.asarray(source_scene['human_skin']['vertices']) + delta
    error = float(np.linalg.norm(points-reference, axis=1).max())
    triangles_equal = np.array_equal(report['triangles'], source_scene['human_skin']['triangles'])
    passed = bool(report['active'] and report['triangles_match'] and triangles_equal and error < 2e-5
                  and report['maximum_skin_vertex_error_m'] < 2e-5
                  and report['robot_skin_pairs_ignored'] == 0
                  and report['proxy_robot_pairs_still_enabled'] == 0
                  and report['proxy_robot_pair_count'] > 0)
    report.update(planning_skin_maximum_error_m=error, planning_triangles_equal=triangles_equal, passed=passed)
    with gzip.open(out / 'exact-human-contact.json.gz', 'wt') as stream:
        json.dump(report, stream)
    if not passed:raise RuntimeError('Native full-human skin does not match the planning model')
    return {key: value for key, value in report.items() if key not in ('vertices_world', 'triangles')}
