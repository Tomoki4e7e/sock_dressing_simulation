from types import SimpleNamespace
import numpy as np
import pytest
from sock_dressing_simulation.exact_human_contact import install


@pytest.mark.parametrize('defect',[None,'proxy','skin_pair','geometry','triangles'])
def test_native_skin_audit_rejects_geometry_and_collision_policy_mismatch(tmp_path,defect):
    vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    delta=np.array([.1,.2,.3])
    report=dict(active=True,vertices_world=(vertices+delta).tolist(),triangles=[0,1,2],
                triangles_match=True,maximum_skin_vertex_error_m=0.,robot_skin_pairs_ignored=0,
                proxy_robot_pairs_still_enabled=0,proxy_robot_pair_count=10)
    if defect=='proxy':report['proxy_robot_pairs_still_enabled']=1
    if defect=='skin_pair':report['robot_skin_pairs_ignored']=1
    if defect=='geometry':report['vertices_world'][0][0]+=.001
    if defect=='triangles':report['triangles']=[0,2,1]
    env=SimpleNamespace(robot=SimpleNamespace(id=1),
        sock_cloth=SimpleNamespace(_send_data=lambda *a:None,data={'sock_exact_human_contact':report}),
        _env=SimpleNamespace(step=lambda *a,**k:None))
    scene=dict(human_skin=dict(vertices=vertices.tolist(),triangles=[0,1,2]))
    if defect:
        with pytest.raises(RuntimeError,match='does not match'):install(env,scene,delta,tmp_path)
    else:assert install(env,scene,delta,tmp_path)['passed']
