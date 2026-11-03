import numpy as np
import pytest

from sock_dressing_simulation.preinsert import generate_preinsert


@pytest.fixture
def source():
    def box(lo, hi):
        vertices = [[x,y,z] for z in (lo[2],hi[2]) for y in (lo[1],hi[1]) for x in (lo[0],hi[0])]
        return dict(vertices=vertices, triangles=[0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5])
    rest = [[.04*np.cos(t),.04*np.sin(t),z] for z in np.linspace(0,.3,25)
            for t in np.arange(32)*2*np.pi/32]
    rest.append([0,0,.3])
    return dict(rest_positions=rest,
                surfaces=[box([-.035,-.025,-.05],[.035,.025,.20]),box([-.035,0,-.04],[.035,.25,.04])],
                toe=[0,0,.18],heel=[0,0,-.03],ankle=[0,0,0],calf=[0,.18,0])


def test_all_stages_are_ordered_with_planar_openings_and_closed_cap(source):
    reports = []
    for stage in 'ABCDE':
        positions, report = generate_preinsert(source, stage)
        reports.append(report)
        cuff = positions[report['opening_particle_indices']]
        center = np.asarray(report['cuff_center_world'])
        normal = np.asarray(report['opening_inward_normal_world'])
        assert np.isfinite(positions).all()
        assert np.max(np.abs((cuff-center) @ normal)) < 1e-7
        assert np.allclose(cuff.mean(axis=0), center)
        assert np.linalg.norm(positions[-1]-positions[-33:-1].mean(axis=0)) < 1e-7
        assert report['material_rest_lengths_modified'] is False
    assert np.all(np.diff([r['insertion_path_m'] for r in reports]) > 0)
    difference = np.asarray(reports[-1]['cuff_center_world'])-reports[-2]['cuff_center_world']
    assert np.isclose(difference[1], .05)


def test_placement_follows_rotated_and_translated_human(source):
    positions, report = generate_preinsert(source, 'E')
    rotation = np.array([[0,0,1],[0,1,0],[-1,0,0]])
    offset = np.array([1.3,.2,-.8])
    for key in ('ankle','heel','toe','calf'):
        source[key] = (np.asarray(source[key]) @ rotation.T + offset).tolist()
    for surface in source['surfaces']:
        surface['vertices'] = (np.asarray(surface['vertices']) @ rotation.T + offset).tolist()
    changed, _ = generate_preinsert(source, 'E')
    assert np.allclose(changed, positions @ rotation.T + offset, atol=1e-7)


@pytest.mark.parametrize('kwargs', [dict(stage='F'),dict(stage='A',clearance_m=-1),dict(stage='E',above_heel_m=float('nan')),dict(stage='E',above_heel_m=.5)])
def test_invalid_or_impossible_placement_is_rejected(source, kwargs):
    with pytest.raises(ValueError): generate_preinsert(source, **kwargs)


def test_audit_detects_outside_to_outside_edges_through_foot(source):
    from sock_dressing_simulation.preinsert import audit_preinsert
    result = audit_preinsert([[-.1,0,.1],[.1,0,.1]], source, [[0,1]], [.2])
    assert result['particles_inside_body'] == 0
    assert result['edges_crossing_body'] == 1
    assert not result['geometry_ok']
    clear = audit_preinsert([[-.1,0,.3],[.1,0,.3]], source, [[0,1]], [.2])
    assert clear['geometry_ok']


def test_native_cap_rounding_and_particle_reordering_keep_closed_toe(source):
    rest = np.asarray(source['rest_positions'])
    rest[-1, 2] -= 2.5e-7  # Native cluster centroid accumulation error.
    order = np.random.default_rng(3).permutation(len(rest))
    source['rest_positions'] = rest[order].tolist()
    points, report = generate_preinsert(source, 'A')
    cap_id = int(np.flatnonzero(order == len(rest)-1)[0])
    toe_ring_ids = np.flatnonzero((np.abs(rest[order, 2]-.3) < 1e-6) & (order != len(rest)-1))
    assert np.allclose(points[cap_id], points[toe_ring_ids].mean(axis=0))
    assert len(report['opening_particle_indices']) == 32


def test_surface_audit_catches_body_through_face_without_inside_cloth_vertices(source):
    from sock_dressing_simulation.preinsert import audit_preinsert_surface
    # A large horizontal cloth triangle cuts through the foot's interior.
    points = [[-1,0,-1],[1,0,-1],[0,0,1]]
    result = audit_preinsert_surface(points, source, [[0,1],[1,2],[2,0]], [[0,1,2]])
    assert result['body_edges_crossing_cloth_faces'] > 0
    assert not result['surface_ok']


def test_preinsert_cannot_silently_run_in_unsupported_player():
    from sock_dressing_simulation.environment import SockDressingEnv
    from sock_dressing_simulation.config import load_config
    config = load_config()
    config['preinsert'] = dict(stage='A')
    with pytest.raises(ValueError, match='native custom Player'):
        SockDressingEnv(config)


def test_preinsert_rejects_the_bent_rest_mesh_before_loading():
    from sock_dressing_simulation.environment import SockDressingEnv
    from sock_dressing_simulation.config import load_config
    config = load_config('config/custom_player.yaml')
    config['preinsert'] = dict(stage='A')
    with pytest.raises(ValueError, match='unbent rest mesh'):
        SockDressingEnv(config)
