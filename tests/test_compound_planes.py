"""Independent numerical checks for the ray-plane diagnostic oracle."""
import sys
import base64
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
pytest.importorskip('fcl')
from probe_compound_planes import reconstruct,solid_vertex_distance,witness_targets,support_seeds
from diagnose_compound_geometry import conditioned_part,vector
from decode_compound_cooked import decode_convex
from report_compound_cooked import compare_cooked


def cube():
    return np.array([[x,y,z] for x in [-1.,1.] for y in [-1.,1.] for z in [-1.,1.]])


def test_closed_solid_distance_is_zero_inside_and_euclidean_outside():
    np.testing.assert_allclose(solid_vertex_distance(np.array([[0,0,0],[1,1,1],[2,2,1],[3,0,0.]]),cube()),[0,0,np.sqrt(2),2])


def test_native_support_seeds_expose_every_vertex_after_query_stretch():
    vertices=cube()*[100,.02,.0001];stretch=1/np.ptp(vertices,axis=0);seeds=support_seeds(vertices,stretch)
    assert len(seeds)==8
    for seed in seeds:
        anchor=np.array([seed['anchor'][a] for a in 'xyz']);direction=np.array([seed['direction'][a] for a in 'xyz'])*stretch;values=(vertices*stretch)@direction
        index=np.argmax(values)
        np.testing.assert_array_equal(vertices[index],anchor)
        assert np.sort(values)[-1]>np.sort(values)[-2]


def test_face_witnesses_have_per_vertex_groups_and_physical_distance_bound():
    vertices=cube();scale=np.array([.04,.02,1e-5]);result=witness_targets(vertices,scale,True)
    assert set(result['targetGroups'])==set(range(8))
    assert len(result['targets'])>8
    for target,reference,group in zip(result['targets'],result['targetReferences'],result['targetGroups']):
        target=np.array([target[a] for a in 'xyz']);reference=np.array([reference[a] for a in 'xyz'])
        np.testing.assert_array_equal(reference,vertices[group])
        assert np.linalg.norm((target-reference)*scale)<=2.500001e-7


def test_plane_recovery_measures_removed_cap_and_validates_new_rays():
    expected=cube();cooked=cube();cooked[cooked[:,0]>0,0]=.8;scale=np.array([.04,.02,.01]);planes=ConvexHull(cooked).equations
    rays=[]
    for normal in np.unique(planes[:,:3],axis=0):
        point=cooked[np.isclose(cooked@normal,(cooked@normal).max())].mean(0)
        rays.append(dict(kind='sphere',point=point.tolist(),normal=normal.tolist(),hit=True))
    for vertex in cooked:
        normal=planes[np.argmax(vertex@planes[:,:3].T+planes[:,3]),:3]
        rays.append(dict(kind='reconstructed_vertex',point=vertex.tolist(),normal=normal.tolist(),hit=True))
    rays.append(dict(kind='independent',origin=[3,0,0],direction=[-1,0,0],distance=2.2,point=[.8,0,0],normal=[1,0,0],hit=True))
    shape=dict(scale=vector(scale),**witness_targets(cooked,scale))
    result=reconstruct(dict(index=7,variant='synthetic',vertices=expected.tolist(),rays=rays),shape)
    assert result['stable']
    assert result['input_outside_reconstructed_m']==pytest.approx(.008)
    assert result['reconstructed_outside_input_m']==0
    assert result['plane_count']==6
    rays[-1]['distance']=2.3
    assert not reconstruct(dict(index=7,variant='synthetic',vertices=expected.tolist(),rays=rays),shape)['stable']


@pytest.mark.parametrize('variant',['native','pcabaked','uniform100_default','uniform100_none','uniform100_nobake','uniform100_reverse'])
def test_cook_experiments_preserve_physical_vertices(variant):
    from scipy.spatial.transform import Rotation
    points=cube()*[.04,.02,.001]+[.08,.03,.1]
    shape=conditioned_part(dict(link='test',vertices=[vector(p) for p in points]),3,variant)
    vertices=np.array([[p[a] for a in 'xyz'] for p in shape['vertices']]);scale=np.array([shape['scale'][a] for a in 'xyz']);position=np.array([shape['position'][a] for a in 'xyz']);rotation=Rotation.from_quat([shape['rotation'][a] for a in 'xyzw']).as_matrix()
    np.testing.assert_allclose((vertices*scale)@rotation.T+position,points.astype(np.float32),atol=2e-8,rtol=0)


def test_actual_unity_binary_distinguishes_vertex_removal_from_plane_error():
    fixture=json.loads((Path(__file__).parent/'fixtures/physx_cooked_convex.json').read_text())
    for row in fixture['parts']:
        blob=base64.b64decode(row['binary_base64']);decoded=decode_convex(blob)
        result=compare_cooked(dict(decoded,cooking_options=row['cooking_options']),row['shape'])
        if row['shape']['index']==-1:
            assert decoded['vertex_count']==8
            assert result['cooked_vertex_hull_hausdorff_m']==pytest.approx(0,abs=1e-12)
            assert result['serialized_plane_hull_hausdorff_m']==pytest.approx(0,abs=1e-12)
        elif row['shape']['index']==1528:
            assert decoded['vertex_count']==10
            assert result['input_vertices']==16
            assert result['cooked_vertex_hull_hausdorff_m']==pytest.approx(.000285622,abs=1e-9)
            assert result['serialized_plane_hull_hausdorff_m']==pytest.approx(.000378200,abs=1e-9)
        else:
            # The stored face equations need not enclose the stored vertices,
            # and can even leave an open recession direction on a thin hull.
            assert row['shape']['index']==214
            assert decoded['vertex_count']==5
            assert not result['serialized_plane_hull_closed']
            assert result['serialized_plane_hull_hausdorff_m'] is None
        with pytest.raises(ValueError):decode_convex(b'bad header'+blob[10:])
        with pytest.raises(ValueError):decode_convex(blob[:60])
