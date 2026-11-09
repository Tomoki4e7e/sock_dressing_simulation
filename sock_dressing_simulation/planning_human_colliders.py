"""Conservative convex envelopes of active Unity human colliders.

A sphere mesh is scaled by its minimum facet distance so that EVERY facet
supports the exact sphere. Convex hulls of two such spheres contain capsules.
"""
import copy,json,hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

@lru_cache(None)
def sphere_envelope():
 import trimesh
 vertices=np.asarray(trimesh.creation.icosphere(subdivisions=2,radius=1.).vertices)
 hull=ConvexHull(vertices);distance=np.min(-hull.equations[:,3]/np.linalg.norm(hull.equations[:,:3],axis=1))
 return vertices*(1.+1e-9)/distance

def envelope(row):
 r=copy.deepcopy(row);r['name']='human_proxy_skin/'+r['name']
 if r['kind'] in ('MeshCollider','BoxCollider'):return r
 if r['kind'] not in ('CapsuleCollider','SphereCollider'):raise ValueError('Unsupported human collider '+r['kind'])
 scale=np.asarray(r['scale']);rotation=Rotation.from_quat(r['rotation']).as_matrix();center=rotation@(np.asarray(r['center'])*scale)+r['position']
 if r['kind']=='SphereCollider':radius=r['radius']*np.abs(scale).max();centers=np.array([center])
 else:
  axis=int(r['direction']);radius=r['radius']*max(abs(scale[j]) for j in range(3) if j!=axis);half=max(0.,r['height']*abs(scale[axis])/2-radius);direction=rotation[:,axis];centers=np.array([center-half*direction,center+half*direction])
 points=np.concatenate([sphere_envelope()*radius+c for c in centers]);hull=ConvexHull(points)
 return dict(name=r['name'],kind='MeshCollider',convex=True,vertices_world=points.tolist(),triangles=hull.simplices.ravel().tolist(),position=[0.,0.,0.],rotation=[0.,0.,0.,1.],scale=[1.,1.,1.],link='',source_kind=r['kind'],conservative_envelope=True)

@lru_cache(128)
def _read(path,digest):
 raw=Path(path).read_bytes()
 if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Human collider snapshot hash mismatch')
 return json.loads(raw)

def augmented_scene(scene,root,manifest_path,numbers):
 result=copy.deepcopy(scene);path=Path(manifest_path)
 if not path.is_absolute():path=Path(root)/path
 manifest=json.loads(path.read_text())
 seen=set()
 for n in sorted(set(numbers)):
  entry=manifest['scenes'][str(n)];data=_read(str(Path(root)/entry['path']),entry['sha256'])
  if data['human_skin_error_m']>5e-6:raise ValueError('Unregistered human collision shapes')
  for row in data['colliders']:
   key=json.dumps(row,sort_keys=True)
   if key in seen:continue
   seen.add(key)
   shape=envelope(row);shape['name']+='@'+str(n);result['obstacles'].append(shape)
 return result
