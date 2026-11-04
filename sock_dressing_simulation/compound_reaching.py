"""Compound convex distance constraints for rigid scene translation experiments.

The normal collision policy is retained. Only explicitly supplied link geometry
is replaced. Obstacles move by a rigid transform without rebuilding their BVH.
"""
from pathlib import Path
import json
import numpy as np
import fcl
from .arm_reaching import CollisionScene,shape_vertices
from .grasp_geometry import convex_geometry,HANDEDNESS


def distance_feature(a,b):
    # FCL dispatches mesh/primitive queries with the BVH first internally.
    # Query in that order explicitly so witness points and normals have a
    # defined owner, then restore the caller's object order.
    if a.getObjectType()!=fcl.OBJECT_TYPE.OT_BVH and b.getObjectType()==fcl.OBJECT_TYPE.OT_BVH:
        d,pb,pa,n=distance_feature(b,a)
        return d,pa,pb,-n
    result=fcl.DistanceResult()
    raw=float(fcl.distance(a,b,fcl.DistanceRequest(enable_nearest_points=True),result))
    if not np.isfinite(raw):raise ValueError('Nonfinite FCL distance')
    if raw>0:
        pa,pb=np.array(result.nearest_points)
        return raw,pa,pb,(pa-pb)/raw
    contact=fcl.CollisionResult()
    fcl.collide(a,b,fcl.CollisionRequest(num_max_contacts=32,enable_contact=True),contact)
    contacts=[c for c in contact.contacts if np.isfinite(c.penetration_depth)]
    if not contacts:return -1e-7,np.zeros(3),np.zeros(3),np.zeros(3)
    c=max(contacts,key=lambda c:c.penetration_depth)
    # FCL normals point from object A toward object B.
    return -max(float(c.penetration_depth),1e-7),np.array(c.pos),np.array(c.pos),-np.array(c.normal)


def signed_distance(a,b):
    return distance_feature(a,b)[0]


def aabb_distance(a,b):
    return float(np.linalg.norm(np.maximum(np.maximum(a[0]-b[1],b[0]-a[1]),0)))


class CompoundScene(CollisionScene):
    def __init__(self,k,scene,faces,manifest,root):
        super().__init__(k,scene,faces)
        manifest=json.loads(Path(manifest).read_text())
        if not manifest['passed']:raise ValueError('Unvalidated decomposition')
        replacements={r['link']:r for r in manifest['parts']};self.padding={}
        self.shapes=[r for r in self.shapes if r['link'] not in replacements]
        for link,r in replacements.items():
            data=json.loads((Path(root)/r['parts_file']).read_text())
            self.padding[link]=max(.0001,r['maximum_original_surface_outside_union_m'])
            for i,part in enumerate(data['parts']):
                geometry,vertices,_=convex_geometry(np.array(part['vertices'])@HANDEDNESS)
                self.shapes.append(dict(link=link,part=i,obj=fcl.CollisionObject(geometry),vertices=vertices,moving=True))
        self.obstacle_points={
            'human_full_skin':np.array(scene['human_skin']['vertices']),
            'frozen_cloth':np.array(scene['particles_world'])}
        self.obstacle_points.update({r['name']:shape_vertices(r) for r in scene['obstacles']})
        self.base_obstacle_bounds={name:(p.min(0),p.max(0)) for name,p in self.obstacle_points.items()}
        self.groups={}
        for row in self.shapes:self.groups.setdefault(row['link'],[]).append(row)
        names=list(self.groups);self.pairs=[]
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if not (self.groups[a][0]['moving'] or self.groups[b][0]['moving']) or self.excluded(a,b):continue
                margin=0. if a.startswith('right_gripper/') and b.startswith('right_gripper/') else .005
                self.pairs.append((a,b,margin+self.padding.get(a,0)+self.padding.get(b,0)))
        for a in names:
            hand=a.startswith('right_gripper/')
            for b,_ in self.obstacles:
                if 'Ground' in b and not self.groups[a][0]['moving']:continue
                margin=0. if b=='frozen_cloth' else (.001 if hand and 'skin' in b else .005)
                self.pairs.append((a,b,margin+self.padding.get(a,0)))
        floors=[r for r in scene['obstacles'] if 'Ground' in r['name']]
        if len(floors)!=1 or floors[0]['kind']!='BoxCollider':raise ValueError('Expected one flat box floor')
        floor_points=shape_vertices(floors[0]);self.floor_top=float(floor_points[:,1].max())
        self.floor_minima={name:float(points[:,1].min()) for name,points in self.obstacle_points.items() if 'Ground' not in name}
        self.obstacle_objects=dict(self.obstacles)
        self.links=[r['link'] for r in self.shapes]
        from .arm_reaching import ARM
        self.dependencies={}
        for link in self.groups:
            chain=[];node=link
            while node in k.parents:
                j=k.parents[node]
                if j.get('name') in ARM:chain.append(ARM.index(j.get('name')))
                node=j.find('parent').get('link')
            self.dependencies[link]=chain


    def update(self,q,gap,delta):
        frames=self.k.fk(q,gap,only=self.links)
        for row in self.shapes:
            T=frames[row['link']];R=self.k.A@T[:3,:3]@HANDEDNESS;t=self.k.A@T[:3,3]+self.k.b
            row['obj'].setTransform(fcl.Transform(R,t));points=row['vertices']@R.T+t;row['bounds']=(points.min(0),points.max(0))
        self.obstacle_bounds={}
        for name,obj in self.obstacles:
            d=np.zeros(3) if 'Ground' in name else np.asarray(delta)
            obj.setTransform(fcl.Transform(np.eye(3),d));lo,hi=self.base_obstacle_bounds[name];self.obstacle_bounds[name]=(lo+d,hi+d)

    def slacks(self,q,gap,delta,allow_cloth_overlap=False,cap=.02,details=False,with_jacobian=False):
        self.update(q,gap,delta);values=[];issues=[];jacobian=[]
        if with_jacobian:
            from scipy.spatial.transform import Rotation
            T=self.k.arm_parent.copy();axes=[];origins=[]
            for angle,O,axis in zip(q,self.k.arm_origins,self.k.arm_axes):
                T=T@O;origins.append(self.k.A@T[:3,3]+self.k.b);axes.append(-self.k.A@T[:3,:3]@axis)
                T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(axis*angle).as_matrix()
            axes=np.array(axes);origins=np.array(origins)
            def point_jac(name,point):
                J=np.zeros((3,10))
                if name in self.dependencies:
                    ids=self.dependencies[name];J[:,:7][:,ids]=np.cross(axes[ids],point-origins[ids]).T
                elif 'Ground' not in name:J[:,7:]=np.eye(3)
                return J
        for a,b,margin in self.pairs:
            if allow_cloth_overlap and b=='frozen_cloth' and a.startswith('right_gripper/'):
                values.append(cap);jacobian.append(np.zeros(10));continue
            best=cap+margin;feature=None
            rows_b=self.groups[b] if b in self.groups else [dict(obj=self.obstacle_objects[b],bounds=self.obstacle_bounds[b])]
            rows_a=self.groups[a]
            lo_a=np.array([r['bounds'][0] for r in rows_a]);hi_a=np.array([r['bounds'][1] for r in rows_a])
            lo_b=np.array([r['bounds'][0] for r in rows_b]);hi_b=np.array([r['bounds'][1] for r in rows_b])
            lower=np.linalg.norm(np.maximum(np.maximum(lo_a[:,None]-hi_b[None,:],lo_b[None,:]-hi_a[:,None]),0),axis=2)
            ids=np.flatnonzero(lower.ravel()<=best)
            ids=ids[np.argsort(lower.ravel()[ids])]
            for idx in ids:
                if lower.ravel()[idx]>max(best,0):break
                i,j=divmod(int(idx),len(rows_b))
                f=distance_feature(rows_a[i]['obj'],rows_b[j]['obj'])
                if f[0]<best:best=f[0];feature=f
            slack=best-margin;values.append(slack)
            if with_jacobian:
                jacobian.append(np.zeros(10) if feature is None else feature[3]@(point_jac(a,feature[1])-point_jac(b,feature[2])))
            if details and slack < -1e-9:issues.append(dict(pair=[a,b],signed_distance_m=best,required_clearance_m=margin,slack_m=slack))
        for name,minimum_y in self.floor_minima.items():
            raw=minimum_y+float(delta[1])-self.floor_top;slack=min(cap,raw);values.append(slack)
            if with_jacobian:
                row=np.zeros(10)
                if raw<cap:row[8]=1.
                jacobian.append(row)
            if details and raw< -1e-9:issues.append(dict(pair=[name,'RCareWorld/Ground'],signed_distance_m=raw,required_clearance_m=0.,slack_m=raw))
        values=np.array(values)
        if details:return dict(passed=bool(np.all(values>=-1e-9)),minimum_slack_m=float(values.min()),issues=issues)
        return (values,np.array(jacobian)) if with_jacobian else values

    def is_clear(self,q,gap,delta):
        """Early-exit version of the same clearance policy, for motion planning."""
        if any(y+float(delta[1])-self.floor_top < -1e-9 for y in self.floor_minima.values()):return False
        self.update(q,gap,delta)
        for a,b,margin in self.pairs:
            rows_a=self.groups[a];rows_b=self.groups[b] if b in self.groups else [dict(obj=self.obstacle_objects[b],bounds=self.obstacle_bounds[b])]
            lo_a=np.array([r['bounds'][0] for r in rows_a]);hi_a=np.array([r['bounds'][1] for r in rows_a]);lo_b=np.array([r['bounds'][0] for r in rows_b]);hi_b=np.array([r['bounds'][1] for r in rows_b])
            lower=np.linalg.norm(np.maximum(np.maximum(lo_a[:,None]-hi_b[None,:],lo_b[None,:]-hi_a[:,None]),0),axis=2)
            for flat in np.flatnonzero(lower.ravel()<=margin):
                i,j=divmod(int(flat),len(rows_b));distance=float(fcl.distance(rows_a[i]['obj'],rows_b[j]['obj'],fcl.DistanceRequest(),fcl.DistanceResult()))
                if not np.isfinite(distance):raise ValueError('Nonfinite planning distance')
                if distance<=0 or distance<margin-1e-9:return False
        return True
