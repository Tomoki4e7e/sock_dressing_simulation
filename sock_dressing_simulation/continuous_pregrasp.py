"""Conservative swept-surface verification of articulated pregrasp polylines.

Midpoint set distances minus URDF-chain Hausdorff displacement bounds certify
whole intervals. Unresolved intervals are never counted as collision-free.
Local overlap is bounded by testing an inward-offset convex claw against only
the eligible surface triangles; every other surface keeps the old clearance.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import fcl
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree
from .arm_reaching import ARM, triangle_object
from .fixed_torso_reaching import LEFT_ARM
from .grasp_geometry import convex_geometry
from .contact_pregrasp import ContactGoal, material_band
from .controlled_arm_reaching import ControlledArmKinematics

SIDES=('left','right')


@dataclass(frozen=True)
class SweptPolicy:
    endpoint_radius_m:float=.005
    local_material_depth_m:float=.030
    skin_rim_radius_m:float=.030
    permitted_intrusion_m:float=.0005
    numerical_allowance_m:float=1e-7
    maximum_depth:int=24
    condition_timeout_s:float=600.


def state_vector(left,right,left_gap,right_gap):
    return np.r_[left,right,left_gap,right_gap].astype(float)


def eroded_vertices(vertices,depth):
    h=ConvexHull(vertices).equations.copy();h[:,3]+=depth
    fit=linprog([0.,0.,0.,-1.],A_ub=np.c_[h[:,:3],np.ones(len(h))],b_ub=-h[:,3],
                bounds=[(None,None)]*3+[(0.,None)],method='highs')
    if not fit.success or fit.x[3]<1e-9:raise ValueError('Empty inward-offset claw: cannot grant this intrusion tolerance')
    return HalfspaceIntersection(h,fit.x[:3]).intersections


def capsule_triangles(vertices,faces,rim,radius):
    """Only triangles wholly inside one convex rim-edge capsule are eligible."""
    triangles=np.asarray(vertices)[faces];mask=np.zeros(len(faces),bool)
    if not len(triangles):return mask
    lo=rim.reshape(-1,3).min(0)-radius;hi=rim.reshape(-1,3).max(0)+radius
    ids=np.flatnonzero((triangles.min(1)<=hi).all(1)&(triangles.max(1)>=lo).all(1))
    if not len(ids):return mask
    t=triangles[ids]
    for a,b in rim:
        v=b-a;u=np.clip(((t-a)@v)/max(v@v,1e-20),0.,1.)
        mask[ids]|=(np.linalg.norm(t-(a+u[...,None]*v),axis=2).max(1)<=radius)
    return mask


def surface_object(vertices,faces):
    if not len(faces):return None
    ids,inv=np.unique(np.asarray(faces).reshape(-1),return_inverse=True)
    return triangle_object(np.asarray(vertices)[ids],inv.reshape(-1,3))


class MotionBounds:
    """Radians/metres -> maximum point motion, with shared ancestors removed."""
    def __init__(self,k,groups):
        self.k=k;self.chains={};self.coefficients={};self.scale=float(np.linalg.svd(k.A,compute_uv=False).max())
        masters={n:(i,1.)for i,n in enumerate(LEFT_ARM+ARM)}
        masters.update({'left_gripper/finger_joint':(14,1.),'right_gripper/finger_joint':(15,1.)})
        def master(name):
            if name in masters:return masters[name]
            mimic=k.joints[name].find('mimic')
            if mimic is None:return None
            value=master(mimic.get('joint'))
            return None if value is None else(value[0],value[1]*float(mimic.get('multiplier','1')))
        def chain(link):
            c=[]
            while link in k.parents:
                j=k.parents[link];c.append(j.get('name'));link=j.find('parent').get('link')
            return c[::-1]
        for link,rows in groups.items():
            self.chains[link]=chain(link)
            radius=max(float(np.linalg.norm(r['vertices'],axis=1).max())for r in rows)
            entries={};downstream=radius
            for name in reversed(self.chains[link]):
                j=k.joints[name];m=master(name);kind=j.get('type')
                if m is not None:
                    entries[name]=(m[0],abs(m[1])*float(np.linalg.norm(k.axes[name]))*(1. if kind=='prismatic' else downstream)*self.scale)
                downstream+=float(np.linalg.norm(k.origins[name][:3,3]))
                if kind=='prismatic':
                    lim=j.find('limit');downstream+=max(abs(float(lim.get('lower','0'))),abs(float(lim.get('upper','0'))))*float(np.linalg.norm(k.axes[name]))
            self.coefficients[link]=entries

    def pair(self,a,b):
        common=set(self.chains.get(a,()))&set(self.chains.get(b,()))
        v=np.zeros(16)
        for link in (a,b):
            for name,(i,r)in self.coefficients.get(link,{}).items():
                if name not in common:v[i]+=r
        return v

    def tcp(self,side):
        # A point at the grasp centre is contained in this bounding sphere;
        # enlarge the base-link coefficients if its mesh is smaller.
        link=side+'_gripper/gripper_base';v=self.pair(link,'world')
        radius=max(float(np.linalg.norm(r['vertices'],axis=1).max())for r in self._groups[link])if hasattr(self,'_groups')else 0.
        extra=max(0.,float(np.linalg.norm(self.k.hand.grasp_center_urdf))-radius)*self.scale
        for name,(i,_)in self.coefficients[link].items():
            if self.k.joints[name].get('type')!='prismatic':v[i]+=extra
        return v


class SweptScene:
    def __init__(self,k,base,state,route,delta,policy=SweptPolicy()):
        self.k,self.base,self.state,self.route,self.delta,self.policy=k,base,state,route,np.asarray(delta),policy
        self.goal=ContactGoal(state,k.hand);self.bounds=MotionBounds(k,base.groups);self.bounds._groups=base.groups
        self.models={s:ControlledArmKinematics(k,s,route['initial_right_q_rad'],.04)for s in SIDES}
        self.targets={s:np.array(route['terminal'][s]['tcp_world_m'])for s in SIDES}
        self.tcp_bounds={s:self.bounds.tcp(s)for s in SIDES}
        self.pairs=list(base.pairs)+[(s+'_arm/link_2','torso/link_3',base.pairs[base.shoulder_pairs[s]][2])for s in SIDES]
        self.motion=np.array([self.bounds.pair(a,b)for a,b,_ in self.pairs])
        self.split={};self.eroded={};self._make_local_surfaces()
        self.queries=0;self.updates=0;self.request=fcl.DistanceRequest()

    def _make_local_surfaces(self):
        p=self.policy;state=self.state;faces=np.asarray(state['triangles']).reshape(-1,3);points=np.asarray(state['particles_world'])
        rest=np.asarray(state['source']['rest_positions']);depth=rest[:,2]-rest[:,2].min()
        local_v,local_f=material_band(points,faces,depth,0.,p.local_material_depth_m)
        remote_v,remote_f=material_band(points,faces,depth,p.local_material_depth_m,float(depth.max()+1.))
        self.split['frozen_cloth']=(surface_object(local_v,local_f),surface_object(remote_v,remote_f))
        sources={'human_full_skin':(np.asarray(self.base.scene['human_skin']['vertices']),np.asarray(self.base.scene['human_skin']['triangles']).reshape(-1,3))}if hasattr(self.base,'scene')else{}
        # Source scene vertices are supplied explicitly by the caller below.
        sources.update(getattr(self.base,'swept_skin_sources',{}))
        skin=np.asarray(state['source']['skin_surface']['vertices']);tree=cKDTree(skin)
        for name,(vertices,triangles)in sources.items():
            vertices=np.asarray(vertices);triangles=np.asarray(triangles).reshape(-1,3)
            mask=capsule_triangles(vertices,triangles,self.goal.rim,p.skin_rim_radius_m)
            if name=='human_full_skin':
                # Full-body triangles must also belong to the actual target leg.
                matched=tree.query(vertices)[0]<=2e-5;mask&=matched[triangles].all(1)
            self.split[name]=(surface_object(vertices,triangles[mask]),surface_object(vertices,triangles[~mask]))
        for link,rows in self.base.groups.items():
            if not link.endswith(('/left_finger_link','/right_finger_link')):continue
            erosion=p.permitted_intrusion_m-self.base.padding.get(link,0.)
            if erosion<=p.numerical_allowance_m:continue
            objects=[]
            for row in rows:
                geometry,_,_=convex_geometry(eroded_vertices(row['vertices'],erosion))
                objects.append(fcl.CollisionObject(geometry))
            self.eroded[link]=objects

    def update(self,x):
        self.base.park_left(x[:7],float(x[14]));self.base.update(x[7:14],float(x[15]),self.delta);self.updates+=1
        self.positions={s:self.models[s].pose(x[:7]if s=='left'else x[7:14])[0]for s in SIDES}
        for name,objects in self.split.items():
            for obj in objects:
                if obj is not None:obj.setTransform(fcl.Transform(np.eye(3),self.delta))
        for link,objects in self.eroded.items():
            for obj in objects:obj.setTransform(self.base.group_transforms[link])

    def window(self,side,dx):
        return np.linalg.norm(self.positions[side]-self.targets[side])+self.tcp_bounds[side]@abs(dx)/2<=self.policy.endpoint_radius_m

    def _distance(self,objects,obstacle):
        if obstacle is None:return float('inf')
        best=float('inf')
        for obj in objects:
            d=float(fcl.distance(obj,obstacle,self.request,fcl.DistanceResult()));self.queries+=1
            if not np.isfinite(d):raise ValueError('Nonfinite continuous distance')
            best=min(best,d)
            if d<=0:break
        return best

    def _group_distance(self,a,b,cap):
        rows_a=self.base.groups[a];lo_a,hi_a=self.base.group_bounds[a]
        if b in self.base.groups:
            rows_b=self.base.groups[b];lo_b,hi_b=self.base.group_bounds[b]
        else:
            rows_b=[None];lo_b,hi_b=(v[None,:]for v in self.base.obstacle_bounds[b])
        lower=np.linalg.norm(np.maximum(np.maximum(lo_a[:,None]-hi_b[None,:],lo_b[None,:]-hi_a[:,None]),0),axis=2)
        ids=np.flatnonzero(lower.ravel()<=cap);ids=ids[np.argsort(lower.ravel()[ids])];best=cap
        for index in ids:
            if lower.ravel()[index]>max(best,0.):break
            i,j=divmod(int(index),len(rows_b));oa=self.base.query_object(rows_a[i]);ob=self.base.query_object(rows_b[j])if rows_b[j]is not None else self.base.obstacle_objects[b]
            value=float(fcl.distance(oa,ob,self.request,fcl.DistanceResult()));self.queries+=1
            if not np.isfinite(value):raise ValueError('Nonfinite continuous distance')
            best=min(best,value)
            if best<=0:break
        return best

    def slack(self,index,x,dx,motion_bound=0.):
        a,b,margin=self.pairs[index]
        if index>=len(self.base.pairs):
            side='left'if a.startswith('left_')else'right'
            return self.base.shoulder_distance(x[7:14],side)-margin,'shoulder_guard'
        side='left'if a.startswith('left_')else'right'
        rows=self.base.groups[a];objects=[self.base.query_object(r)for r in rows]
        if a in self.eroded and b in self.split and self.window(side,dx):
            local,remote=self.split[b]
            if local is not None:
                # No surface may enter the inward-offset claw core. This
                # bounds surface intrusion depth into the full claw by 0.5 mm.
                value=min(self._distance(self.eroded[a],local),self._distance(objects,remote)-margin)
                return value,'localized_claw_core'
        best=self._group_distance(a,b,margin+motion_bound+2*self.policy.numerical_allowance_m)
        return best-margin,'strict'

    def broad_slacks(self):
        lo={n:v[0].min(0)for n,v in self.base.group_bounds.items()};hi={n:v[1].max(0)for n,v in self.base.group_bounds.items()}
        for n,(a,b)in self.base.obstacle_bounds.items():lo[n]=a;hi[n]=b
        a=np.array([lo[n]for n,_,_ in self.pairs]);b=np.array([hi[n]for n,_,_ in self.pairs]);c=np.array([lo[n]for _,n,_ in self.pairs]);d=np.array([hi[n]for _,n,_ in self.pairs])
        # Strict whole-shape AABBs are a conservative early certificate even
        # when a localized allowance could also be used.
        return np.linalg.norm(np.maximum(np.maximum(a-d,c-b),0),axis=1)-np.array([m for _,_,m in self.pairs])


def certify_interval(scene,a,b,maximum_depth=None,deadline=None):
    p=scene.policy;maximum_depth=p.maximum_depth if maximum_depth is None else maximum_depth
    stack=[(0.,1.,np.arange(len(scene.pairs)),0)];checked=0;certified=0;deepest=0;allowance_pairs=set()
    while stack:
        lo,hi,ids,depth=stack.pop();deepest=max(deepest,depth)
        if deadline is not None and time.monotonic()>deadline:
            return dict(status='unresolved',reason='time_budget',interval=[lo,hi],checks=checked,certified_pair_intervals=certified,maximum_depth=deepest)
        x=a+(b-a)*((lo+hi)/2);dx=(b-a)*(hi-lo);scene.update(x);checked+=1
        displacement=scene.motion[ids]@abs(dx)/2
        broad=scene.broad_slacks()[ids]-displacement-p.numerical_allowance_m
        remaining=[]
        for slot,index in enumerate(ids):
            if broad[slot]>0:certified+=1;continue
            slack,kind=scene.slack(int(index),x,dx,float(displacement[slot]))
            if kind=='localized_claw_core':allowance_pairs.add(int(index))
            if slack>displacement[slot]+p.numerical_allowance_m:certified+=1;continue
            if slack < -p.numerical_allowance_m:
                point_slack,point_kind=scene.slack(int(index),x,np.zeros(16))
                if point_slack < -p.numerical_allowance_m:
                    return dict(status='failed',reason='midpoint_clearance_violation',pair=list(scene.pairs[index]),mode=point_kind,
                                slack_m=point_slack,parameter=(lo+hi)/2,configuration=x.tolist(),checks=checked)
            if depth>=maximum_depth:
                return dict(status='unresolved',reason='subdivision_limit',pair=list(scene.pairs[index]),
                            slack_m=slack,motion_bound_m=float(displacement[slot]),interval=[lo,hi],checks=checked)
            remaining.append(index)
        if remaining:
            mid=(lo+hi)/2;remaining=np.array(remaining,dtype=int)
            stack.append((mid,hi,remaining,depth+1));stack.append((lo,mid,remaining,depth+1))
    return dict(status='certified',checks=checked,certified_pair_intervals=certified,maximum_depth=deepest,
                local_allowance_pairs=[list(scene.pairs[i])for i in sorted(allowance_pairs)])


def trajectory_intervals(route,initial):
    """Exact joint polylines; quintic timing is monotone on each interval."""
    x=state_vector(initial['left_q_rad'],initial['right_q_rad'],initial['left_gap_m'],initial['right_gap_m'])
    values=[]
    def arm(segment,stage):
        nonlocal x
        side=segment['side'];s=slice(0,7)if side=='left'else slice(7,14)
        knots=np.asarray(segment['knots_q_rad']);np.testing.assert_allclose(x[s],knots[0],atol=1e-10,rtol=0)
        for target,duration in zip(knots[1:],segment['segment_durations_s']):
            nxt=x.copy();nxt[s]=target;values.append(dict(stage=stage,side=side,start=x.copy(),end=nxt.copy(),duration_s=duration));x=nxt
    for segment in route.get('initial_transition_segments',[]):arm(segment,'initial_transition')
    for side in SIDES:
        i=14 if side=='left'else 15;gap=route['terminal'][side]['opening_gap_m'];nxt=x.copy();nxt[i]=gap
        values.append(dict(stage='prepare_open_width',side=side,start=x.copy(),end=nxt.copy(),duration_s=max(.2,1.875*abs(gap-x[i])/.01)));x=nxt
    for segment in route['segments']:arm(segment,'approach')
    return values
