"""Fresh robot-frame overlay and independent shoulder-distance guard for reruns."""
import copy
from pathlib import Path
import numpy as np,fcl
from scipy.spatial import ConvexHull
from sock_dressing_simulation.arm_reaching import triangle_object
from sock_dressing_simulation.compound_reaching import CompoundScene,distance_feature

ROBOT_KEYS=('links','robot_colliders','robot_data','externally_stepped')
def overlay_robot(environment,robot):
    scene=copy.deepcopy(environment)
    for key in ROBOT_KEYS:scene[key]=copy.deepcopy(robot[key])
    if 'self_overlap_audit' in robot:scene['self_overlap_audit']=copy.deepcopy(robot['self_overlap_audit'])
    else:scene.pop('self_overlap_audit',None)
    return scene

class AuditedScene(CompoundScene):
    """Retain ordinary checks and additionally cap shoulder distance by BVH distance."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);names=('right_arm/link_2','torso/link_3');self.shoulder_names=names;self.shoulder_index=next(i for i,(a,b,m) in enumerate(self.pairs) if {a,b}==set(names));self.shoulder_objects={}
        for name in names:
            row=self.groups[name][0];self.shoulder_objects[name]=triangle_object(row['vertices'],ConvexHull(row['vertices']).simplices)
    def shoulder_distance(self,q):
        from sock_dressing_simulation.grasp_geometry import HANDEDNESS
        frames=self.k.fk(q,only=list(self.shoulder_names))
        for name,obj in self.shoulder_objects.items():
            T=frames[name];obj.setTransform(fcl.Transform(self.k.A@T[:3,:3]@HANDEDNESS,self.k.A@T[:3,3]+self.k.b))
        a,b=self.pairs[self.shoulder_index][:2];return max(0,float(fcl.distance(self.shoulder_objects[a],self.shoulder_objects[b],fcl.DistanceRequest(),fcl.DistanceResult())))
    def slacks(self,q,gap,delta,allow_cloth_overlap=False,cap=.02,details=False,with_jacobian=False):
        raw=super().slacks(q,gap,delta,allow_cloth_overlap=allow_cloth_overlap,cap=cap,details=details,with_jacobian=with_jacobian);i=self.shoulder_index;a,b,m=self.pairs[i];value=min(cap,self.shoulder_distance(q)-m)
        if details:
            raw['minimum_slack_m']=min(raw['minimum_slack_m'],value)
            if value < -1e-9:
                raw['issues']=[v for v in raw['issues'] if set(v['pair'])!={a,b}]+[dict(pair=[a,b],distance_method='triangle_surface_guard',signed_distance_m=value+m,required_clearance_m=m,slack_m=value)];raw['passed']=False
            return raw
        values,J=raw if with_jacobian else (raw,None)
        if value<values[i]:
            values[i]=value
            if with_jacobian:
                J[i]=0.;h=1e-5
                for j in [0,1]:
                    plus=np.array(q);minus=plus.copy();plus[j]+=h;minus[j]-=h;J[i,j]=(self.shoulder_distance(plus)-self.shoulder_distance(minus))/(2*h)
        return (values,J) if with_jacobian else values
    def is_clear(self,q,gap,delta):
        return self.shoulder_distance(q)>=.005-1e-9 and super().is_clear(q,gap,delta)


def start_margin_transition(q,lower,upper,start,margin_deg=5.):
    """Honor the explicit 131-degree start; only joint 4 may begin at 4 degrees.

    The exception ends at the first fully comfortable waypoint. Joint 4 must
    move monotonically inward until then and never exceed the requested start.
    """
    q=np.asarray(q);lower=np.asarray(lower);upper=np.asarray(upper);start=np.asarray(start);tol=1e-7
    margins=np.rad2deg(np.minimum(q-lower,upper-q));hard=bool(np.all(q>=lower-tol) and np.all(q<=upper+tol));other=bool(np.all(margins[:,[0,1,2,4,5,6]]>=margin_deg-1e-6));ready=np.flatnonzero(np.min(margins,axis=1)>=margin_deg-1e-6);index=int(ready[0]) if len(ready) else None
    inward=index is not None and bool(np.all(np.diff(q[:index+1,3])<=tol));after=index is not None and bool(np.all(margins[index:]>=margin_deg-1e-6));no_further_limit_approach=bool(np.all(q[:,3]<=max(start[3],upper[3]-np.deg2rad(margin_deg))+tol));matches=bool(np.allclose(q[0],start,atol=1e-7,rtol=0))
    return dict(passed=bool(hard and other and inward and after and no_further_limit_approach and matches),hard_limits_passed=hard,other_joints_margin_passed=other,initial_margin_deg=float(margins[0].min()),minimum_margin_deg=float(margins.min()),first_full_margin_index=index,inward_initial_motion=inward,margin_after_transition_passed=after,exception_joint='right_arm/joint_4',exception_reason='User-requested apron start has joint 4 = 131 degrees and a 135-degree upper limit',uniform_5deg_passed=bool(np.all(margins>=margin_deg-1e-6)))
