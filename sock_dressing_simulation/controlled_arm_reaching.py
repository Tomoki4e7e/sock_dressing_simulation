"""Kinematics and collision adapters for either arm, preserving a fixed torso."""
import numpy as np
import hashlib
from .arm_reaching import Kinematics,ARM
from .fixed_torso_reaching import LEFT_ARM


def shared_gripper_geometry_audit(k):
    """Verify shared hand geometry below each gripper_base, excluding mounts."""
    mesh_rows=[];joint_rows=[]
    for tail in ('gripper_base','left_finger_base_link','right_finger_base_link','left_finger_link','right_finger_link'):
        values=[]
        for side in ('right','left'):
            link=k.tree.find("link[@name='"+side+'_gripper/'+tail+"']");collision=link.find('collision');mesh=collision.find('geometry/mesh');path=k.path.parent/mesh.get('filename')
            values.append(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale=mesh.get('scale'),origin=collision.find('origin').attrib if collision.find('origin') is not None else {}))
        mesh_rows.append(dict(link=tail,passed=values[0]==values[1],right=values[0],left=values[1]))
    def canonical(value):return value.replace('right_gripper/','gripper/').replace('left_gripper/','gripper/')
    def descendants(side):
        root=side+'_gripper/gripper_base';links={root};result=[]
        for _ in range(len(k.joints)):
            found=False
            for j in k.joints.values():
                if j.find('parent').get('link') not in links or j.find('child').get('link') in links:continue
                links.add(j.find('child').get('link'));found=True
                fields={t:None if j.find(t) is None else {a:canonical(v) for a,v in j.find(t).attrib.items()} for t in ('parent','child','origin','axis','mimic')}
                result.append(dict(name=canonical(j.get('name')),type=j.get('type'),fields=fields))
            if not found:break
        return sorted(result,key=lambda r:r['name'])
    right,left=descendants('right'),descendants('left');passed=all(r['passed'] for r in mesh_rows) and right==left
    return dict(passed=passed,meshes=mesh_rows,joint_graph_identical=right==left,right_joint_graph=right,left_joint_graph=left,
        scope='Identical collision STL bytes, scale, origins, and kinematic/mimic graphs below gripper_base. Different arm-to-gripper mounts are handled by full arm FK.')


class ControlledArmKinematics(Kinematics):
    def __init__(self,base,side,fixed_right_q=None,fixed_right_gap=.04):
        if side not in ('left','right'):raise ValueError('Unknown controlled arm')
        self.__dict__=base.__dict__.copy();self.base=base;self.side=side
        self.fixed_right_q=np.array(base.start if fixed_right_q is None else fixed_right_q);self.fixed_right_gap=fixed_right_gap
        self.arm_names=ARM if side=='right' else LEFT_ARM
        frames=base.fk(self.fixed_right_q);names=self.arm_names
        self.lower=np.array([float(base.joints[n].find('limit').get('lower')) for n in names]);self.upper=np.array([float(base.joints[n].find('limit').get('upper')) for n in names])
        self.velocity=np.array([float(base.joints[n].find('limit').get('velocity')) for n in names]);self.acceleration=np.array([float(base.joints[n].find('limit').get('acceleration')) for n in names])
        self.start=np.array([base.q0[n] for n in names]);self.arm_origins=[base.origins[n] for n in names];self.arm_axes=np.array([base.axes[n] for n in names])
        self.arm_parent=frames[base.joints[names[0]].find('parent').get('link')];last=base.joints[names[-1]].find('child').get('link')
        self.arm_tail=np.linalg.inv(frames[last])@frames[side+'_gripper/gripper_base']

    def fk(self,q,gap=None,only=None):
        if self.side=='right':return self.base.fk(q,gap,only)
        names=LEFT_ARM+['left_gripper/finger_joint'];old={n:self.base.q0[n] for n in names}
        try:
            self.base.q0.update(zip(LEFT_ARM,q))
            if gap is not None:self.base.q0[names[-1]]=self.hand.joint_for_gap(gap)
            return self.base.fk(self.fixed_right_q,self.fixed_right_gap,only)
        finally:self.base.q0.update(old)


class ControlledCollisionScene:
    def __init__(self,base,side,fixed_right_q,fixed_right_gap=.04):
        self.base=base;self.side=side;self.fixed_right_q=np.array(fixed_right_q);self.fixed_right_gap=fixed_right_gap

    def prepare(self,q,gap):
        self.base.closure_side=self.side
        if self.side=='left':
            # Fixed right-arm/body/obstacle pairs can be reused between left
            # poses. The key includes every right joint and its jaw opening;
            # CompoundScene adds translation and distance cap to each query.
            self.base.fixed_pair_cache_key=('left',tuple(self.fixed_right_q),float(self.fixed_right_gap))
            self.base.park_left(q,gap);return self.fixed_right_q,self.fixed_right_gap
        # With the right arm moving, bind the full held left posture before
        # reusing pairs that contain no right-arm or right-hand link.
        self.base.fixed_pair_cache_key=('right',tuple(self.base.k.q0[n]for n in LEFT_ARM),
                                       float(self.base.k.q0['left_gripper/finger_joint']))
        return q,gap

    def is_clear(self,q,gap,delta):
        right,right_gap=self.prepare(q,gap);return self.base.is_clear(right,right_gap,delta)

    def slacks(self,q,gap,delta,allow_cloth_overlap=False,cap=.02,details=False,with_jacobian=False):
        right,right_gap=self.prepare(q,gap)
        if self.side=='right' or not with_jacobian:
            return self.base.slacks(right,right_gap,delta,allow_cloth_overlap=allow_cloth_overlap,cap=cap,details=details,with_jacobian=with_jacobian)
        g,J=self.base.slacks(right,right_gap,delta,allow_cloth_overlap=allow_cloth_overlap,cap=cap,with_jacobian=True);G=J.copy();h=1e-5
        for j in range(7):
            plus=np.array(q);minus=plus.copy();plus[j]+=h;minus[j]-=h
            self.base.park_left(plus,gap);gp=self.base.slacks(right,right_gap,delta,allow_cloth_overlap=allow_cloth_overlap,cap=cap)
            self.base.park_left(minus,gap);gm=self.base.slacks(right,right_gap,delta,allow_cloth_overlap=allow_cloth_overlap,cap=cap);G[:,j]=(gp-gm)/(2*h)
        self.base.park_left(q,gap);return g,G

    def gripper_clearances(self,q,gap,delta):
        right,right_gap=self.prepare(q,gap);return self.base.gripper_clearances(right,right_gap,delta)
