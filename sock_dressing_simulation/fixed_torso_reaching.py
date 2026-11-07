"""Both-arm collision coverage for reselecting starts with an unchanged torso.

Geometry is calibrated from the original native snapshot before any arm joint
is changed. The left arm may then be parked and held while the right arm moves.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import fcl
import numpy as np
from scipy.spatial import ConvexHull
from .arm_reaching import triangle_object
from .compound_reaching import CompoundScene
from .grasp_geometry import HANDEDNESS

LEFT_ARM=[f'left_arm/joint_{i}' for i in range(1,8)]


class FixedTorsoScene(CompoundScene):
    def __init__(self,k,scene,faces,manifest,root):
        policy=json.loads((Path(root)/'artifacts/arm-reaching-reselection/model-self-collision-policy.json').read_text())
        srdf=Path(policy['source'])
        if hashlib.sha256(srdf.read_bytes()).hexdigest()!=policy['sha256']:
            raise ValueError('The model collision policy source changed')
        pairs=ET.parse(srdf).getroot().findall('disable_collisions')
        if not any({v.get('link1'),v.get('link2')}=={'left_arm/link_5','left_arm/link_7'} and v.get('reason')=='Default' for v in pairs):
            raise ValueError('Left nested wrist exclusion is not present in the matching SRDF')
        self.srdf_policy=dict(source=str(srdf),sha256=policy['sha256'],left_default_pair=['left_arm/link_5','left_arm/link_7'])
        super().__init__(k,scene,faces,manifest,root)
        self.left_lower=np.array([float(k.joints[n].find('limit').get('lower')) for n in LEFT_ARM])
        self.left_upper=np.array([float(k.joints[n].find('limit').get('upper')) for n in LEFT_ARM])
        self.reference_fixed_joints={n:v for n,v in k.q0.items() if not n.startswith(('right_arm/','left_arm/','right_gripper/','left_gripper/'))}
        self.reference_torso={n:v for n,v in k.q0.items() if n.startswith('torso/')}
        # The left arm is fixed during a right-arm trajectory, but all of its
        # self/body/obstacle pairs must be tested when selecting its parked pose.
        for row in self.shapes:
            if row['link'].startswith(('left_arm/','left_gripper/')):row['moving']=True
        names=list(self.groups);self.pairs=[]
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if not (self.groups[a][0]['moving'] or self.groups[b][0]['moving']) or self.excluded(a,b):continue
                same_hand=any(a.startswith(s+'_gripper/') and b.startswith(s+'_gripper/') for s in ('left','right'))
                margin=0. if same_hand else .005
                self.pairs.append((a,b,margin+self.padding.get(a,0)+self.padding.get(b,0)))
        for a in names:
            hand=a.startswith(('right_gripper/','left_gripper/'))
            for b,_ in self.obstacles:
                if 'Ground' in b and not self.groups[a][0]['moving']:continue
                margin=0. if b=='frozen_cloth' else (.001 if hand and 'skin' in b else .005)
                self.pairs.append((a,b,margin+self.padding.get(a,0)))
        self.shoulder_pairs={};self.shoulder_objects={}
        for side in ('right','left'):
            pair=(side+'_arm/link_2','torso/link_3')
            self.shoulder_pairs[side]=next(i for i,(a,b,m) in enumerate(self.pairs) if {a,b}==set(pair))
            for name in pair:
                if name not in self.shoulder_objects:
                    vertices=self.groups[name][0]['vertices']
                    self.shoulder_objects[name]=triangle_object(vertices,ConvexHull(vertices).simplices)

    def excluded(self,a,b):
        if {a,b}=={'left_arm/link_5','left_arm/link_7'}:return True
        if a.startswith('left_gripper/') and b.startswith('left_gripper/'):
            return {a,b}!={'left_gripper/left_finger_link','left_gripper/right_finger_link'}
        return super().excluded(a,b)

    def park_left(self,angles,gap=.04):
        angles=np.asarray(angles)
        if angles.shape!=(7,) or np.any(angles<self.left_lower-1e-9) or np.any(angles>self.left_upper+1e-9):
            raise ValueError('Left parked posture exceeds its URDF joint limits')
        for n,v in zip(LEFT_ARM,angles):self.k.q0[n]=float(v)
        self.k.q0['left_gripper/finger_joint']=self.k.hand.joint_for_gap(gap)
        self.assert_fixed()

    def allow_grounded_left_foot_penetration(self,depth_m):
        """Apply an explicit floor exception to the human, retaining other floors.

        The saved raised right leg has a separate skin surface. Its floor
        constraint, the sock floor, the support floor, and all robot collision
        pairs remain active. Record the allowance rather than moving the floor.
        """
        if not np.isfinite(depth_m) or depth_m<0 or depth_m>.25:
            raise ValueError('Grounded left foot allowance must be within 0–250 mm')
        if hasattr(self,'left_foot_floor_allowance_m'):
            raise ValueError('Floor allowance has already been configured')
        self.left_foot_floor_allowance_m=float(depth_m)
        self.original_human_minimum_y=self.floor_minima['human_full_skin']
        self.floor_minima['human_full_skin']+=float(depth_m)

    def assert_fixed(self):
        if any(self.k.q0[n]!=v for n,v in self.reference_fixed_joints.items()):
            raise ValueError('A torso, head or base joint changed during fixed-torso search')

    def shoulder_distance(self,q,side):
        pair=(side+'_arm/link_2','torso/link_3')
        frames=self.k.fk(q,only=pair)
        for name in pair:
            T=frames[name];self.shoulder_objects[name].setTransform(fcl.Transform(self.k.A@T[:3,:3]@HANDEDNESS,self.k.A@T[:3,3]+self.k.b))
        return max(0.,float(fcl.distance(*(self.shoulder_objects[n] for n in pair),fcl.DistanceRequest(),fcl.DistanceResult())))

    def slacks(self,q,gap,delta,allow_cloth_overlap=False,cap=.02,details=False,with_jacobian=False):
        self.assert_fixed()
        raw=super().slacks(q,gap,delta,allow_cloth_overlap=allow_cloth_overlap,cap=cap,details=details,with_jacobian=with_jacobian,cloth_overlap_side=getattr(self,'closure_side','right'))
        values,J=raw if with_jacobian else (raw,None)
        for side,i in self.shoulder_pairs.items():
            a,b,m=self.pairs[i];value=min(cap,self.shoulder_distance(q,side)-m)
            if details:
                raw['minimum_slack_m']=min(raw['minimum_slack_m'],value)
                if value < -1e-9:
                    raw['issues']=[v for v in raw['issues'] if set(v['pair'])!={a,b}]+[dict(pair=[a,b],distance_method='triangle_surface_guard',signed_distance_m=value+m,required_clearance_m=m,slack_m=value)]
                    raw['passed']=False
            elif value<values[i]:
                values[i]=value
                if with_jacobian:
                    J[i]=0.
                    if side=='right':
                        h=1e-5
                        for j in (0,1):
                            plus=np.array(q);minus=plus.copy();plus[j]+=h;minus[j]-=h
                            J[i,j]=(self.shoulder_distance(plus,side)-self.shoulder_distance(minus,side))/(2*h)
        return raw

    def is_clear(self,q,gap,delta):
        self.assert_fixed()
        return all(self.shoulder_distance(q,s)>=.005-1e-9 for s in ('left','right')) and super().is_clear(q,gap,delta)

    def gripper_clearances(self,q,gap,delta):
        values=self.slacks(q,gap,delta,cap=2.)
        result={side:dict(skin_m=float('inf'),cloth_m=float('inf'),robot_m=float('inf'),robot_body_m=float('inf')) for side in ('left','right')}
        for (a,b,margin),slack in zip(self.pairs,values):
            for side in ('left','right'):
                hand=side+'_gripper/'
                if not (a.startswith(hand) or b.startswith(hand)):continue
                other=b if a.startswith(hand) else a
                if other.startswith(hand):continue
                if 'skin' in other:kind='skin_m';base=.001
                elif other=='frozen_cloth':kind='cloth_m';base=0.
                elif other in self.groups:kind='robot_m';base=.005
                else:continue
                # Remove the policy margin but retain shape-coverage padding.
                result[side][kind]=min(result[side][kind],float(slack+base))
                if kind=='robot_m' and other not in {side+'_arm/link_5',side+'_arm/link_6',side+'_arm/link_7'}:
                    result[side]['robot_body_m']=min(result[side]['robot_body_m'],float(slack+base))
        return result
