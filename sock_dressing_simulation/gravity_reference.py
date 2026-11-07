"""Static URDF gravity loads, independent of drive gains and motion forces.

The output is the generalized effort required to hold a configuration. It is
the gradient of potential energy, rather than a servo/position-error force.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES=[name for side in ('left','right') for name in
             ([f'{side}_arm/joint_{i}' for i in range(1,8)]+[f'{side}_gripper/finger_joint',f'{side}_gripper/mimic_joint'])]
ARM_COLUMNS=list(range(7))+list(range(9,16))
ARM_NAMES=[JOINT_NAMES[i] for i in ARM_COLUMNS]
SOCK_TORSO={f'torso/joint_{i}':np.deg2rad(v) for i,v in enumerate((-45.,100.,0.),1)}


class StaticGravityReference:
    """Load the calibrated physical model without ROS or the Unity SDK."""
    def __init__(self,path):
        self.path=Path(path);self.metadata=json.loads(self.path.read_text())
        urdf=Path(self.metadata['calibrated_urdf'])
        if hashlib.sha256(urdf.read_bytes()).hexdigest()!=self.metadata['calibrated_urdf_sha256']:
            raise ValueError('Calibrated gravity URDF changed')
        self.tree=GravityTree(urdf)

    def predict(self,q):
        return self.tree.predict(q,fixed=self.metadata['fixed_joint_positions_rad'],gravity=self.metadata['gravity_root_m_s2'])

    def configure_unity(self,env):
        """Set corrected inertials and gravity for an isolated static hold.

        This explicit profile removes dry friction/body drag and uses
        Kp=1e5, Kd=5e3 drive gains; it does not identify real servo dynamics.
        """
        payload=dict(bodies=[dict(name=name,mass=mass,center_of_mass_ros=self.tree.coms[name].tolist()) for name,mass in self.tree.masses.items()])
        env._env.SetGravity(0.,-9.81,0.)
        env.sock_cloth._send_data('ConfigureSockGravityReference',env.robot.id,json.dumps(payload));env._env.step(simulate=False)
        result=env.sock_cloth.data.get('sock_gravity_reference_applied')
        if not result:raise RuntimeError('RegisteredGravityV1 player is required')
        if not all(b['use_gravity'] for b in result['after']):raise RuntimeError('Joint gravity was not enabled')
        for b in result['after']:
            name=b['name'];mass=max(self.tree.masses[name],1e-6)
            if abs(b['mass_kg']-mass)>max(1e-7,mass*1e-6):raise RuntimeError('Native mass differs from gravity model')
            com=np.array([-self.tree.coms[name][1],self.tree.coms[name][2],self.tree.coms[name][0]])
            if np.linalg.norm(np.array(b['center_of_mass_unity_local'])-com)>1e-6:raise RuntimeError('Native COM differs from gravity model')
            if any(abs(b.get(key,1.))>1e-9 for key in ('joint_friction','linear_damping','angular_damping')):
                raise RuntimeError('Gravity-only profile still contains dry friction or body drag')
        return result


def origin(element):
    t=np.eye(4)
    if element is not None:
        t[:3,3]=np.fromstring(element.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(element.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t


class GravityTree:
    def __init__(self,urdf):
        self.path=Path(urdf);root=ET.parse(urdf).getroot()
        self.links={e.get('name'):e for e in root.findall('link')}
        self.joints={e.get('name'):e for e in root.findall('joint')}
        self.parent={e.find('child').get('link'):e for e in self.joints.values()}
        self.origins={name:origin(e.find('origin')) for name,e in self.joints.items()}
        self.axes={name:np.fromstring(e.find('axis').get('xyz','1 0 0'),sep=' ') if e.find('axis') is not None else np.array([1.,0.,0.]) for name,e in self.joints.items()}
        self.masses={};self.coms={}
        for name,e in self.links.items():
            inertial=e.find('inertial')
            self.masses[name]=float(inertial.find('mass').get('value')) if inertial is not None else 0.
            self.coms[name]=origin(inertial.find('origin'))[:3,3] if inertial is not None else np.zeros(3)
        self.groups=[f'{side}_arm/link_{i}' for side in ('left','right') for i in range(1,8)]
        self.groups += ['left_gripper','right_gripper']
        self.group_index={g:i for i,g in enumerate(self.groups)}
        self.body_group={}
        self.ancestors={}
        for link in self.links:
            chain=[];p=link
            while p in self.parent:
                j=self.parent[p];chain.append(j.get('name'));p=j.find('parent').get('link')
            self.ancestors[link]=chain
            if link.startswith(('left_gripper/','right_gripper/')):
                self.body_group[link]=self.group_index[link.split('/')[0]]
            else:
                for joint in chain:
                    if joint in ARM_NAMES:
                        self.body_group[link]=self.group_index[self.joints[joint].find('child').get('link')];break

    def frames(self,q,fixed=None):
        q=np.atleast_2d(np.asarray(q,dtype=float))
        if q.shape[1]!=18 or not np.isfinite(q).all():raise ValueError('Expected finite N by 18 joint coordinates')
        n=len(q);values=dict(SOCK_TORSO if fixed is None else fixed)
        values.update({name:q[:,i] for i,name in enumerate(JOINT_NAMES)})
        frames={};joint_frames={}
        def position(j):
            m=j.find('mimic')
            if m is not None:return position(self.joints[m.get('joint')])*float(m.get('multiplier','1'))+float(m.get('offset','0'))
            return np.broadcast_to(values.get(j.get('name'),0.),(n,))
        def frame(link):
            if link in frames:return frames[link]
            if link not in self.parent:t=np.broadcast_to(np.eye(4),(n,4,4)).copy()
            else:
                j=self.parent[link];name=j.get('name');t=frame(j.find('parent').get('link'))@self.origins[name];joint_frames[name]=t.copy()
                if j.get('type')!='fixed':
                    d=position(j);move=np.broadcast_to(np.eye(4),(n,4,4)).copy();axis=self.axes[name]
                    if j.get('type')=='prismatic':move[:,:3,3]=d[:,None]*axis
                    else:move[:,:3,:3]=Rotation.from_rotvec(d[:,None]*axis).as_matrix()
                    t=t@move
            frames[link]=t;return t
        for link in self.links:frame(link)
        return frames,joint_frames

    def contributions(self,q,fixed=None,gravity=(0.,0.,-9.81)):
        """N × 14 arm efforts × 16 positive body-mass scale factors."""
        frames,joints=self.frames(q,fixed);n=len(np.atleast_2d(q));result=np.zeros((n,14,len(self.groups)))
        index={name:i for i,name in enumerate(ARM_NAMES)}
        g=np.asarray(gravity,dtype=float)
        for link,group in self.body_group.items():
            mass=self.masses[link]
            if mass==0:continue
            t=frames[link];com=t[:,:3,3]+np.einsum('nij,j->ni',t[:,:3,:3],self.coms[link]);force=-mass*g
            for name in self.ancestors[link]:
                if name not in index:continue
                j=joints[name];axis=np.einsum('nij,j->ni',j[:,:3,:3],self.axes[name])
                result[:,index[name],group]+=np.cross(axis,com-j[:,:3,3])@force
        return result

    def predict(self,q,scales=None,fixed=None,gravity=(0.,0.,-9.81)):
        scales=np.ones(len(self.groups)) if scales is None else np.asarray(scales,dtype=float)
        if scales.shape!=(len(self.groups),) or np.any(scales<=0):raise ValueError('Mass scales must be positive')
        return self.contributions(q,fixed,gravity)@scales

    def regressor(self,q,fixed=None,gravity=(0.,0.,-9.81)):
        """Mass scales and group first-moment corrections, in kg*m.

        These static data do not identify inertia tensors. Regularization is
        required: individual masses/centres need not be uniquely identifiable.
        """
        mass=self.contributions(q,fixed,gravity);frames,joints=self.frames(q,fixed)
        h=np.zeros((*mass.shape,3));index={name:i for i,name in enumerate(ARM_NAMES)}
        for gi,group in enumerate(self.groups):
            anchor=group if '_arm/' in group else group+'/gripper_base'
            for name in self.ancestors[anchor]:
                if name not in index:continue
                axis=np.einsum('nij,j->ni',joints[name][:,:3,:3],self.axes[name])
                for k in range(3):h[:,index[name],gi,k]=np.cross(axis,frames[anchor][:,:3,k])@(-np.asarray(gravity))
        return np.concatenate((mass,h.reshape(len(mass),14,-1)),axis=2)

    @property
    def group_masses(self):
        return np.array([sum(self.masses[name] for name,g in self.body_group.items() if g==i) for i in range(len(self.groups))])

    def write_calibrated_urdf(self,destination,parameters):
        """Apply positive mass scales and small COM translations to a copy."""
        parameters=np.asarray(parameters);ng=len(self.groups)
        scales=parameters[:ng];moments=parameters[ng:].reshape(ng,3)
        if np.any(scales<=0):raise ValueError('Negative physical mass')
        tree=ET.parse(self.path);frames,_=self.frames(np.zeros((1,18)))
        for e in tree.getroot().findall('link'):
            name=e.get('name');gi=self.body_group.get(name)
            if gi is None or e.find('inertial') is None or self.masses[name]==0:continue
            group=self.groups[gi];anchor=group if '_arm/' in group else group+'/gripper_base'
            shift=frames[name][0,:3,:3].T@frames[anchor][0,:3,:3]@moments[gi]/(scales[gi]*self.group_masses[gi])
            inertial=e.find('inertial');inertial.find('mass').set('value',repr(self.masses[name]*scales[gi]))
            o=inertial.find('origin')
            if o is None:o=ET.SubElement(inertial,'origin')
            o.set('xyz',' '.join(map(repr,self.coms[name]+shift)))
            tensor=inertial.find('inertia')
            if tensor is not None:
                for key,value in list(tensor.attrib.items()):tensor.set(key,repr(float(value)*scales[gi]))
        for mesh in tree.getroot().iter('mesh'):
            name=mesh.get('filename','')
            if name and not Path(name).is_absolute() and '://' not in name:
                mesh.set('filename',str((self.path.parent/name).resolve()))
        Path(destination).parent.mkdir(parents=True,exist_ok=True);tree.write(destination,encoding='utf-8',xml_declaration=True)

    def potential(self,q,scales=None,fixed=None,gravity=(0.,0.,-9.81)):
        frames,_=self.frames(q,fixed);n=len(np.atleast_2d(q));energy=np.zeros(n)
        scales=np.ones(len(self.groups)) if scales is None else np.asarray(scales)
        for link,group in self.body_group.items():
            t=frames[link];com=t[:,:3,3]+np.einsum('nij,j->ni',t[:,:3,:3],self.coms[link])
            energy-=self.masses[link]*scales[group]*(com@np.asarray(gravity))
        return energy
