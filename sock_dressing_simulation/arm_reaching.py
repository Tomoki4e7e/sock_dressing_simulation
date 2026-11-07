"""Right-arm kinematics and collision checking from an authoritative Unity snapshot.

The offline model is calibrated to a native link, then checked against every
native articulation frame before planning. Angles are radians, lengths metres.
"""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation
from .grasp_geometry import origin,RightHand,HANDEDNESS,convex_geometry
import fcl

ROS_UNITY=np.array([[0.,-1.,0.],[0.,0.,1.],[1.,0.,0.]])
ARM=[f'right_arm/joint_{i}' for i in range(1,8)]

class Kinematics:
    def __init__(self,urdf,scene):
        self.path=Path(urdf);self.tree=ET.parse(urdf).getroot()
        self.joints={j.get('name'):j for j in self.tree.findall('joint')}
        self.parents={j.find('child').get('link'):j for j in self.joints.values()}
        self.origins={k:origin(j.find('origin')) for k,j in self.joints.items()}
        self.axes={k:np.fromstring(j.find('axis').get('xyz','1 0 0'),sep=' ') if j.find('axis') is not None else np.array([1.,0.,0.]) for k,j in self.joints.items()}
        self.hand=RightHand(urdf)
        data=scene['robot_data'];mimic={k for k,j in self.joints.items() if j.find('mimic') is not None}
        independent=[(n,t) for n,t in zip(data['names'],data['types']) if t!='FixedJoint' and n not in mimic]
        if len(independent)!=len(data['joint_positions']):raise ValueError('Native movable joint mapping differs')
        self.q0={n:float(v if t=='PrismaticJoint' else np.deg2rad(v)) for (n,t),v in zip(independent,data['joint_positions'])}
        self.lower=np.array([float(self.joints[n].find('limit').get('lower')) for n in ARM])
        self.upper=np.array([float(self.joints[n].find('limit').get('upper')) for n in ARM])
        native_limits={n:(float(np.deg2rad(lo)),float(np.deg2rad(hi))) for (n,t),lo,hi in
            zip(independent,data['joint_lower_limit'],data['joint_upper_limit']) if n in ARM}
        self.native_limits=native_limits
        if any(abs(native_limits[n][0]-lo)>1e-5 or abs(native_limits[n][1]-hi)>1e-5 for n,lo,hi in zip(ARM,self.lower,self.upper)):
            raise ValueError('Native joint limits differ from the planning URDF')
        self.velocity=np.array([float(self.joints[n].find('limit').get('velocity')) for n in ARM])
        self.acceleration=np.array([float(self.joints[n].find('limit').get('acceleration')) for n in ARM])
        self.start=np.array([self.q0[n] for n in ARM])
        native={r['name']:r for r in scene['links']}
        link='right_gripper/gripper_base'
        anchor=native[link];T=self.fk(self.start)[link]
        self.A=Rotation.from_quat(anchor['quaternion']).as_matrix()@ROS_UNITY@T[:3,:3].T
        self.b=np.asarray(anchor['position'])-self.A@T[:3,3]
        frames=self.fk(self.start);errors=[]
        for name,r in native.items():
            if name not in frames:continue
            expected=self.A@frames[name][:3,3]+self.b
            actual=np.asarray(r['position'])
            expected_rotation=self.A@frames[name][:3,:3]@ROS_UNITY.T
            angle=Rotation.from_matrix(expected_rotation@Rotation.from_quat(r['quaternion']).as_matrix().T).magnitude()
            errors.append(dict(link=name,position_error_m=float(np.linalg.norm(expected-actual)),orientation_error_rad=float(angle)))
        self.calibration=dict(frames=errors,maximum_position_error_m=max(r['position_error_m'] for r in errors),maximum_orientation_error_rad=max(r['orientation_error_rad'] for r in errors))
        if self.calibration['maximum_position_error_m']>2e-5 or self.calibration['maximum_orientation_error_rad']>2e-4:
            raise ValueError('Native/URDF FK mismatch: '+json.dumps([r for r in errors if r['position_error_m']>2e-5 or r['orientation_error_rad']>2e-4]))
        self.arm_origins=[self.origins[n] for n in ARM]
        self.arm_axes=np.asarray([self.axes[n] for n in ARM])
        first=self.joints[ARM[0]].find('parent').get('link')
        last=self.joints[ARM[-1]].find('child').get('link')
        self.arm_parent=frames[first]
        self.arm_tail=np.linalg.inv(frames[last])@frames['right_gripper/gripper_base']

    def fk(self,q,gap=None,only=None):
        values=dict(self.q0);values.update(zip(ARM,q))
        if gap is not None:values['right_gripper/finger_joint']=self.hand.joint_for_gap(gap)
        def value(j):
            mimic=j.find('mimic')
            if mimic is not None:return value(self.joints[mimic.get('joint')])*float(mimic.get('multiplier','1'))+float(mimic.get('offset','0'))
            return values.get(j.get('name'),0.)
        frames={}
        def frame(link):
            if link in frames:return frames[link]
            if link not in self.parents:T=np.eye(4)
            else:
                j=self.parents[link];name=j.get('name');T=frame(j.find('parent').get('link'))@self.origins[name]
                if j.get('type')!='fixed':
                    move=np.eye(4)
                    if j.get('type')=='prismatic':move[:3,3]=self.axes[name]*value(j)
                    else:move[:3,:3]=Rotation.from_rotvec(self.axes[name]*value(j)).as_matrix()
                    T=T@move
            frames[link]=T;return T
        for name in ([link.get('name') for link in self.tree.findall('link')] if only is None else only):frame(name)
        return frames

    def pose(self,q):
        if hasattr(self,'arm_parent'):return self.fast_pose(q)[:2]
        T=self.fk(q,only=['right_gripper/gripper_base'])['right_gripper/gripper_base']
        return self.A@(T[:3,3]+T[:3,:3]@self.hand.grasp_center_urdf)+self.b,self.A@T[:3,:3]@HANDEDNESS

    def fast_pose(self,q):
        T=self.arm_parent.copy();axes=[];points=[]
        for a,O,axis in zip(q,self.arm_origins,self.arm_axes):
            T=T@O;points.append(T[:3,3].copy());axes.append(T[:3,:3]@axis)
            T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(axis*a).as_matrix()
        T=T@self.arm_tail;raw=T[:3,3]+T[:3,:3]@self.hand.grasp_center_urdf
        angular=-np.asarray(axes)@self.A.T
        jac_position=self.A@np.cross(np.asarray(axes),raw-np.asarray(points)).T
        return self.A@raw+self.b,self.A@T[:3,:3]@HANDEDNESS,jac_position,angular.T

    def solve(self,p,R,seed):
        def residual(q):
            actual,rotation=self.pose(q)
            return np.r_[actual-p,.15*Rotation.from_matrix(rotation@R.T).as_rotvec()]
        def jacobian(q):
            _,rotation,position_jac,angular_jac=self.fast_pose(q)
            phi=Rotation.from_matrix(rotation@R.T).as_rotvec();theta=np.linalg.norm(phi)
            x,y,z=phi;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            a=1/12 if theta<1e-5 else (1-.5*theta/np.tan(.5*theta))/theta**2
            inverse=np.eye(3)-.5*skew+a*skew@skew
            return np.vstack([position_jac,.15*inverse@angular_jac])
        result=least_squares(residual,np.clip(seed,self.lower+1e-9,self.upper-1e-9),bounds=(self.lower,self.upper),
                             jac=jacobian if hasattr(self,'arm_parent') else '2-point',max_nfev=180,ftol=1e-11,xtol=1e-11,gtol=1e-11)
        actual,rotation=self.pose(result.x)
        return dict(q=result.x.tolist(),position_error_m=float(np.linalg.norm(actual-p)),
                    orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(rotation@R.T).magnitude())),evaluations=result.nfev)

    def multistart(self,p,R,count=48):
        rng=np.random.default_rng(20261104);solutions=[];attempts=[]
        for i in range(count):
            seed=getattr(self,'preferred_seed',self.start) if i==0 else rng.uniform(self.lower,self.upper)
            result=self.solve(p,R,seed);attempts.append(result)
            if result['position_error_m']<=.0001 and result['orientation_error_deg']<=.5:
                q=np.array(result['q'])
                if all(np.linalg.norm(q-np.asarray(r['q']))>.05 for r in solutions):solutions.append(result)
        solutions.sort(key=lambda r:np.linalg.norm(np.asarray(r['q'])-self.start))
        return solutions,attempts

    def position_constrained_orientation(self,p,R,count=24):
        """Find reachable-position seeds, then minimize attitude at that position."""
        rng=np.random.default_rng(30412);results=[]
        for i in range(count):
            seed=self.start if i==0 else rng.uniform(self.lower,self.upper)
            position=least_squares(lambda q:self.pose(q)[0]-p,seed,bounds=(self.lower,self.upper),
                                   max_nfev=220,gtol=1e-10,ftol=1e-10,xtol=1e-10)
            if np.linalg.norm(self.pose(position.x)[0]-p)>.0001:continue
            def attitude(q):return Rotation.from_matrix(self.pose(q)[1]@R.T).magnitude()**2
            optimized=minimize(attitude,position.x,method='SLSQP',bounds=list(zip(self.lower,self.upper)),
                constraints=[dict(type='eq',fun=lambda q:self.pose(q)[0]-p)],
                options=dict(maxiter=250,ftol=1e-11))
            q=optimized.x;actual,rotation=self.pose(q)
            results.append(dict(q=q.tolist(),position_error_m=float(np.linalg.norm(actual-p)),
                orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(rotation@R.T).magnitude())),
                optimizer_success=bool(optimized.success)))
        results.sort(key=lambda r:(r['position_error_m']>.0001,r['orientation_error_deg']))
        return results


def triangle_object(vertices,faces):
    vertices=np.asarray(vertices,dtype=float);faces=np.asarray(faces,dtype=np.int32).reshape(-1,3)
    model=fcl.BVHModel();model.beginModel(len(faces),len(vertices));model.addSubModel(vertices,faces);model.endModel()
    return fcl.CollisionObject(model)


def shape_vertices(row):
    if row['kind']=='MeshCollider':return np.asarray(row['vertices_world'])
    if row['kind']!='BoxCollider':raise ValueError('Unsupported native planning shape '+row['kind'])
    size=np.asarray(row['size']);center=np.asarray(row['center']);scale=np.asarray(row['scale'])
    points=np.array([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])*size+center
    return (points*scale)@Rotation.from_quat(row['rotation']).as_matrix().T+row['position']


class CollisionScene:
    def __init__(self,k,scene,faces):
        self.k=k;self.shapes=[];frames=k.fk(k.start)
        for row in scene['robot_colliders']:
            link=row['link'];T=frames[link];points=shape_vertices(row)
            local=((points-k.b)@k.A-T[:3,3])@T[:3,:3]
            geometry,vertices,_=convex_geometry(local@HANDEDNESS)
            self.shapes.append(dict(link=link,obj=fcl.CollisionObject(geometry),vertices=vertices,
                moving=link.startswith('right_arm/') or link.startswith('right_gripper/')))
        self.obstacles=[('human_full_skin',triangle_object(scene['human_skin']['vertices'],scene['human_skin']['triangles']))]
        for row in scene['obstacles']:
            if row['kind']=='MeshCollider' and not row['convex']:
                obj=triangle_object(row['vertices_world'],row['triangles'])
            else:
                geometry,_,_=convex_geometry(shape_vertices(row));obj=fcl.CollisionObject(geometry)
            self.obstacles.append((row['name'],obj))
        self.obstacles.append(('frozen_cloth',triangle_object(scene['particles_world'],faces)))
        # Fixed-joint assemblies and directly connected bodies are mechanical
        # neighbours, not collision-avoidance pairs. Never disable cross-arm pairs.
        self.group={link.get('name'):link.get('name') for link in k.tree.findall('link')}
        def root(x):
            while self.group[x]!=x:x=self.group[x]
            return x
        for j in k.joints.values():
            if j.get('type')=='fixed':self.group[root(j.find('child').get('link'))]=root(j.find('parent').get('link'))
        self.group={link:root(link) for link in self.group}
        self.neighbours=set()
        for j in k.joints.values():
            self.neighbours.add(frozenset([self.group[j.find('parent').get('link')],self.group[j.find('child').get('link')]]))
        # The matching Dry-AIREC gripper_v2 SRDF explicitly marks this
        # nested wrist housing pair as Default. Do not infer exclusions
        # merely because a pair collided in the starting posture.
        self.model_exclusions={frozenset(['right_arm/link_5','right_arm/link_7'])}

    def excluded(self,a,b):
        if frozenset([a,b]) in getattr(self,'model_exclusions',set()):return True
        if a.startswith('right_gripper/') and b.startswith('right_gripper/'):
            return {a,b}!={'right_gripper/left_finger_link','right_gripper/right_finger_link'}
        ga,gb=self.group[a],self.group[b]
        return ga==gb or frozenset([ga,gb]) in self.neighbours

    def check(self,q,gap,allow_cloth_overlap=False):
        frames=self.k.fk(q,gap);issues=[];minimum=float('inf');limiting=None
        for row in self.shapes:
            T=frames[row['link']];row['obj'].setTransform(fcl.Transform(self.k.A@T[:3,:3]@HANDEDNESS,self.k.A@T[:3,3]+self.k.b))
        def inspect(a,b,margin,name):
            nonlocal minimum,limiting
            raw_distance=float(fcl.distance(a,b,fcl.DistanceRequest(),fcl.DistanceResult()))
            if not np.isfinite(raw_distance):raise ValueError('Nonfinite collision distance')
            # FCL's -1 sentinel means intersection, not one metre penetration.
            distance=max(0.,raw_distance)
            if distance-margin<minimum:minimum=distance-margin;limiting=name
            if distance<margin-1e-9 or raw_distance<=0:issues.append(dict(pair=name,distance_m=distance,intersects=raw_distance<=0,required_clearance_m=margin))
        for i,a in enumerate(self.shapes):
            if not a['moving']:continue
            hand=a['link'].startswith('right_gripper/')
            for name,obj in self.obstacles:
                if name=='frozen_cloth' and allow_cloth_overlap and hand:continue
                margin=0. if name=='frozen_cloth' else (.001 if hand and ('skin' in name) else .005)
                inspect(a['obj'],obj,margin,[a['link'],name])
            for j,b in enumerate(self.shapes):
                if j==i or (b['moving'] and j<i) or self.excluded(a['link'],b['link']):continue
                margin=0. if hand and b['link'].startswith('right_gripper/') else .005
                inspect(a['obj'],b['obj'],margin,[a['link'],b['link']])
        return dict(passed=not issues,issues=issues,minimum_margin_slack_m=minimum,limiting_pair=limiting)


def verify(env,manifest,scene,out,snapshot):
    # This gate precedes any robot commands. A failed endpoint cannot be
    # repaired by a motion planner, so never replay the nearest IK iterate.
    return evaluate(manifest,scene,out)


def evaluate(manifest,scene,out):
    out=Path(out);k=Kinematics(manifest['urdf'],scene);c=manifest['candidate']['candidate']
    p=np.asarray(c['grasp_center_unity_world_m']);R=np.asarray(c['rotation_matrix'])
    solutions,attempts=k.multistart(p,R,count=96)
    pre_solutions,pre_attempts=k.multistart(p-R[:,0]*.05,R,count=48)
    constrained=k.position_constrained_orientation(p,R,count=24)
    cost=lambda r:r['position_error_m']**2+(.15*np.deg2rad(r['orientation_error_deg']))**2
    best=min(attempts,key=cost)
    limits=[dict(joint=n,angle_deg=float(np.rad2deg(v)),lower_deg=float(np.rad2deg(lo)),upper_deg=float(np.rad2deg(hi)),
                 at_lower=bool(abs(v-lo)<1e-4),at_upper=bool(abs(v-hi)<1e-4)) for n,v,lo,hi in zip(ARM,best['q'],k.lower,k.upper)]
    raw=json.loads((Path(__file__).resolve().parents[1]/manifest['candidate']['initial_state_source']).read_text())
    checker=CollisionScene(k,scene,raw['triangles'])
    initial_gap=k.q0['right_gripper/finger_joint']+k.hand.gap_offset_m
    initial_collision=checker.check(k.start,initial_gap)
    open_start_collision=checker.check(k.start,manifest['candidate']['opening_gap_m'])
    result=dict(number=manifest['number'],candidate_index=manifest['candidate_index'],
        status='stopped_at_ik_gate' if not solutions else 'requires_collision_free_path_planning',
        reaching_executed=False,closure_executed=False,robot_base_or_torso_moved=False,
        position_tolerance_m=.0001,orientation_tolerance_deg=.5,calibration=k.calibration,
        ik_solution_count=len(solutions),ik_seed_count=len(attempts),best_weighted_ik=best,
        preapproach_ik_solution_count=len(pre_solutions),preapproach_ik_seed_count=len(pre_attempts),
        position_constrained_orientation=constrained,active_joint_bounds=limits,
        initial_collision=initial_collision,open_start_collision=open_start_collision,
        collision_scope='12 moving right-arm/hand convex pieces versus all remaining robot pieces, full human skin, foot skin, chair, floor and frozen cloth; mechanical adjacency exclusions',
        ik_attempts=attempts,preapproach_ik_attempts=pre_attempts,
        note='Multistart failure is not a mathematical proof of infeasibility. No collision-free path is claimed when the IK gate fails.')
    result['native_joint_limits_rad']=k.native_limits
    software=manifest['config']['joints'];software={n:(lo,hi) for n,lo,hi in zip(software['names'],software['lower'],software['upper'])}
    result['software_limit_mismatches']=[dict(joint=n,configured_rad=software[n],native_rad=k.native_limits[n]) for n in ARM if not np.allclose(software[n],k.native_limits[n],atol=1e-5)]
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(number=manifest['number'],status=result['status'],ik_solutions=len(solutions),preapproach_solutions=len(pre_solutions),
        best_position_error_mm=best['position_error_m']*1000,best_orientation_error_deg=best['orientation_error_deg'],
        initial_collision=initial_collision,open_start_collision=open_start_collision)),flush=True)
    return result
