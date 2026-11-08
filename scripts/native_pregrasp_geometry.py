"""Evaluate measured Unity link transforms, including torso/mimic drive drift."""
import numpy as np
import fcl
from scipy.spatial.transform import Rotation
from sock_dressing_simulation.arm_reaching import ROS_UNITY
from sock_dressing_simulation.grasp_geometry import HANDEDNESS
from sock_dressing_simulation.contact_pregrasp import rim_intervals


def native_update(scene,x,links):
    base=scene.base;frames={r['link']:r for r in links};base.transform_epoch+=1;base.group_transforms={};base.group_bounds={};scene.positions={};scene.rotations={}
    for name,rows in base.groups.items():
        link=frames[name];R=Rotation.from_quat(link['rotation']).as_matrix()@ROS_UNITY@HANDEDNESS;t=np.array(link['position'])
        base.group_transforms[name]=fcl.Transform(R,t);points=base.group_vertices[name]@R.T+t;starts=base.group_starts[name]
        lo=np.minimum.reduceat(points,starts,axis=0);hi=np.maximum.reduceat(points,starts,axis=0);base.group_bounds[name]=(lo,hi)
        for i,row in enumerate(rows):row['bounds']=(lo[i],hi[i])
    base.obstacle_bounds={}
    for name,obj in base.obstacles:
        d=np.zeros(3)if 'Ground'in name else scene.delta;obj.setTransform(fcl.Transform(np.eye(3),d));lo,hi=base.base_obstacle_bounds[name];base.obstacle_bounds[name]=(lo+d,hi+d)
    for side in ('left','right'):
        link=frames[side+'_gripper/gripper_base'];R=Rotation.from_quat(link['rotation']).as_matrix()
        scene.positions[side]=np.array(link['position'])+R@ROS_UNITY@scene.k.hand.grasp_center_urdf
        scene.rotations[side]=R@ROS_UNITY@HANDEDNESS
    for objects in scene.split.values():
        for obj in objects:
            if obj is not None:obj.setTransform(fcl.Transform(np.eye(3),scene.delta))
    for link,objects in scene.eroded.items():
        for obj in objects:obj.setTransform(base.group_transforms[link])
    def shoulder_distance(q,side):
        pair=(side+'_arm/link_2','torso/link_3')
        for name in pair:base.shoulder_objects[name].setTransform(base.group_transforms[name])
        return max(0.,float(fcl.distance(*(base.shoulder_objects[n]for n in pair),fcl.DistanceRequest(),fcl.DistanceResult())))
    base.shoulder_distance=shoulder_distance


def native_terminal(scene,saved):
    results={};goal=scene.goal
    for surface in (goal.band.obj,goal.foot.obj):surface.setTransform(fcl.Transform(np.eye(3),scene.delta))
    for side in ('left','right'):
        old=saved['final_contact_checks'][side];contacts={};p=scene.positions[side];R=scene.rotations[side]
        for kind,surface in (('cloth',goal.band.obj),('foot',goal.foot.obj)):
            name=side+'_gripper/'+old[kind+'_finger'];d=scene._distance([scene.base.query_object(r)for r in scene.base.groups[name]],surface)
            contacts[kind]=dict(part=name,distance_m=d if d>=0 else None,intersects=d<0,passed=d<=.0005)
        fingers=[]
        for suffix in ('left_finger_link','right_finger_link'):
            name=side+'_gripper/'+suffix;T=scene.base.group_transforms[name];world=scene.base.group_vertices[name]@T.getRotation().T+T.getTranslation();fingers.append((world-p)@R)
        fingers.sort(key=lambda v:v[:,1].mean());rim=rim_intervals((goal.rim+scene.delta-p)@R,*fingers);length=sum(v['length_m']for v in rim)
        signed=float((1 if side=='left'else-1)*np.dot(p-scene.delta-goal.center,goal.lateral))
        passed=bool(all(c['passed']for c in contacts.values())and length>=goal.criteria.minimum_rim_length_m and signed>0 and scene.window(side,np.zeros(16)))
        results[side]=dict(relaxed_passed=passed,passed=passed,contacts=contacts,rim_length_m=length,rim_segments=rim,signed_foot_side_m=signed,
            tcp_world_m=p.tolist(),rotation=R.tolist(),pose_source='Measured Unity gripper and individual finger link transforms; no fixed-torso FK substitution',
            overlap_scope='An intersecting contact passes only if the full-scene inward-offset-claw test also passes')
    return results
