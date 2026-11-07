#if OBI
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using Unity.Robotics.UrdfImporter;

namespace RCareWorld.Attributes
{
    public partial class SockClothAttr
    {
        [Serializable] private class GravityReferenceBody
        {
            public string name;
            public float mass;
            public float[] center_of_mass_ros;
        }
        [Serializable] private class GravityReferenceSpec
        {
            public GravityReferenceBody[] bodies;
        }

        private List<object> GravityReferenceSnapshot(BaseAttr robot)
        {
            return robot.GetComponentsInChildren<ArticulationBody>(true).Select(body => (object)new Dictionary<string,object> {
                {"name",body.name},{"mass_kg",body.mass},{"center_of_mass_unity_local",PreinsertVector(body.centerOfMass)},
                {"center_of_mass_world",PreinsertVector(body.worldCenterOfMass)},
                {"position",PreinsertVector(body.transform.position)},
                {"quaternion",new float[] {body.transform.rotation.x,body.transform.rotation.y,body.transform.rotation.z,body.transform.rotation.w}},
                {"use_gravity",body.useGravity},{"joint_type",body.jointType.ToString()},{"root",body.isRoot},
                {"joint_friction",body.jointFriction},{"linear_damping",body.linearDamping},{"angular_damping",body.angularDamping},
                {"drive_stiffness",body.xDrive.stiffness},{"drive_damping",body.xDrive.damping}
            }).ToList();
        }

        [RFUAPI] public void GetSockGravityReference(int robotId)
        {
            if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
            collectData.AddDataNextStep("sock_gravity_reference",new Dictionary<string,object> {
                {"gravity_world",PreinsertVector(Physics.gravity)},{"bodies",GravityReferenceSnapshot(robot)}});
        }

        [RFUAPI] public void SynchronizeSockGravityReferenceMimics(int robotId)
        {
            if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
            var done=new HashSet<ArticulationUnit>();var pending=new HashSet<ArticulationUnit>();int count=0;
            Action<ArticulationUnit> sync=null;
            sync=unit=> {
                if(done.Contains(unit))return;
                if(!pending.Add(unit))throw new InvalidOperationException("Cyclic mimic mapping");
                if(unit.mimicParent!=null)
                {
                    var parent=unit.mimicParent.GetComponent<ArticulationUnit>();sync(parent);
                    unit.SetJointPositionDirectly(parent.CalculateCurrentJointPosition()*unit.mimicMultiplier+unit.mimicOffset);
                    var body=unit.GetComponent<ArticulationBody>();
                    body.jointVelocity=new ArticulationReducedSpace(0f,0f,0f);
                    var drive=body.xDrive;drive.targetVelocity=0;body.xDrive=drive;count++;
                }
                pending.Remove(unit);done.Add(unit);
            };
            foreach(var unit in robot.GetComponentsInChildren<ArticulationUnit>(true))sync(unit);
            collectData.AddDataNextStep("sock_gravity_mimics_synchronized",count);
        }

        [RFUAPI] public void ConfigureSockGravityReference(int robotId,string json)
        {
            if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
            var spec=JsonUtility.FromJson<GravityReferenceSpec>(json);
            var parameters=spec.bodies.ToDictionary(body=>body.name);
            var before=GravityReferenceSnapshot(robot);
            var bodies=robot.GetComponentsInChildren<ArticulationBody>(true);
            // Missing URDF inertials otherwise leave Unity's default 1kg on
            // fixed tip/load/grasp helper links. Use a documented positive
            // numerical floor; do not turn those helpers into payloads.
            foreach(var body in bodies)
            {
                if(!parameters.TryGetValue(body.name,out GravityReferenceBody entry))
                    throw new ArgumentException("Unmapped articulation body: "+body.name);
                if(entry.mass<0 || entry.center_of_mass_ros==null || entry.center_of_mass_ros.Length!=3)
                    throw new ArgumentException("Invalid gravity inertial: "+body.name);
            }
            foreach(var body in bodies)
            {
                var entry=parameters[body.name];var c=entry.center_of_mass_ros;
                body.mass=Mathf.Max(entry.mass,1e-6f);
                body.centerOfMass=new Vector3(-c[1],c[2],c[0]);
                body.useGravity=true;
                // The SDK SetJointDamping helper also sets jointFriction and
                // rigid-body drag to the same value. That stiction can carry
                // gravity without motor effort. Isolate gravity explicitly.
                body.jointFriction=0;body.linearDamping=0;body.angularDamping=0;
                if(body.dofCount>0)
                {
                    var x=body.xDrive;x.stiffness=1e5f;x.damping=5e3f;body.xDrive=x;
                    var y=body.yDrive;y.stiffness=1e5f;y.damping=5e3f;body.yDrive=y;
                    var z=body.zDrive;z.stiffness=1e5f;z.damping=5e3f;body.zDrive=z;
                }
                if(entry.mass==0)
                {
                    body.inertiaTensor=Vector3.one*1e-8f;
                    body.inertiaTensorRotation=Quaternion.identity;
                }
                var inertial=body.GetComponent<UrdfInertial>();
                if(inertial!=null)
                {
                    inertial.centerOfMass=body.centerOfMass;
                    inertial.inertiaTensor=body.inertiaTensor;
                    inertial.useUrdfData=true;
                }
            }
            collectData.AddDataNextStep("sock_gravity_reference_applied",new Dictionary<string,object> {
                {"before",before},{"after",GravityReferenceSnapshot(robot)},
                {"gravity_world",PreinsertVector(Physics.gravity)}, {"minimum_mass_kg",1e-6f},
                {"geometry_changed",false},{"inertia_identified",false},
                {"friction_and_body_drag",0f},{"hold_stiffness",1e5f},{"hold_drive_damping",5e3f}});
        }
    }
}
#endif
