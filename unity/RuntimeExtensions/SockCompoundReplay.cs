#if OBI
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
using Obi;

namespace RCareWorld.Attributes
{
 internal sealed class SockReplayContactRecorder:MonoBehaviour
 {
  public readonly List<object> events=new List<object>();public int dropped;
  private void Record(Collision collision)
  {
   if(events.Count>=4096){dropped++;return;}
   foreach(var contact in collision.contacts)
   {
    var a=contact.thisCollider;var b=contact.otherCollider;
    events.Add(new Dictionary<string,object>{{"a",a.name},{"b",b.name},{"a_link",a.GetComponentInParent<ArticulationBody>()?.name??""},{"b_link",b.GetComponentInParent<ArticulationBody>()?.name??""},{"separation_m",contact.separation},{"point",new[]{contact.point.x,contact.point.y,contact.point.z}}});
   }
  }
  private void OnCollisionEnter(Collision c){Record(c);}private void OnCollisionStay(Collision c){Record(c);}
 }
 public partial class SockClothAttr
 {
  private readonly List<SockReplayContactRecorder> compoundReplayRecorders=new List<SockReplayContactRecorder>();
  [Serializable] private class ReplayDriveVelocity {public float[] values;}

  [RFUAPI] public void SetSockReplayFixedFloorPolicy(int robotId,bool ignore)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
   var root=robot.GetComponentsInChildren<ArticulationBody>(true).First(b=>b.isRoot);
   if(!root.immovable)throw new InvalidOperationException("Fixed chassis floor policy requires an immovable root");
   var floors=FindObjectsOfType<Collider>().Where(c=>c.enabled && !c.transform.IsChildOf(robot.transform) && ColliderHierarchyPath(c.transform).Contains("Ground")).ToArray();
   if(floors.Length!=1)throw new InvalidOperationException("Expected one floor collider");
   var pairs=new List<object>();
   foreach(var c in robot.GetComponentsInChildren<Collider>(true))
   {
    if(!c.enabled)continue;string link=c.GetComponentInParent<ArticulationBody>()?.name??"";
    if(link!="base_link" && link!="dummy_base_link" && !link.StartsWith("base_"))continue;
    Physics.IgnoreCollision(c,floors[0],ignore);pairs.Add(new Dictionary<string,object>{{"link",link},{"collider",ColliderHierarchyPath(c.transform)},{"ignored",ignore}});
   }
   collectData.AddDataNextStep("sock_replay_fixed_floor_policy",new Dictionary<string,object>{{"root_immovable",root.immovable},{"pairs",pairs},{"source","Stationary chassis versus floor; planning already excludes stationary links versus Ground"}});
  }

  [RFUAPI] public void ConfigureSockReplayDrives(int robotId,float holdStiffness,float holdDamping)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot) || !(robot is ControllerAttr controller) || holdStiffness<=0 || holdDamping<=0)throw new ArgumentException("Invalid replay drive settings");
   var rows=new List<object>();
   foreach(var body in robot.GetComponentsInChildren<ArticulationBody>(true))
   {
    body.solverIterations=64;body.solverVelocityIterations=16;
    if(body.jointType==ArticulationJointType.FixedJoint)continue;
    bool fixedSide=!body.name.StartsWith("right_arm/") && !body.name.StartsWith("right_gripper/");
    if(fixedSide)
    {
     var x=body.xDrive;x.stiffness=holdStiffness;x.damping=holdDamping;body.xDrive=x;
     var y=body.yDrive;y.stiffness=holdStiffness;y.damping=holdDamping;body.yDrive=y;
     var z=body.zDrive;z.stiffness=holdStiffness;z.damping=holdDamping;body.zDrive=z;
    }
    rows.Add(new Dictionary<string,object>{{"link",body.name},{"fixed_side",fixedSide},{"controller_master",controller.MoveableJoints.Contains(body)},{"stiffness",body.xDrive.stiffness},{"damping",body.xDrive.damping},{"solver_iterations",body.solverIterations},{"solver_velocity_iterations",body.solverVelocityIterations}});
   }
   collectData.AddDataNextStep("sock_replay_drive_settings",rows);
  }

  // Drive targets only. ControllerAttr.SetJointVelocity also overwrites the
  // articulation's actual velocity, which would invalidate a dynamics replay.
  [RFUAPI] public void SetSockReplayDriveVelocities(int robotId,string json)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot) || !(robot is ControllerAttr controller))throw new ArgumentException("Missing replay controller");
   var query=JsonUtility.FromJson<ReplayDriveVelocity>(json);
   if(query.values.Length!=controller.MoveableJoints.Count || query.values.Any(v=>float.IsNaN(v)||float.IsInfinity(v)))throw new ArgumentException("Invalid drive velocities");
   for(int i=0;i<query.values.Length;i++)
   {var body=controller.MoveableJoints[i];body.SetDriveTargetVelocity(ArticulationDriveAxis.X,query.values[i]);body.SetDriveTargetVelocity(ArticulationDriveAxis.Y,query.values[i]);body.SetDriveTargetVelocity(ArticulationDriveAxis.Z,query.values[i]);}
  }

  [RFUAPI] public void ReadSockRegisteredCompound(int start,int count,bool rays=true)
  {
   if(start<0 || count<1 || start+count>compoundObjects.Count)throw new ArgumentOutOfRangeException();
   Physics.SyncTransforms();var rows=new List<object>();var type=typeof(ObiCollider).Assembly.GetType("Obi.ObiConvexCollisionMesh",true);var get=type.GetMethod("Get",BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic);
   for(int index=start;index<start+count;index++)
   {
    var collider=compoundObjects[index].GetComponent<MeshCollider>();var mesh=collider.sharedMesh;var obiMesh=(Mesh)get.Invoke(null,new object[]{mesh});var vertices=mesh.vertices;var triangles=mesh.triangles;var observations=new List<object>();
    Vector3 center=compoundInteriors[index];Vector3 worldCenter=collider.transform.TransformPoint(center);float radius=vertices.Max(v=>Vector3.Distance(collider.transform.TransformPoint(v),worldCenter));
    void Shoot(Vector3 origin,Vector3 direction,float length,string kind)
    {
     direction=CompoundUnit(direction);bool found=collider.Raycast(new Ray(origin,direction),out RaycastHit hit,length);
     observations.Add(new Dictionary<string,object>{{"kind",kind},{"origin",PreinsertVector(origin)},{"direction",PreinsertVector(direction)},{"hit",found},{"distance",found?hit.distance:-1},{"point",PreinsertVector(found?hit.point:Vector3.zero)}});
    }
    if(rays)
    {
     for(int sample=0;sample<96;sample++)
     {double z=1-2*(sample+.5)/96.0,angle=sample*2.399963229728653;float rr=(float)Math.Sqrt(1-z*z);Vector3 direction=new Vector3(rr*(float)Math.Cos(angle),rr*(float)Math.Sin(angle),(float)z);Shoot(worldCenter+direction*(radius+.01f),-direction,2*radius+.02f,"sphere");}
     var normalTransform=collider.transform.localToWorldMatrix.inverse.transpose;
     for(int face=0;face<triangles.Length;face+=3)
     {
      Vector3 a=vertices[triangles[face]],b=vertices[triangles[face+1]],c=vertices[triangles[face+2]];
      double ux=(double)b.x-a.x,uy=(double)b.y-a.y,uz=(double)b.z-a.z,vx=(double)c.x-a.x,vy=(double)c.y-a.y,vz=(double)c.z-a.z;
      Vector3 localNormal=CompoundUnit(new Vector3((float)(uy*vz-uz*vy),(float)(uz*vx-ux*vz),(float)(ux*vy-uy*vx)));
      Vector3 normal=CompoundUnit(normalTransform.MultiplyVector(localNormal));
      // Every external ray passes through a certified common interior. Tiny
      // input triangles can disappear within the accepted cooking tolerance;
      // a rounded ray through their edge is not an expected-hit witness.
      Shoot(worldCenter+normal*(radius+.01f),-normal,2*radius+.02f,"face_direction");
     }
    }
    rows.Add(new Dictionary<string,object>{{"index",index},{"link",collider.GetComponentInParent<ArticulationBody>().name},{"matrix",CompoundMatrix(collider.transform.localToWorldMatrix)},{"vertices_world",vertices.Select(v=>PreinsertVector(collider.transform.TransformPoint(v))).ToList()},{"obi_vertices_world",obiMesh.vertices.Select(v=>PreinsertVector(collider.transform.TransformPoint(v))).ToList()},{"obi_triangles",obiMesh.triangles.ToList()},{"rays",observations},{"convex",collider.convex},{"obi_enabled",collider.GetComponent<ObiCollider>().enabled},{"contact_offset_m",collider.contactOffset}});
   }
   collectData.AddDataNextStep("sock_registered_compound",new Dictionary<string,object>{{"parts",rows},{"total_parts",compoundObjects.Count},{"physics_executed",false}});
  }

  [RFUAPI] public void TranslateSockReplayScene(int robotId,float x,float y,float z)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot) || sockHuman==null || !canonicalTaskPoseConfigured)throw new InvalidOperationException("Missing replay scene");
   var delta=new Vector3(x,y,z);var bodyRoot=sockHuman.root??sockHuman.transform;var transforms=new List<Transform>{bodyRoot};
   if(Attrs.TryGetValue(footClearanceChairId,out BaseAttr chair))transforms.Add(chair.transform);
   foreach(var collider in FindObjectsOfType<Collider>())
   {
    if(!collider.enabled || collider.isTrigger || collider.transform.IsChildOf(robot.transform) || collider.transform.IsChildOf(bodyRoot) || ColliderHierarchyPath(collider.transform).Contains("Ground"))continue;
    transforms.Add(collider.transform);
   }
   var roots=transforms.Distinct().Where(t=>!transforms.Any(p=>p!=t && t.IsChildOf(p))).ToArray();foreach(var root in roots)root.position+=delta;
   foreach(var target in sockHuman.ikTargets)if(target!=null && !roots.Any(t=>target.IsChildOf(t)))target.position+=delta;
   visualRightHip+=delta;visualRightKnee+=delta;visualRightFoot+=delta;visualSeatPosition+=delta;
   lockedVisualRightHip+=delta;lockedVisualRightKnee+=delta;lockedVisualRightFoot+=delta;lockedVisualSeatPosition+=delta;
   lockedRightToePosition+=delta;lockedHumanRootPosition+=delta;lockedChairPosition+=delta;rigidTaskPoseTranslation=true;targetFootClearance=-1;
   if(sockPatternHeldPositions!=null)for(int i=0;i<sockPatternHeldPositions.Length;i++)sockPatternHeldPositions[i]+=delta;
   SyncRightLegCollidersToBones();Physics.SyncTransforms();
   collectData.AddDataNextStep("sock_replay_translation",new Dictionary<string,object>{{"delta",PreinsertVector(delta)},{"moved_roots",roots.Select(t=>ColliderHierarchyPath(t)).ToList()}});
  }

  [RFUAPI] public void BeginSockCompoundReplay(int robotId)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new InvalidOperationException("Missing robot");
   foreach(var body in robot.GetComponentsInChildren<ArticulationBody>(true))
   {var recorder=body.gameObject.GetComponent<SockReplayContactRecorder>()??body.gameObject.AddComponent<SockReplayContactRecorder>();recorder.events.Clear();recorder.dropped=0;if(!compoundReplayRecorders.Contains(recorder))compoundReplayRecorders.Add(recorder);}
   GetSockCompoundReplay(robotId);
  }

  [RFUAPI] public void GetSockCompoundReplay(int robotId)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new InvalidOperationException("Missing robot");
   var links=robot.GetComponentsInChildren<ArticulationBody>(true).Select(b=>new Dictionary<string,object>{{"link",b.name},{"position",PreinsertVector(b.transform.position)},{"rotation",new[]{b.transform.rotation.x,b.transform.rotation.y,b.transform.rotation.z,b.transform.rotation.w}}}).ToList();
   collectData.AddDataNextStep("sock_compound_replay",new Dictionary<string,object>{{"links",links},{"contacts",compoundReplayRecorders.SelectMany(r=>r.events).ToList()},{"dropped",compoundReplayRecorders.Sum(r=>r.dropped)},{"cloth_fixed_steps",solver.externalFixedSteps},{"cloth_time_s",(float)solver.externalSimulatedTime},{"cloth_held",sockPatternHeldPositions!=null}});
   foreach(var recorder in compoundReplayRecorders){recorder.events.Clear();recorder.dropped=0;}
  }
 }
}
#endif
