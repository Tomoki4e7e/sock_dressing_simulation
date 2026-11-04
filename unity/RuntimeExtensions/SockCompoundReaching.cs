#if OBI
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
using Obi;
namespace RCareWorld.Attributes
{
 internal sealed class SockCompoundContactRecorder:MonoBehaviour
 {
  public readonly List<ContactPoint> contacts=new List<ContactPoint>();
  private void OnCollisionEnter(Collision collision){contacts.AddRange(collision.contacts);}
  private void OnCollisionStay(Collision collision){contacts.AddRange(collision.contacts);}
 }
 public partial class SockClothAttr
 {
  [Serializable] private class CompoundPartQuery {public string link;public Vector3[] vertices;public int[] triangles;}
  [Serializable] private class CompoundQuery {public CompoundPartQuery[] parts;public float cookUnitScale=100f;}
  private readonly List<GameObject> compoundObjects=new List<GameObject>();
  [Serializable] private class CompoundSupportSeed {public Vector3 anchor,direction;}
  [Serializable] private class CompoundProbePart
  {
   public int index;public string link,variant,cookingMode;public Vector3[] vertices;public int[] triangles;public bool primitiveBox,skipBake;
   public Vector3 position,scale;public Quaternion rotation;public Vector3[] targets,directions;public bool measureSupport,supportStretch;
   public CompoundSupportSeed[] supportSeeds;
  }
  [Serializable] private class CompoundProbeQuery {public CompoundProbePart[] parts;}
  [Serializable] private class CompoundContactPoint {public int sample;public string kind;public Vector3 center,normal;public float radius;}
  [Serializable] private class CompoundContactRay {public int sample;public string kind;public Vector3 origin,direction;public float length;}
  [Serializable] private class CompoundContactPart {public CompoundProbePart shape;public CompoundContactPoint[] points;public CompoundContactRay[] rays;public int sphereDirections;public bool vertexDirections;public float[] interiorContactRadii;public bool surfaceOverlapProbes;public float surfaceProbeRadius;public bool approachContacts;}
  [Serializable] private class CompoundContactQuery {public CompoundContactPart[] parts;public bool sceneOnly,identityPose;}
  private static List<List<float>> CompoundMatrix(Matrix4x4 m)
  {var rows=new List<List<float>>();for(int i=0;i<4;i++){var row=new List<float>();for(int j=0;j<4;j++)row.Add(m[i,j]);rows.Add(row);}return rows;}
  private static Collider CompoundProbeCollider(GameObject obj,Mesh mesh,CompoundProbePart p,MeshColliderCookingOptions cooking)
  {
   if(p.primitiveBox){var box=obj.AddComponent<BoxCollider>();box.center=mesh.bounds.center;box.size=mesh.bounds.size;return box;}
   var collider=obj.AddComponent<MeshCollider>();collider.convex=true;collider.cookingOptions=cooking;collider.sharedMesh=mesh;return collider;
  }
  private static MeshColliderCookingOptions CompoundCooking(CompoundProbePart p)
  {
   if(p.cookingMode=="none")return MeshColliderCookingOptions.None;
   if(p.cookingMode=="default")return MeshColliderCookingOptions.CookForFasterSimulation|MeshColliderCookingOptions.EnableMeshCleaning|MeshColliderCookingOptions.WeldColocatedVertices|MeshColliderCookingOptions.UseFastMidphase;
   return MeshColliderCookingOptions.CookForFasterSimulation|MeshColliderCookingOptions.UseFastMidphase;
  }
  // Read supporting faces in the cooked mesh's own coordinates. This avoids
  // world-position rounding and non-uniform mesh scale in the query itself.
  [RFUAPI] public void ReadSockCompoundCookedPlanes(string json)
  {
   var query=JsonUtility.FromJson<CompoundProbeQuery>(json);var rows=new List<object>();
   foreach(var p in query.parts)
   {
    var mesh=new Mesh{name="compound_plane_probe_"+p.index};mesh.vertices=p.vertices;mesh.triangles=p.triangles;mesh.RecalculateBounds();var obj=new GameObject("compound_plane_probe");
    try
    {
     var cooking=CompoundCooking(p);if(!p.skipBake)Physics.BakeMesh(mesh.GetInstanceID(),true,cooking);var collider=CompoundProbeCollider(obj,mesh,p,cooking);Physics.SyncTransforms();
     Vector3 center=new Vector3((float)p.vertices.Average(v=>(double)v.x),(float)p.vertices.Average(v=>(double)v.y),(float)p.vertices.Average(v=>(double)v.z));float radius=p.vertices.Max(v=>Vector3.Distance(v,center));var directions=new List<Vector3>();var kinds=new List<string>();
     for(int sample=0;sample<512;sample++){double z=1-2*(sample+.5)/512.0,angle=sample*2.399963229728653;float rr=(float)Math.Sqrt(1-z*z);directions.Add(new Vector3(rr*(float)Math.Cos(angle),rr*(float)Math.Sin(angle),(float)z));kinds.Add("sphere");}
     foreach(var v in p.vertices){directions.Add(CompoundUnit(v-center));kinds.Add("input_vertex");}
     if(p.targets!=null)foreach(var v in p.targets){directions.Add(CompoundUnit(v-center));kinds.Add("reconstructed_vertex");}
     if(p.directions!=null)foreach(var d in p.directions){directions.Add(CompoundUnit(d));kinds.Add("independent");}
     var rays=new List<object>();
     for(int i=0;i<directions.Count;i++)
     {
      var ray=new Ray(center+directions[i]*(4*radius+1),-directions[i]);bool found=collider.Raycast(ray,out RaycastHit hit,8*radius+2);
      rays.Add(new Dictionary<string,object>{{"kind",kinds[i]},{"origin",PreinsertVector(ray.origin)},{"direction",PreinsertVector(ray.direction)},{"hit",found},{"distance",found?hit.distance:-1},{"point",PreinsertVector(found?hit.point:Vector3.zero)},{"normal",PreinsertVector(found?hit.normal:Vector3.zero)}});
     }
     var supports=new List<object>();var vertexProjections=new List<object>();
     if(p.measureSupport)
     {
      Vector3 stretch=p.supportStretch?new Vector3(1/mesh.bounds.size.x,1/mesh.bounds.size.y,1/mesh.bounds.size.z):Vector3.one;
      var supportObject=new GameObject("compound_support_probe");supportObject.transform.localScale=stretch;var supportCollider=CompoundProbeCollider(supportObject,mesh,p,cooking);Physics.SyncTransforms();
      Vector3 supportCenter=Vector3.Scale(center,stretch);var supportVertices=p.vertices.Select(v=>Vector3.Scale(v,stretch)).ToArray();float supportRadius=supportVertices.Max(v=>Vector3.Distance(v,supportCenter));
      var queryDirections=new List<Vector3>(directions.Take(512));var anchors=new List<Vector3>();for(int i=0;i<512;i++)anchors.Add(supportCenter);
      for(int i=0;i<p.vertices.Length;i++)
      {
       var normals=new List<Vector3>();
       for(int f=0;f<p.triangles.Length;f+=3)if(p.triangles[f]==i||p.triangles[f+1]==i||p.triangles[f+2]==i)
       {Vector3 a=supportVertices[p.triangles[f]],b=supportVertices[p.triangles[f+1]],c=supportVertices[p.triangles[f+2]],n=CompoundUnit(Vector3.Cross(b-a,c-a));if(Vector3.Dot(n,supportVertices[i]-supportCenter)<0)n=-n;if(!normals.Any(v=>Vector3.Dot(v,n)>.999999f))normals.Add(n);}
       Vector3 normal=Vector3.zero;foreach(var n in normals)normal+=n;if(normal.sqrMagnitude==0)normal=supportVertices[i]-supportCenter;queryDirections.Add(CompoundUnit(normal));anchors.Add(supportVertices[i]);
      }
      int ordinaryDirections=queryDirections.Count;
      if(p.supportSeeds!=null)foreach(var seed in p.supportSeeds){queryDirections.Add(CompoundUnit(Vector3.Scale(seed.direction,stretch)));anchors.Add(Vector3.Scale(seed.anchor,stretch));}
      for(int i=0;i<queryDirections.Count;i++)foreach(float multiplier in new[]{1024f,65536f})
      {
       Vector3 queryPoint=anchors[i]+queryDirections[i]*(supportRadius*multiplier+1),closest=supportCollider.ClosestPoint(queryPoint),normal=Vector3.Scale(queryPoint-closest,stretch);
       supports.Add(new Dictionary<string,object>{{"kind",i<512?"direction":i<ordinaryDirections?"input_vertex_normal_cone":"serialized_vertex_normal_cone"},{"index",i<512?i:i-512},{"range_multiplier",multiplier},{"query",PreinsertVector(supportObject.transform.InverseTransformPoint(queryPoint))},{"point",PreinsertVector(supportObject.transform.InverseTransformPoint(closest))},{"normal",PreinsertVector(normal)}});
      }
      // Input vertices themselves are independent probes of vertex removal.
      // Unlike far support queries these do not rely on discovering a narrow
      // vertex normal cone. Keep each observation, including inside returns.
      for(int i=0;i<supportVertices.Length;i++)foreach(float distance in new[]{0f,.000001f,.0001f,.01f})
      {
       Vector3 queryPoint=supportVertices[i]+queryDirections[512+i]*distance;
       Vector3 closest=supportCollider.ClosestPoint(queryPoint);
       vertexProjections.Add(new Dictionary<string,object>{{"index",i},{"offset_query_units",distance},{"query",PreinsertVector(supportObject.transform.InverseTransformPoint(queryPoint))},{"point",PreinsertVector(supportObject.transform.InverseTransformPoint(closest))}});
      }
      DestroyImmediate(supportObject);
     }
     rows.Add(new Dictionary<string,object>{{"index",p.index},{"variant",p.variant},{"rays",rays},{"supports",supports},{"vertex_projections",vertexProjections},{"vertices",p.vertices.Select(PreinsertVector).ToList()},{"bounds_center",PreinsertVector(collider.bounds.center)},{"bounds_size",PreinsertVector(collider.bounds.size)}});
    }
    finally{DestroyImmediate(obj);DestroyImmediate(mesh);}
   }
   collectData.AddDataNextStep("sock_compound_cooked_planes",new Dictionary<string,object>{{"parts",rows},{"physics_executed",false}});
  }
  // Advance only a disposable local physics scene. The robot and cloth scene
  // remains frozen; this tests actual contact generation, not query helpers.
  [RFUAPI] public void SimulateSockCompoundContacts(int robotId,string json)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
   var bodies=robot.GetComponentsInChildren<ArticulationBody>(true).ToDictionary(b=>b.name);
   var query=JsonUtility.FromJson<CompoundContactQuery>(json);var rows=new List<object>();
   var scene=UnityEngine.SceneManagement.SceneManager.CreateScene("compound_contact_test_"+Guid.NewGuid().ToString("N"),new UnityEngine.SceneManagement.CreateSceneParameters(UnityEngine.SceneManagement.LocalPhysicsMode.Physics3D));
   var physicsScene=scene.GetPhysicsScene();
   try
   {
    foreach(var part in query.parts)
    {
     var p=part.shape;var pose=new GameObject("compound_contact_pose");pose.transform.SetParent(bodies[p.link].transform,false);pose.transform.localPosition=p.position;pose.transform.localRotation=p.rotation;pose.transform.localScale=p.scale;
     if(query.identityPose)pose.transform.SetPositionAndRotation(Vector3.zero,Quaternion.identity);
     var mesh=new Mesh{name="compound_simulated_"+p.index};mesh.vertices=p.vertices;mesh.triangles=p.triangles;mesh.RecalculateBounds();var target=new GameObject("compound_contact_target");
     UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(target,scene);target.transform.SetPositionAndRotation(Vector3.zero,pose.transform.rotation);target.transform.localScale=pose.transform.lossyScale;
     try
     {
      var cooking=CompoundCooking(p);if(!p.skipBake)Physics.BakeMesh(mesh.GetInstanceID(),true,cooking);var collider=CompoundProbeCollider(target,mesh,p,cooking);collider.contactOffset=.00001f;
      var points=new List<CompoundContactPoint>(part.points??new CompoundContactPoint[0]);var vertices=p.vertices.Select(v=>pose.transform.TransformPoint(v)).ToArray();
      Vector3 interior=new Vector3((float)vertices.Average(v=>(double)v.x),(float)vertices.Average(v=>(double)v.y),(float)vertices.Average(v=>(double)v.z));
      if(part.interiorContactRadii!=null)foreach(float r in part.interiorContactRadii)points.Add(new CompoundContactPoint{sample=-1,kind="interior",center=interior,radius=r});
      if(part.surfaceOverlapProbes)
      {
       float probeRadius=part.surfaceProbeRadius>0?part.surfaceProbeRadius:.0001f;
       for(int i=0;i<vertices.Length;i++)
       {
        Vector3 normal=Vector3.zero;float bestArea=-1;
        for(int f=0;f<p.triangles.Length;f+=3)if(p.triangles[f]==i||p.triangles[f+1]==i||p.triangles[f+2]==i)
        {Vector3 va=p.vertices[p.triangles[f]],vb=p.vertices[p.triangles[f+1]],vc=p.vertices[p.triangles[f+2]];Vector3 cross=Vector3.Cross(pose.transform.TransformVector(vb-va),pose.transform.TransformVector(vc-va));float area=cross.sqrMagnitude;if(area>bestArea){bestArea=area;normal=CompoundUnit(cross);}}
        if(Vector3.Dot(normal,vertices[i]-interior)<0)normal=-normal;points.Add(new CompoundContactPoint{sample=i,kind="vertex_near",center=vertices[i],normal=normal,radius=probeRadius});
       }
       for(int i=0;i<p.triangles.Length;i+=3)
       {
        Vector3 va=p.vertices[p.triangles[i]],vb=p.vertices[p.triangles[i+1]],vc=p.vertices[p.triangles[i+2]],center=pose.transform.TransformPoint((va+vb+vc)/3);
        Vector3 normal=CompoundUnit(pose.transform.localToWorldMatrix.inverse.transpose.MultiplyVector(Vector3.Cross(vb-va,vc-va)));if(Vector3.Dot(normal,center-interior)<0)normal=-normal;
        points.Add(new CompoundContactPoint{sample=i/3,kind="face_near",center=center,normal=normal,radius=probeRadius});points.Add(new CompoundContactPoint{sample=i/3,kind="face_outside",center=center+normal*(probeRadius+.0001f),radius=probeRadius});
       }
      }
      var contacts=new List<object>();
      foreach(var point in points)
      {
       var ball=new GameObject("compound_contact_ball");UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(ball,scene);Vector3 start=point.center-pose.transform.position;ball.transform.position=start;
       var sphere=ball.AddComponent<SphereCollider>();sphere.radius=point.radius;sphere.contactOffset=.00001f;var recorder=ball.AddComponent<SockCompoundContactRecorder>();var body=ball.AddComponent<Rigidbody>();body.useGravity=false;body.sleepThreshold=0;body.maxDepenetrationVelocity=.1f;body.WakeUp();
       int steps=part.approachContacts&&point.normal.sqrMagnitude>0?15:1;
       for(int step=0;step<steps;step++)
       {
        Vector3 center=steps==1?start:start+point.normal*(point.radius+.0005f-step*.00005f);body.position=center;body.velocity=Vector3.zero;body.angularVelocity=Vector3.zero;body.WakeUp();recorder.contacts.Clear();Physics.SyncTransforms();physicsScene.Simulate(1f/240f);
        var hits=recorder.contacts.Select(c=>new Dictionary<string,object>{{"point",PreinsertVector(c.point)},{"normal",PreinsertVector(c.normal)},{"separation_m",c.separation}}).ToList();
        contacts.Add(new Dictionary<string,object>{{"sample",point.sample},{"kind",point.kind},{"approach_step",steps==1?-1:step},{"radius_m",point.radius},{"center_shifted",PreinsertVector(center)},{"center_after",PreinsertVector(body.position)},{"contacts",hits}});
       }
       DestroyImmediate(ball);
      }
      var q=pose.transform.rotation;rows.Add(new Dictionary<string,object>{{"index",p.index},{"variant",p.variant},{"vertices_local",p.vertices.Select(PreinsertVector).ToList()},{"world_position",PreinsertVector(pose.transform.position)},{"world_rotation",new List<float>{q.x,q.y,q.z,q.w}},{"world_scale",PreinsertVector(pose.transform.lossyScale)},{"contacts",contacts}});
     }
     finally{DestroyImmediate(target);DestroyImmediate(pose);DestroyImmediate(mesh);}
    }
   }
   finally{UnityEngine.SceneManagement.SceneManager.UnloadSceneAsync(scene);}
   collectData.AddDataNextStep("sock_compound_simulated_contacts",new Dictionary<string,object>{{"parts",rows},{"physics_executed",true},{"isolated_scene_only",true},{"robot_motion_executed",false}});
  }
  [RFUAPI] public void ProbeSockCompoundContacts(int robotId,string json)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
   var bodies=robot.GetComponentsInChildren<ArticulationBody>(true).ToDictionary(b=>b.name);
   var query=JsonUtility.FromJson<CompoundContactQuery>(json);var rows=new List<object>();
   var sphereObject=new GameObject("compound_contact_sphere");var sphere=sphereObject.AddComponent<SphereCollider>();
   var controlObject=new GameObject("compound_contact_control");var control=controlObject.AddComponent<BoxCollider>();
   var controls=new List<object>();sphere.radius=.2f;
   foreach(float x in new[]{.6f,.8f})
   {bool overlap=Physics.ComputePenetration(sphere,new Vector3(x,0,0),Quaternion.identity,control,Vector3.zero,Quaternion.identity,out Vector3 normal,out float depth);controls.Add(new Dictionary<string,object>{{"x",x},{"overlap",overlap},{"depth",depth}});}
   DestroyImmediate(controlObject);
   try
   {
    foreach(var part in query.parts)
    {
     var p=part.shape;var mesh=new Mesh{name="compound_contact_"+p.index};mesh.vertices=p.vertices;mesh.triangles=p.triangles;mesh.RecalculateBounds();
     var world=new GameObject("compound_contact_world");var shifted=new GameObject("compound_contact_shifted");
     try
     {
      world.layer=31;shifted.layer=31;world.transform.SetParent(bodies[p.link].transform,false);world.transform.localPosition=p.position;world.transform.localRotation=p.rotation;world.transform.localScale=p.scale;
      if(query.identityPose)world.transform.SetPositionAndRotation(Vector3.zero,Quaternion.identity);
      shifted.transform.SetPositionAndRotation(Vector3.zero,world.transform.rotation);shifted.transform.localScale=world.transform.lossyScale;
      var cooking=CompoundCooking(p);if(!p.skipBake)Physics.BakeMesh(mesh.GetInstanceID(),true,cooking);
      var a=CompoundProbeCollider(world,mesh,p,cooking);
      var b=CompoundProbeCollider(shifted,mesh,p,cooking);Physics.SyncTransforms();
      var contacts=new List<object>();var rays=new List<object>();var queryPoints=new List<CompoundContactPoint>(part.points??new CompoundContactPoint[0]);var queryRays=new List<CompoundContactRay>(part.rays??new CompoundContactRay[0]);
      var vertices=p.vertices.Select(v=>world.transform.TransformPoint(v)).ToArray();
      Vector3 interior=new Vector3((float)vertices.Average(v=>(double)v.x),(float)vertices.Average(v=>(double)v.y),(float)vertices.Average(v=>(double)v.z));float radius=vertices.Max(v=>Vector3.Distance(v,interior));
      for(int i=0;i<part.sphereDirections+(part.vertexDirections?vertices.Length:0);i++)
      {
       Vector3 direction;string kind;
       if(i<part.sphereDirections){double z=1-2*(i+.5)/part.sphereDirections,angle=i*2.399963229728653;float rr=(float)Math.Sqrt(1-z*z);direction=new Vector3(rr*(float)Math.Cos(angle),rr*(float)Math.Sin(angle),(float)z);kind="world_sphere";}
       else{direction=CompoundUnit(vertices[i-part.sphereDirections]-interior);kind="world_vertex";}
       queryRays.Add(new CompoundContactRay{sample=i,kind=kind,origin=interior+direction*(radius+.01f),direction=-direction,length=2*radius+.02f});
      }
      if(part.interiorContactRadii!=null)foreach(float r in part.interiorContactRadii)queryPoints.Add(new CompoundContactPoint{sample=-1,kind="interior",center=interior,radius=r});
      if(part.surfaceOverlapProbes)
      {
       float probeRadius=part.surfaceProbeRadius>0?part.surfaceProbeRadius:.0001f;
       for(int i=0;i<vertices.Length;i++)queryPoints.Add(new CompoundContactPoint{sample=i,kind="vertex_near",center=vertices[i],radius=probeRadius});
       for(int i=0;i<p.triangles.Length;i+=3)
       {
        Vector3 va=p.vertices[p.triangles[i]],vb=p.vertices[p.triangles[i+1]],vc=p.vertices[p.triangles[i+2]],center=world.transform.TransformPoint((va+vb+vc)/3);
        Vector3 normal=CompoundUnit(world.transform.localToWorldMatrix.inverse.transpose.MultiplyVector(Vector3.Cross(vb-va,vc-va)));
        if(Vector3.Dot(normal,center-interior)<0)normal=-normal;
        queryPoints.Add(new CompoundContactPoint{sample=i/3,kind="face_near",center=center,radius=probeRadius});
        queryPoints.Add(new CompoundContactPoint{sample=i/3,kind="face_outside",center=center+normal*(probeRadius+.0001f),radius=probeRadius});
       }
      }
      foreach(var point in queryPoints)
      {
       sphere.radius=point.radius;Vector3 center=point.center-world.transform.position;
       Vector3 normal=Vector3.zero,shiftedNormal=Vector3.zero;float depth=0,shiftedDepth=0;bool hit=false,shiftedHit=false;
       if(!query.sceneOnly)
       {hit=Physics.ComputePenetration(sphere,point.center,Quaternion.identity,a,world.transform.position,world.transform.rotation,out normal,out depth);
        shiftedHit=Physics.ComputePenetration(sphere,center,Quaternion.identity,b,Vector3.zero,shifted.transform.rotation,out shiftedNormal,out shiftedDepth);}
       bool sceneHit=Physics.OverlapSphere(point.center,point.radius,1<<31,QueryTriggerInteraction.Collide).Contains(a);
       bool shiftedSceneHit=Physics.OverlapSphere(center,point.radius,1<<31,QueryTriggerInteraction.Collide).Contains(b);
       contacts.Add(new Dictionary<string,object>{{"sample",point.sample},{"kind",point.kind},{"center_world",PreinsertVector(point.center)},{"center_shifted",PreinsertVector(center)},{"radius_m",point.radius},
        {"compute_penetration_executed",!query.sceneOnly},{"scene_overlap",sceneHit},{"shifted_scene_overlap",shiftedSceneHit},
        {"overlap",hit},{"depth_m",depth},{"direction",PreinsertVector(normal)},{"shifted_overlap",shiftedHit},{"shifted_depth_m",shiftedDepth},{"shifted_direction",PreinsertVector(shiftedNormal)},
        {"closest_world",PreinsertVector(a.ClosestPoint(point.center))},{"closest_shifted",PreinsertVector(b.ClosestPoint(center))}});
      }
      foreach(var ray in queryRays)
      {
       var nativeRay=new Ray(ray.origin,ray.direction);var shiftedRay=new Ray(ray.origin-world.transform.position,ray.direction);
       bool hit=a.Raycast(nativeRay,out RaycastHit h,ray.length),shiftedHit=b.Raycast(shiftedRay,out RaycastHit sh,ray.length);
       rays.Add(new Dictionary<string,object>{{"sample",ray.sample},{"kind",ray.kind??"residual"},{"origin_world",PreinsertVector(nativeRay.origin)},{"direction_world",PreinsertVector(nativeRay.direction)},
        {"origin_shifted",PreinsertVector(shiftedRay.origin)},{"direction_shifted",PreinsertVector(shiftedRay.direction)},{"hit",hit},{"distance_m",hit?h.distance:-1},{"shifted_hit",shiftedHit},{"shifted_distance_m",shiftedHit?sh.distance:-1}});
      }
      var q=world.transform.rotation;
      rows.Add(new Dictionary<string,object>{{"index",p.index},{"variant",p.variant},{"link",p.link},{"vertices_local",p.vertices.Select(PreinsertVector).ToList()},
       {"vertices_world",p.vertices.Select(v=>PreinsertVector(world.transform.TransformPoint(v))).ToList()},
       {"world_matrix",CompoundMatrix(world.transform.localToWorldMatrix)},{"world_position",PreinsertVector(world.transform.position)},
       {"world_rotation",new List<float>{q.x,q.y,q.z,q.w}},{"world_scale",PreinsertVector(world.transform.lossyScale)},
       {"shifted_matrix",CompoundMatrix(shifted.transform.localToWorldMatrix)},{"contacts",contacts},{"rays",rays}});
     }
     finally{DestroyImmediate(world);DestroyImmediate(shifted);DestroyImmediate(mesh);}
    }
   }
   finally{DestroyImmediate(sphereObject);}
   collectData.AddDataNextStep("sock_compound_contacts",new Dictionary<string,object>{{"parts",rows},{"controls",controls},{"physics_executed",false},{"robot_motion_executed",false}});
  }
  private static Vector3 CompoundUnit(Vector3 v)
  {
   double length=Math.Sqrt((double)v.x*v.x+(double)v.y*v.y+(double)v.z*v.z);
   return length>0?v/(float)length:Vector3.zero;
  }
  // Static diagnostics only: compare the same cooked mesh at unit coordinates
  // and at its actual articulation transform, without replacing any colliders.
  [RFUAPI] public void ProbeSockCompoundGeometry(int robotId,string json)
  {
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
   var bodies=robot.GetComponentsInChildren<ArticulationBody>(true).ToDictionary(b=>b.name);
   var query=JsonUtility.FromJson<CompoundProbeQuery>(json);var rows=new List<object>();
   foreach(var p in query.parts)
   {
    var mesh=new Mesh{name="compound_diagnostic_"+p.index+"_"+p.variant};mesh.vertices=p.vertices;mesh.triangles=p.triangles;mesh.RecalculateBounds();
    var unit=new GameObject("compound_unit_probe");var world=new GameObject("compound_world_probe");
    world.transform.SetParent(bodies[p.link].transform,false);world.transform.localPosition=p.position;world.transform.localRotation=p.rotation;world.transform.localScale=p.scale;
    var cooking=MeshColliderCookingOptions.CookForFasterSimulation|MeshColliderCookingOptions.UseFastMidphase;
    Physics.BakeMesh(mesh.GetInstanceID(),true,cooking);
    var a=unit.AddComponent<MeshCollider>();a.convex=true;a.cookingOptions=cooking;a.sharedMesh=mesh;
    var b=world.AddComponent<MeshCollider>();b.convex=true;b.cookingOptions=cooking;b.sharedMesh=mesh;
    Physics.SyncTransforms();
    Vector3 interior=new Vector3((float)p.vertices.Average(v=>(double)v.x),(float)p.vertices.Average(v=>(double)v.y),(float)p.vertices.Average(v=>(double)v.z));
    float radius=p.vertices.Max(v=>Vector3.Distance(v,interior));var rays=new List<object>();
    for(int sample=0;sample<96+p.vertices.Length+p.triangles.Length/3;sample++)
    {
     Vector3 direction,target=interior;string kind;
     if(sample<96)
     {double z=1-2*(sample+.5)/96.0,angle=sample*2.399963229728653;float rr=(float)Math.Sqrt(1-z*z);direction=new Vector3(rr*(float)Math.Cos(angle),rr*(float)Math.Sin(angle),(float)z);kind="sphere";}
     else if(sample<96+p.vertices.Length)
     {direction=CompoundUnit(p.vertices[sample-96]-interior);kind="vertex";}
     else
     {int f=(sample-96-p.vertices.Length)*3;Vector3 va=p.vertices[p.triangles[f]],vb=p.vertices[p.triangles[f+1]],vc=p.vertices[p.triangles[f+2]];target=(va+vb+vc)/3;direction=CompoundUnit(Vector3.Cross(vb-va,vc-va));if(Vector3.Dot(direction,target-interior)<0)direction=-direction;kind="face";}
     Vector3 origin=target+direction*(2*radius+1),toward=-direction;
     bool localHit=a.Raycast(new Ray(origin,toward),out RaycastHit lh,4*radius+2);
     Vector3 worldOrigin=world.transform.TransformPoint(origin),worldVector=world.transform.TransformVector(toward);float scale=worldVector.magnitude;
     bool worldHit=b.Raycast(new Ray(worldOrigin,worldVector/scale),out RaycastHit wh,(4*radius+2)*scale);
     rays.Add(new Dictionary<string,object>{{"kind",kind},{"origin_local",PreinsertVector(origin)},{"direction_local",PreinsertVector(toward)},
      {"origin_world",PreinsertVector(worldOrigin)},{"direction_world",PreinsertVector(worldVector/scale)},
      {"local_hit",localHit},{"local_distance",localHit?lh.distance:-1},{"world_hit",worldHit},{"world_distance_m",worldHit?wh.distance:-1},
      {"local_hit_world",PreinsertVector(world.transform.TransformPoint(localHit?lh.point:interior))},{"world_hit_point",PreinsertVector(worldHit?wh.point:worldOrigin)},
      {"local_closest_to_hit",PreinsertVector(a.ClosestPoint(localHit?lh.point:interior))},{"local_hit_point",PreinsertVector(localHit?lh.point:interior)},
      {"world_closest_to_hit",PreinsertVector(b.ClosestPoint(worldHit?wh.point:worldOrigin))}});
    }
    var worldVertices=p.vertices.Select(v=>world.transform.TransformPoint(v)).ToArray();
    Vector3 worldInterior=new Vector3((float)worldVertices.Average(v=>(double)v.x),(float)worldVertices.Average(v=>(double)v.y),(float)worldVertices.Average(v=>(double)v.z));
    float worldRadius=worldVertices.Max(v=>Vector3.Distance(v,worldInterior));
    for(int sample=0;sample<96;sample++)
    {
     double z=1-2*(sample+.5)/96.0,angle=sample*2.399963229728653;float rr=(float)Math.Sqrt(1-z*z);
     Vector3 outward=new Vector3(rr*(float)Math.Cos(angle),rr*(float)Math.Sin(angle),(float)z),wo=worldInterior+outward*(worldRadius+.01f),wd=-outward;
     Vector3 lo=world.transform.InverseTransformPoint(wo),ld=CompoundUnit(world.transform.InverseTransformVector(wd));
     bool localHit=a.Raycast(new Ray(lo,ld),out RaycastHit lh,Vector3.Distance(lo,interior)*3+radius);
     bool worldHit=b.Raycast(new Ray(wo,wd),out RaycastHit wh,2*worldRadius+.02f);
     rays.Add(new Dictionary<string,object>{{"kind","world_sphere"},{"origin_local",PreinsertVector(lo)},{"direction_local",PreinsertVector(ld)},
      {"origin_world",PreinsertVector(wo)},{"direction_world",PreinsertVector(wd)},
      {"local_hit",localHit},{"local_distance",localHit?lh.distance:-1},{"world_hit",worldHit},{"world_distance_m",worldHit?wh.distance:-1},
      {"local_hit_world",PreinsertVector(world.transform.TransformPoint(localHit?lh.point:interior))},{"world_hit_point",PreinsertVector(worldHit?wh.point:wo)},
      {"local_closest_to_hit",PreinsertVector(a.ClosestPoint(localHit?lh.point:interior))},{"local_hit_point",PreinsertVector(localHit?lh.point:interior)},
      {"world_closest_to_hit",PreinsertVector(b.ClosestPoint(worldHit?wh.point:wo))}});
    }
    rows.Add(new Dictionary<string,object>{{"index",p.index},{"link",p.link},{"variant",p.variant},{"vertices_local",p.vertices.Select(PreinsertVector).ToList()},
     {"vertices_world",p.vertices.Select(v=>PreinsertVector(world.transform.TransformPoint(v))).ToList()},
     {"vertex_closest_local",p.vertices.Select(v=>PreinsertVector(a.ClosestPoint(v))).ToList()},
     {"bounds_local_center",PreinsertVector(a.bounds.center)},{"bounds_local_size",PreinsertVector(a.bounds.size)},
     {"bounds_world_center",PreinsertVector(b.bounds.center)},{"bounds_world_size",PreinsertVector(b.bounds.size)},{"rays",rays}});
    DestroyImmediate(unit);DestroyImmediate(world);DestroyImmediate(mesh);
   }
   collectData.AddDataNextStep("sock_compound_probe",new Dictionary<string,object>{{"parts",rows},{"physics_executed",false},{"robot_motion_executed",false}});
  }
  [RFUAPI] public void InstallSockCompoundGeometry(int robotId,string json)
  {
   if(compoundObjects.Count>0)throw new InvalidOperationException("Compound geometry already installed");
   if(!Attrs.TryGetValue(robotId,out BaseAttr robot))throw new ArgumentException("Missing robot");
   var query=JsonUtility.FromJson<CompoundQuery>(json);
   if(query==null || query.parts==null || query.parts.Length==0 || query.cookUnitScale<1 || query.cookUnitScale>1000)throw new ArgumentException("Empty geometry");
   var bodies=robot.GetComponentsInChildren<ArticulationBody>(true).ToDictionary(b=>b.name);
   var old=new Dictionary<string,MeshCollider>();
   foreach(var p in query.parts)
   {
    if(p.link!="right_gripper/gripper_base" && p.link!="right_arm/link_5")throw new ArgumentException("Unsupported replacement link");
    if(p.vertices==null || p.vertices.Length<4 || p.vertices.Length>64 || p.triangles.Length/3>255)throw new ArgumentException("Invalid convex part size");
    if(!bodies.ContainsKey(p.link))throw new ArgumentException("Missing articulation");
    if(!old.ContainsKey(p.link))old[p.link]=bodies[p.link].GetComponentsInChildren<MeshCollider>(true).Single(c=>c.enabled && c.GetComponentInParent<ArticulationBody>()==bodies[p.link]);
   }
   var all=FindObjectsOfType<Collider>().Where(c=>c.enabled).ToArray();
   var created=new List<KeyValuePair<MeshCollider,MeshCollider>>();
   var validatedType=typeof(ObiCollider).Assembly.GetType("Obi.ObiConvexCollisionMesh",true);var register=validatedType.GetMethod("RegisterValidatedConvex",BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic);
   foreach(var p in query.parts)
   {
    var source=old[p.link];var obj=new GameObject("sock_compound_"+compoundObjects.Count);obj.layer=source.gameObject.layer;
    obj.transform.SetParent(bodies[p.link].transform,false);compoundObjects.Add(obj);
    // Cook in unit-size coordinates: PhysX cleaning tolerances can erase
    // small thin convex parts when their vertices are given directly in metres.
    // The inverse uniform transform preserves every physical world vertex.
    var bounds=new Bounds(p.vertices[0],Vector3.zero);foreach(var vertex in p.vertices)bounds.Encapsulate(vertex);
    float extent=Mathf.Max(bounds.size.x,Mathf.Max(bounds.size.y,bounds.size.z));if(extent<1e-8f)throw new ArgumentException("Degenerate part");
    extent/=query.cookUnitScale;obj.transform.localPosition=bounds.center;obj.transform.localScale=Vector3.one*extent;
    var mesh=new Mesh();mesh.name=obj.name;mesh.vertices=p.vertices.Select(v=>(v-bounds.center)/extent).ToArray();mesh.triangles=p.triangles;mesh.RecalculateBounds();
    var cooking=MeshColliderCookingOptions.CookForFasterSimulation|MeshColliderCookingOptions.UseFastMidphase;
    Physics.BakeMesh(mesh.GetInstanceID(),true,cooking);register.Invoke(null,new object[]{mesh});
    var c=obj.AddComponent<MeshCollider>();c.convex=true;c.cookingOptions=cooking;c.sharedMesh=mesh;c.sharedMaterial=source.sharedMaterial;c.contactOffset=source.contactOffset;
    var originalObi=source.GetComponent<ObiCollider>();var obi=obj.AddComponent<ObiCollider>();obi.sourceCollider=c;
    if(originalObi!=null){obi.CollisionMaterial=originalObi.CollisionMaterial;obi.Thickness=originalObi.Thickness;obi.Filter=originalObi.Filter;}
    foreach(var other in all)if(other!=source)Physics.IgnoreCollision(c,other,Physics.GetIgnoreCollision(source,other));
    created.Add(new KeyValuePair<MeshCollider,MeshCollider>(c,source));
   }
   for(int i=0;i<created.Count;i++)for(int j=i+1;j<created.Count;j++)
    Physics.IgnoreCollision(created[i].Key,created[j].Key,created[i].Value==created[j].Value || Physics.GetIgnoreCollision(created[i].Value,created[j].Value));
   foreach(var source in old.Values){var obi=source.GetComponent<ObiCollider>();if(obi!=null)obi.enabled=false;source.enabled=false;}
   Physics.SyncTransforms();
   var rows=new List<object>();var type=typeof(ObiCollider).Assembly.GetType("Obi.ObiConvexCollisionMesh",true);
   var get=type.GetMethod("Get",BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic);
   var release=type.GetMethod("Release",BindingFlags.Static|BindingFlags.Public|BindingFlags.NonPublic);
   foreach(var pair in created)
   {
    var c=pair.Key;var mesh=c.sharedMesh;var obiMesh=(Mesh)get.Invoke(null,new object[]{mesh});
    var probeObject=new GameObject("compound_cook_probe");probeObject.transform.SetPositionAndRotation(c.transform.position,c.transform.rotation);probeObject.transform.localScale=c.transform.lossyScale;
    var probeCollider=probeObject.AddComponent<MeshCollider>();probeCollider.convex=true;probeCollider.cookingOptions=c.cookingOptions;probeCollider.sharedMesh=mesh;Physics.SyncTransforms();
    var probes=new List<object>();var rayProbes=new List<object>();var radialProbes=new List<object>();float vertexError=0;var meshBounds=mesh.bounds;
    foreach(var v in mesh.vertices)
    {
     vertexError=Mathf.Max(vertexError,Vector3.Distance(c.transform.TransformPoint(v),probeCollider.ClosestPoint(c.transform.TransformPoint(v))));
     var direction=(v-meshBounds.center).normalized;var point=c.transform.TransformPoint(v)+c.transform.TransformDirection(direction).normalized*.01f;
     probes.Add(new Dictionary<string,object>{{"query_world",PreinsertVector(point)},{"closest_world",PreinsertVector(c.ClosestPoint(point))},{"explicit_closest_world",PreinsertVector(Physics.ClosestPoint(point,c,c.transform.position,c.transform.rotation))},{"static_closest_world",PreinsertVector(probeCollider.ClosestPoint(point))}});
    }
    var vertices=mesh.vertices;var triangles=mesh.triangles;
    for(int f=0;f<triangles.Length;f+=3)
    {
     Vector3 a=vertices[triangles[f]],b=vertices[triangles[f+1]],d=vertices[triangles[f+2]];
     Vector3 center=c.transform.TransformPoint((a+b+d)/3),normal=c.transform.TransformDirection(Vector3.Cross(b-a,d-a).normalized).normalized;
     bool found=probeCollider.Raycast(new Ray(center+normal*.01f,-normal),out RaycastHit hit,.03f);
     rayProbes.Add(new Dictionary<string,object>{{"hit",found},{"distance_m",found?hit.distance:-1},{"expected_distance_m",.01f}});
    }
    Vector3 interior=Vector3.zero;foreach(var v in vertices)interior+=c.transform.TransformPoint(v);interior/=vertices.Length;
    float radius=0;foreach(var v in vertices)radius=Mathf.Max(radius,Vector3.Distance(interior,c.transform.TransformPoint(v)));
    for(int sample=0;sample<96+vertices.Length;sample++)
    {
     double z=1-2*(Math.Min(sample,95)+.5)/96.0,angle=sample*2.399963229728653;float rr=(float)Math.Sqrt(1-z*z);
     Vector3 direction=sample<96?new Vector3(rr*(float)Math.Cos(angle),rr*(float)Math.Sin(angle),(float)z):(c.transform.TransformPoint(vertices[sample-96])-interior).normalized;
     Vector3 start=interior+direction*(radius+.01f);bool hitFound=probeCollider.Raycast(new Ray(start,-direction),out RaycastHit rayHit,2*radius+.02f);
     radialProbes.Add(new Dictionary<string,object>{{"origin_world",PreinsertVector(start)},{"direction_world",PreinsertVector(-direction)},{"hit",hitFound},{"distance_m",hitFound?rayHit.distance:-1}});
    }
    rows.Add(new Dictionary<string,object>{{"link",c.GetComponentInParent<ArticulationBody>().name},{"vertices_world",mesh.vertices.Select(v=>PreinsertVector(c.transform.TransformPoint(v))).ToList()},
     {"obi_vertices_world",obiMesh.vertices.Select(v=>PreinsertVector(c.transform.TransformPoint(v))).ToList()},{"obi_triangles",obiMesh.triangles.ToList()},
     {"physx_vertex_containment_error_m",vertexError},{"physx_radial_probes",radialProbes},{"physx_face_ray_probes",rayProbes},{"physx_closest_probes",probes},{"convex",c.convex},{"obi_enabled",c.GetComponent<ObiCollider>().enabled}});
    release.Invoke(null,new object[]{mesh,obiMesh});DestroyImmediate(probeObject);
   }
   collectData.AddDataNextStep("sock_compound_geometry",new Dictionary<string,object>{{"parts",rows},{"disabled_originals",old.Keys.ToList()},{"cook_unit_scale",query.cookUnitScale},{"physics_executed",false}});
  }
 }
}
#endif
