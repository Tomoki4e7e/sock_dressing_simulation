using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

// Standalone reproduction: no RCareWorld, Obi, robot, or imported assets.
public sealed class ConvexContactRepro : MonoBehaviour
{
 [Serializable] public sealed class Point {public Vector3 center;public float radius,expected;public int sample;public string kind;}
 [Serializable] public sealed class Shape {public string variant;public bool primitiveBox,builtin,overrideRotation;public Vector3[] vertices;public int[] triangles;public Vector3 scale;public Quaternion rotation;public Point[] points;}
 [Serializable] public sealed class Query {public Shape[] parts;}
 private string Argument(string name){var a=Environment.GetCommandLineArgs();return a[Array.IndexOf(a,name)+1];}
 private void Start()
 {
  string output=Argument("--output");var rows=new List<Dictionary<string,object>>();
  try
  {
   Physics.simulationMode=SimulationMode.Script;Physics.autoSyncTransforms=true;
   bool local=Environment.GetCommandLineArgs().Contains("--local-scene"),queriesOnly=Environment.GetCommandLineArgs().Contains("--queries-only");var physicsScene=Physics.defaultPhysicsScene;UnityEngine.SceneManagement.Scene scene=default;
   if(local){scene=UnityEngine.SceneManagement.SceneManager.CreateScene("isolated_repro",new UnityEngine.SceneManagement.CreateSceneParameters(UnityEngine.SceneManagement.LocalPhysicsMode.Physics3D));physicsScene=scene.GetPhysicsScene();}
   var query=JsonUtility.FromJson<Query>(File.ReadAllText(Argument("--payload")));
   if(Environment.GetCommandLineArgs().Contains("--replay")){Replay(query,physicsScene,scene,local,rows);Write(output,rows);Application.Quit(0);return;}
   foreach(var shape in query.parts)foreach(bool rotated in new[]{false,true})
   {
    var target=new GameObject("target");if(local)UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(target,scene);target.transform.rotation=shape.overrideRotation?shape.rotation:rotated?Quaternion.Euler(27,13,-19):Quaternion.identity;target.transform.localScale=shape.scale;
    var mesh=new Mesh();mesh.vertices=shape.vertices;mesh.triangles=shape.triangles;mesh.RecalculateBounds();
    if(shape.builtin)
    {
     var primitive=GameObject.CreatePrimitive(PrimitiveType.Cube);var stock=primitive.GetComponent<MeshFilter>().sharedMesh;var size=mesh.bounds.size;var center=mesh.bounds.center;
     mesh.vertices=stock.vertices.Select(v=>Vector3.Scale(v,size)+center).ToArray();mesh.triangles=stock.triangles;mesh.RecalculateBounds();DestroyImmediate(primitive);
    }
    Collider collider;
    if(shape.primitiveBox){var box=target.AddComponent<BoxCollider>();box.center=mesh.bounds.center;box.size=mesh.bounds.size;collider=box;}
    else{var cooking=MeshColliderCookingOptions.CookForFasterSimulation|MeshColliderCookingOptions.UseFastMidphase;Physics.BakeMesh(mesh.GetInstanceID(),true,cooking);var m=target.AddComponent<MeshCollider>();m.convex=true;m.cookingOptions=cooking;m.sharedMesh=mesh;collider=m;}
    collider.contactOffset=.00001f;Physics.SyncTransforms();Vector3 centerWorld=target.transform.TransformPoint(mesh.bounds.center);
    for(int vertex=0;vertex<shape.vertices.Length;vertex++)
    {
     Vector3 v=target.transform.TransformPoint(shape.vertices[vertex]);var normals=new List<Vector3>();Vector3 face=Vector3.zero;float best=-1;
     for(int f=0;f<shape.triangles.Length;f+=3)
     {
      if(shape.triangles[f]!=vertex&&shape.triangles[f+1]!=vertex&&shape.triangles[f+2]!=vertex)continue;
      Vector3 a=shape.vertices[shape.triangles[f]],b=shape.vertices[shape.triangles[f+1]],c=shape.vertices[shape.triangles[f+2]];
      Vector3 cross=Vector3.Cross(target.transform.TransformVector(b-a),target.transform.TransformVector(c-a)),normal=cross/cross.magnitude;
      if(Vector3.Dot(normal,v-centerWorld)<0)normal=-normal;
      if(!normals.Any(n=>Vector3.Dot(n,normal)>.99999f))normals.Add(normal);
      if(cross.sqrMagnitude>best){best=cross.sqrMagnitude;face=normal;}
     }
     Vector3 bisector=Vector3.zero;foreach(var n in normals)bisector+=n;bisector.Normalize();
     foreach(bool cone in new[]{false,true})
     {
      Vector3 normal=cone?bisector:face;var ball=new GameObject("ball");if(local)UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(ball,scene);var sphere=ball.AddComponent<SphereCollider>();sphere.radius=.001f;sphere.contactOffset=.00001f;var recorder=ball.AddComponent<ConvexContactReproRecorder>();var body=ball.AddComponent<Rigidbody>();body.useGravity=false;body.sleepThreshold=0;body.maxDepenetrationVelocity=.1f;
      for(int step=0;step<15;step++)
      {
       float offset=.0015f-step*.00005f;Vector3 point=v+normal*offset;body.position=point;body.velocity=Vector3.zero;body.angularVelocity=Vector3.zero;body.WakeUp();recorder.contacts.Clear();Physics.SyncTransforms();
       var buffer=new Collider[8];int count=physicsScene.OverlapSphere(point,.001f,buffer,~0,QueryTriggerInteraction.Collide);bool overlap=buffer.Take(count).Contains(collider);bool mtd=Physics.ComputePenetration(sphere,point,Quaternion.identity,collider,target.transform.position,target.transform.rotation,out Vector3 direction,out float depth);
       if(!queriesOnly)physicsScene.Simulate(1f/240f);float measured=recorder.contacts.Count==0?0:Mathf.Max(0,-recorder.contacts.Min(c=>c.separation));
       rows.Add(new Dictionary<string,object>{{"variant",shape.variant},{"rotated",rotated},{"local_scene",local},{"queries_only",queriesOnly},{"direction",cone?"normal_cone_interior":"single_face_normal"},{"vertex",vertex},{"step",step},{"expected_depth_m",.001f-offset},{"scene_overlap",overlap},{"mtd_overlap",mtd},{"mtd_depth_m",depth},{"contact_count",recorder.contacts.Count},{"depth_m",measured},{"movement_m",Vector3.Distance(point,body.position)}});
      }
      DestroyImmediate(ball);
     }
    }
    DestroyImmediate(target);DestroyImmediate(mesh);
   }
   Write(output,rows);Application.Quit(0);
  }
  catch(Exception error){File.WriteAllText(output,"{\"error\":\""+error.ToString().Replace("\\","\\\\").Replace("\"","\\\"").Replace("\n","\\n").Replace("\r","")+"\"}");Application.Quit(2);}
 }
 private void Replay(Query query,PhysicsScene physicsScene,UnityEngine.SceneManagement.Scene scene,bool local,List<Dictionary<string,object>> rows)
 {
  foreach(var shape in query.parts)
  {
   var target=new GameObject("target");if(local)UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(target,scene);target.transform.SetPositionAndRotation(Vector3.zero,shape.rotation);target.transform.localScale=shape.scale;
   var mesh=new Mesh();mesh.vertices=shape.vertices;mesh.triangles=shape.triangles;mesh.RecalculateBounds();Collider collider;
   if(shape.primitiveBox){var box=target.AddComponent<BoxCollider>();box.size=mesh.bounds.size;box.center=mesh.bounds.center;collider=box;}
   else{var cooking=MeshColliderCookingOptions.CookForFasterSimulation|MeshColliderCookingOptions.UseFastMidphase;Physics.BakeMesh(mesh.GetInstanceID(),true,cooking);var c=target.AddComponent<MeshCollider>();c.convex=true;c.cookingOptions=cooking;c.sharedMesh=mesh;collider=c;}
   collider.contactOffset=.00001f;
   foreach(var point in shape.points)
   {
    var ball=new GameObject("ball");if(local)UnityEngine.SceneManagement.SceneManager.MoveGameObjectToScene(ball,scene);ball.transform.position=point.center;var sphere=ball.AddComponent<SphereCollider>();sphere.radius=point.radius;sphere.contactOffset=.00001f;var recorder=ball.AddComponent<ConvexContactReproRecorder>();var body=ball.AddComponent<Rigidbody>();body.useGravity=false;body.sleepThreshold=0;body.maxDepenetrationVelocity=.1f;body.position=point.center;body.WakeUp();Physics.SyncTransforms();
    var buffer=new Collider[8];int count=physicsScene.OverlapSphere(point.center,point.radius,buffer,~0,QueryTriggerInteraction.Collide);bool overlap=buffer.Take(count).Contains(collider);bool mtd=Physics.ComputePenetration(sphere,point.center,Quaternion.identity,collider,target.transform.position,target.transform.rotation,out Vector3 direction,out float depth);physicsScene.Simulate(1f/240f);
    rows.Add(new Dictionary<string,object>{{"variant",shape.variant},{"local_scene",local},{"sample",point.sample},{"kind",point.kind},{"expected_depth_m",point.expected},{"scene_overlap",overlap},{"mtd_overlap",mtd},{"mtd_depth_m",depth},{"contact_count",recorder.contacts.Count},{"depth_m",recorder.contacts.Count==0?0:Mathf.Max(0,-recorder.contacts.Min(c=>c.separation))},{"movement_m",Vector3.Distance(point.center,body.position)}});DestroyImmediate(ball);
   }
   DestroyImmediate(target);DestroyImmediate(mesh);
  }
 }
 private void Write(string output,List<Dictionary<string,object>> rows)
 {
   // Simple JSON writer to avoid any package or third-party dependency.
   var text=new System.Text.StringBuilder();text.Append("{\"unity_version\":\"").Append(Application.unityVersion).Append("\",\"rows\":[");bool first=true;
   foreach(var row in rows){if(!first)text.Append(',');first=false;text.Append('{');bool entry=true;foreach(var pair in row){if(!entry)text.Append(',');entry=false;text.Append('"').Append(pair.Key).Append("\":");if(pair.Value is string)text.Append('"').Append(pair.Value).Append('"');else if(pair.Value is bool)text.Append((bool)pair.Value?"true":"false");else text.Append(Convert.ToString(pair.Value,System.Globalization.CultureInfo.InvariantCulture));}text.Append('}');}text.Append("]}");File.WriteAllText(output,text.ToString());
 }
}
public sealed class ConvexContactReproRecorder:MonoBehaviour
{
 public readonly List<ContactPoint> contacts=new List<ContactPoint>();
 private void OnCollisionEnter(Collision collision){contacts.AddRange(collision.contacts);}
 private void OnCollisionStay(Collision collision){contacts.AddRange(collision.contacts);}
}
#if UNITY_EDITOR
public static class ConvexContactReproBuild
{
 public static void Build()
 {
  var scene=UnityEditor.SceneManagement.EditorSceneManager.NewScene(UnityEditor.SceneManagement.NewSceneSetup.EmptyScene,UnityEditor.SceneManagement.NewSceneMode.Single);new GameObject("repro").AddComponent<ConvexContactRepro>();UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene,"Assets/Repro.unity");
  var options=Environment.GetEnvironmentVariable("SOCK_REPRO_DEVELOPMENT")=="1"?UnityEditor.BuildOptions.Development|UnityEditor.BuildOptions.AllowDebugging:UnityEditor.BuildOptions.None;
  var result=UnityEditor.BuildPipeline.BuildPlayer(new[]{"Assets/Repro.unity"},Environment.GetEnvironmentVariable("SOCK_REPRO_PLAYER"),UnityEditor.BuildTarget.StandaloneLinux64,options);
  if(result.summary.result!=UnityEditor.Build.Reporting.BuildResult.Succeeded)throw new Exception("Minimal player build failed");
 }
}
#endif
