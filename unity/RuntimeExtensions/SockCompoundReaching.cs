#if OBI
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using UnityEngine;
using Obi;
namespace RCareWorld.Attributes
{
 public partial class SockClothAttr
 {
  [Serializable] private class CompoundPartQuery {public string link;public Vector3[] vertices;public int[] triangles;}
  [Serializable] private class CompoundQuery {public CompoundPartQuery[] parts;public float cookUnitScale=100f;}
  private readonly List<GameObject> compoundObjects=new List<GameObject>();
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
