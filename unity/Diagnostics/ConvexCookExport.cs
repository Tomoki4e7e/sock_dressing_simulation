#if UNITY_EDITOR
using System;
using System.IO;
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

// Export Unity's serialized cooked bytes; sharedMesh.vertices is only input.
public static class ConvexCookExport
{
 [Serializable] public sealed class Part {public int index;public string variant,cookingMode;public Vector3[] vertices;public int[] triangles;public bool primitiveBox,skipBake;}
 [Serializable] public sealed class Query {public Part[] parts;}
 public static void Export()
 {
  var query=JsonUtility.FromJson<Query>(File.ReadAllText(Environment.GetEnvironmentVariable("SOCK_COOK_PAYLOAD")));
  var output=Environment.GetEnvironmentVariable("SOCK_COOK_OUTPUT");Directory.CreateDirectory(output);
  Directory.CreateDirectory("Assets/Cooked");EditorSettings.serializationMode=SerializationMode.ForceText;
  var assets=new List<string>();
  foreach(var part in query.parts)
  {
   if(part.primitiveBox)continue;
   var mesh=new Mesh{name="cook_"+part.index+"_"+part.variant};mesh.vertices=part.vertices;mesh.triangles=part.triangles;mesh.RecalculateBounds();
   var flags=part.cookingMode=="none"?MeshColliderCookingOptions.None:MeshColliderCookingOptions.CookForFasterSimulation|MeshColliderCookingOptions.UseFastMidphase;
   if(part.cookingMode=="default")flags|=MeshColliderCookingOptions.EnableMeshCleaning|MeshColliderCookingOptions.WeldColocatedVertices;
   Physics.BakeMesh(mesh.GetInstanceID(),true,flags);
   var path="Assets/Cooked/"+mesh.name+".asset";AssetDatabase.DeleteAsset(path);AssetDatabase.CreateAsset(mesh,path);EditorUtility.SetDirty(mesh);AssetDatabase.SaveAssets();
   File.Copy(path,Path.Combine(output,mesh.name+".asset"),true);
   var obj=new GameObject(mesh.name);var collider=obj.AddComponent<MeshCollider>();collider.convex=true;collider.cookingOptions=flags;collider.sharedMesh=mesh;
   var prefab="Assets/Cooked/"+mesh.name+".prefab";PrefabUtility.SaveAsPrefabAsset(obj,prefab);UnityEngine.Object.DestroyImmediate(obj);assets.Add(prefab);
  }
  AssetDatabase.SaveAssets();PlayerSettings.bakeCollisionMeshes=true;
  BuildPipeline.BuildAssetBundles(output,new[]{new AssetBundleBuild{assetBundleName="cooked-convex",assetNames=assets.ToArray()}},BuildAssetBundleOptions.UncompressedAssetBundle|BuildAssetBundleOptions.ForceRebuildAssetBundle,BuildTarget.StandaloneLinux64);
 }
}
#endif
