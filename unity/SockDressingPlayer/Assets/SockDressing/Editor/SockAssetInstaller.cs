using System;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace SockDressing.Editor
{
    public static class SockAssetInstaller
    {
        private const string Destination = "Assets/SockDressing/Generated";

        [MenuItem("Sock Dressing/Sync Generated Assets")]
        public static void SyncGeneratedAssets()
        {
            string project = Directory.GetParent(Application.dataPath).FullName;
            string packageRoot = Path.GetFullPath(Path.Combine(project, "..", ".."));
            string source = Path.Combine(packageRoot, "assets", "generated");
            if (!Directory.Exists(source))
                throw new DirectoryNotFoundException(
                    $"Generate Python assets first: {source}");
            string destination = Path.Combine(project, Destination);
            CopyDirectory(source, destination);
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Debug.Log($"Synchronized generated assets to {Destination}");
        }

        [MenuItem("Sock Dressing/Add Obi Colliders To Stable Regions")]
        public static void AddObiColliders()
        {
            Type obiCollider = Type.GetType("Obi.ObiCollider, Obi");
            if (obiCollider == null)
                throw new InvalidOperationException(
                    "Licensed Obi package is not loaded in this project.");
            int changed = 0;
            foreach (StableObjectId item in UnityEngine.Object.FindObjectsOfType<StableObjectId>())
            {
                if (!item.obiColliderExpected || item.GetComponent(obiCollider) != null)
                    continue;
                Undo.AddComponent(item.gameObject, obiCollider);
                changed++;
            }
            Debug.Log($"Added Obi colliders to {changed} stable regions.");
        }

        private static void CopyDirectory(string source, string destination)
        {
            Directory.CreateDirectory(destination);
            foreach (string file in Directory.GetFiles(source))
            {
                if (file.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                    continue;
                File.Copy(file, Path.Combine(destination, Path.GetFileName(file)), true);
            }
            foreach (string directory in Directory.GetDirectories(source))
                CopyDirectory(
                    directory,
                    Path.Combine(destination, Path.GetFileName(directory)));
        }
    }
}
