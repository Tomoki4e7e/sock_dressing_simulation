using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace SockDressing.Editor
{
    public static class SockPlayerBuild
    {
        private const string ScenePath = "Assets/SockDressing/Scenes/SockDressing.unity";

        [MenuItem("Sock Dressing/Build Linux Development Player")]
        public static void BuildDevelopment()
        {
            Build(true);
        }

        [MenuItem("Sock Dressing/Build Linux Release Player")]
        public static void BuildRelease()
        {
            Build(false);
        }

        public static void BuildFromCommandLine()
        {
            bool development = Environment.GetCommandLineArgs().Contains("-sockDevelopment");
            Build(development);
        }

        private static void Build(bool development)
        {
            RequireEditorVersion();
            bool obiPresent = AppDomain.CurrentDomain.GetAssemblies()
                .Any(assembly => assembly.GetName().Name == "Obi");
            bool allowStub = Environment.GetEnvironmentVariable("SOCK_ALLOW_STUB_BUILD") == "1";
            if (!obiPresent && !allowStub)
                throw new BuildFailedException(
                    "Licensed Obi package is missing. Import it and enable SOCKDRESSING_OBI. " +
                    "Set SOCK_ALLOW_STUB_BUILD=1 only for communication scaffold tests.");

            SockAssetInstaller.SyncGeneratedAssets();
            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            if (AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath) == null)
            {
                Scene scene = EditorSceneManager.NewScene(
                    NewSceneSetup.EmptyScene, NewSceneMode.Single);
                EditorSceneManager.SaveScene(scene, ScenePath);
            }

            string project = Directory.GetParent(Application.dataPath).FullName;
            string root = Path.GetFullPath(Path.Combine(project, "..", ".."));
            string output = Path.Combine(root, "Build", "SockDressingPlayer", "Player.x86_64");
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            BuildOptions options = development
                ? BuildOptions.Development | BuildOptions.AllowDebugging
                : BuildOptions.None;
            BuildReport report = BuildPipeline.BuildPlayer(
                new[] { ScenePath },
                output,
                BuildTarget.StandaloneLinux64,
                options);
            if (report.summary.result != BuildResult.Succeeded)
                throw new BuildFailedException(
                    $"Linux player build failed: {report.summary.result}");
            Debug.Log($"Built SockDressingPlayer at {output}");
        }

        private static void RequireEditorVersion()
        {
            if (!Application.unityVersion.StartsWith("2022.3.34f1", StringComparison.Ordinal))
                throw new BuildFailedException(
                    $"Unity 2022.3.34f1 is required, got {Application.unityVersion}");
        }
    }
}
