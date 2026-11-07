#if OBI
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace RCareWorld.Attributes
{
    public partial class SockClothAttr
    {
        [Serializable] private class LearningLightSpec
        {
            public float[] rotation;
            public float intensity;
        }
        [Serializable] private class LearningLightingSpec
        {
            public string profile;
            public bool correct_normals;
            public float ambient = .2f;
            public float sun_intensity = .7f;
            public float sun_shadow_strength = .25f;
            public float[] sun_rotation;
            public float shadow_distance = 20;
            public LearningLightSpec[] lights;
        }
        private readonly List<Light> learningLights = new List<Light>();
        private Light[] learningOriginalLights;
        private float[] learningOriginalIntensity, learningOriginalShadowStrength;
        private Quaternion[] learningOriginalRotation;
        private LightShadows[] learningOriginalShadows;
        private AmbientMode learningOriginalAmbientMode;
        private Color learningOriginalAmbient;
        private SphericalHarmonicsL2 learningOriginalProbe;
        private bool learningNormalsSeparated;

        [RFUAPI] public void CaptureSockLearningVisibility(int cameraId,int width,int height,float fov)
        {
            if(!Attrs.TryGetValue(cameraId,out BaseAttr attr) || !(attr is CameraAttr camera))
                throw new ArgumentException("Missing head camera");
            var renderer = sockVisualFilter.GetComponent<MeshRenderer>();
            var block = new MaterialPropertyBlock();renderer.GetPropertyBlock(block);
            block.SetColor("_IDColor",RCareWorldUtility.EncodeIDAsColor(ID));renderer.SetPropertyBlock(block);
            // ID replacement rendering retains real occlusion, excludes cast
            // shadows, and never modifies training RGB or physical geometry.
            var image = camera.GetID(width,height,fov);
            collectData.AddDataNextStep("sock_learning_visibility",image.EncodeToPNG());
        }
        private static Vector3 LearningVector(float[] values, Vector3 fallback)
            => values != null && values.Length == 3 ? new Vector3(values[0],values[1],values[2]) : fallback;

        // The two visual submeshes occupy exactly the same surface. Separate
        // their normal accumulators; particle positions, triangles, colliders
        // and the original front-face indices remain unchanged.
        private void SeparateLearningVisualNormals()
        {
            if (learningNormalsSeparated || sockVisualMesh == null) return;
            var vertices = sockVisualMesh.vertices;
            var uv = sockVisualMesh.uv;
            var front = sockVisualMesh.GetTriangles(0);
            var back = sockVisualMesh.GetTriangles(1);
            int count = vertices.Length;
            sockVisualMesh.vertices = vertices.Concat(vertices).ToArray();
            if(uv.Length == count) sockVisualMesh.uv = uv.Concat(uv).ToArray();
            sockVisualVertexParticles = sockVisualVertexParticles.Concat(sockVisualVertexParticles).ToArray();
            sockVisualMesh.SetTriangles(front,0);
            sockVisualMesh.SetTriangles(back.Select(index=>index+count).ToArray(),1);
            sockVisualMesh.RecalculateNormals();
            sockVisualMesh.RecalculateBounds();
            learningNormalsSeparated = true;
        }

        [RFUAPI] public void ConfigureSockLearningLighting(string json)
        {
            var spec = JsonUtility.FromJson<LearningLightingSpec>(json);
            if(learningOriginalLights == null)
            {
                learningOriginalLights = FindObjectsOfType<Light>();
                learningOriginalIntensity = learningOriginalLights.Select(light=>light.intensity).ToArray();
                learningOriginalShadowStrength = learningOriginalLights.Select(light=>light.shadowStrength).ToArray();
                learningOriginalRotation = learningOriginalLights.Select(light=>light.transform.rotation).ToArray();
                learningOriginalShadows = learningOriginalLights.Select(light=>light.shadows).ToArray();
                learningOriginalAmbientMode = RenderSettings.ambientMode;
                learningOriginalAmbient = RenderSettings.ambientLight;
                learningOriginalProbe = RenderSettings.ambientProbe;
            }
            foreach(var light in learningLights) { light.gameObject.SetActive(false);Destroy(light.gameObject); }
            learningLights.Clear();
            bool baseline = spec.profile == "baseline" || spec.profile == "normals_only";
            for(int i=0;i<learningOriginalLights.Length;i++)
            {
                var light = learningOriginalLights[i];
                light.intensity = baseline ? learningOriginalIntensity[i] : spec.sun_intensity;
                light.shadowStrength = baseline ? learningOriginalShadowStrength[i] : spec.sun_shadow_strength;
                light.shadows = baseline ? learningOriginalShadows[i] : LightShadows.Soft;
                light.transform.rotation = baseline ? learningOriginalRotation[i] : Quaternion.Euler(LearningVector(spec.sun_rotation,new Vector3(55,180,0)));
            }
            if(baseline)
            {
                RenderSettings.ambientMode = learningOriginalAmbientMode;
                RenderSettings.ambientLight = learningOriginalAmbient;
                RenderSettings.ambientProbe = learningOriginalProbe;
            }
            else
            {
                RenderSettings.ambientMode = AmbientMode.Flat;
                RenderSettings.ambientLight = Color.white*spec.ambient;
                var probe = new SphericalHarmonicsL2();probe.AddAmbientLight(Color.white*spec.ambient);
                RenderSettings.ambientProbe = probe;
            }
            QualitySettings.shadowDistance = spec.shadow_distance;
            QualitySettings.pixelLightCount = Math.Max(QualitySettings.pixelLightCount,4);
            if(spec.correct_normals) SeparateLearningVisualNormals();
            if(spec.lights != null) foreach(var entry in spec.lights)
            {
                var obj = new GameObject("sock_learning_fill");
                obj.transform.rotation = Quaternion.Euler(LearningVector(entry.rotation,Vector3.zero));
                var light = obj.AddComponent<Light>();light.type = LightType.Directional;
                light.color = Color.white;light.intensity = entry.intensity;light.shadows = LightShadows.None;
                learningLights.Add(light);
            }
            int zeroNormals = sockVisualMesh.normals.Count(normal=>normal.sqrMagnitude<1e-8f);
            collectData.AddDataNextStep("sock_learning_lighting",new Dictionary<string,object>{
                {"profile",spec.profile},{"normals_separated",learningNormalsSeparated},
                {"visual_vertices",sockVisualMesh.vertexCount},{"zero_normals",zeroNormals},
                {"physical_particles",cloth.particleCount},{"ambient",spec.ambient},
                {"sun_intensity",spec.sun_intensity},{"shadow_strength",spec.sun_shadow_strength},
                {"fill_lights",learningLights.Count},{"rendering_only",true}});
        }
    }
}
#endif
