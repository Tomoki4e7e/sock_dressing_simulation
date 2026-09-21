using System;
using UnityEngine;

namespace SockDressing
{
    public readonly struct MaskCoverage
    {
        public MaskCoverage(
            bool valid,
            float coverage,
            int sockPixels,
            int legPixels,
            int overlapPixels,
            string reason)
        {
            Valid = valid;
            Coverage = coverage;
            SockPixels = sockPixels;
            LegPixels = legPixels;
            OverlapPixels = overlapPixels;
            Reason = reason;
        }

        public bool Valid { get; }
        public float Coverage { get; }
        public int SockPixels { get; }
        public int LegPixels { get; }
        public int OverlapPixels { get; }
        public string Reason { get; }
    }

    public static class SynchronizedMaskCapture
    {
        public static byte[] CaptureRgbPng(Camera camera, int width, int height)
        {
            return CaptureCameraPng(camera, width, height, null);
        }

        public static byte[] CaptureDepthPng(Camera camera, int width, int height)
        {
            Shader shader = Shader.Find("Hidden/SockLinearDepth");
            if (shader == null)
                return Array.Empty<byte>();
            return CaptureCameraPng(camera, width, height, shader);
        }

        public static byte[] CaptureAmodalMaskPng(
            Camera camera,
            Renderer target,
            int width,
            int height)
        {
            return CaptureAmodalMaskPng(camera, new[] { target }, width, height);
        }

        public static byte[] CaptureAmodalMaskPng(
            Camera camera,
            Renderer[] targets,
            int width,
            int height)
        {
            bool[] mask = RenderBinary(camera, targets, width, height);
            var texture = new Texture2D(width, height, TextureFormat.RGB24, false);
            var colors = new Color32[mask.Length];
            for (int i = 0; i < mask.Length; i++)
                colors[i] = mask[i] ? Color.white : Color.black;
            texture.SetPixels32(colors);
            texture.Apply(false);
            byte[] png = texture.EncodeToPNG();
            UnityEngine.Object.Destroy(texture);
            return png;
        }

        public static byte[] CaptureInstanceMaskPng(Camera camera, int width, int height)
        {
            StableObjectId[] objects = UnityEngine.Object.FindObjectsOfType<StableObjectId>();
            var renderers = new Renderer[objects.Length];
            var original = new Material[objects.Length][];
            var replacements = new Material[objects.Length];
            Shader shader = Shader.Find("Unlit/Color");
            if (shader == null)
                return Array.Empty<byte>();
            CameraClearFlags clearFlags = camera.clearFlags;
            Color background = camera.backgroundColor;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = Color.black;
            for (int i = 0; i < objects.Length; i++)
            {
                renderers[i] = objects[i].GetComponent<Renderer>();
                if (renderers[i] == null)
                    continue;
                original[i] = renderers[i].sharedMaterials;
                int id = objects[i].objectId;
                replacements[i] = new Material(shader)
                {
                    color = new Color32(
                        (byte)(id & 255),
                        (byte)((id >> 8) & 255),
                        (byte)((id >> 16) & 255),
                        255)
                };
                renderers[i].sharedMaterial = replacements[i];
            }
            try
            {
                return CaptureCameraPng(camera, width, height, null);
            }
            finally
            {
                camera.clearFlags = clearFlags;
                camera.backgroundColor = background;
                for (int i = 0; i < renderers.Length; i++)
                {
                    if (renderers[i] != null)
                        renderers[i].sharedMaterials = original[i];
                    if (replacements[i] != null)
                        UnityEngine.Object.Destroy(replacements[i]);
                }
            }
        }

        public static MaskCoverage Measure(
            Camera camera,
            Renderer sock,
            Renderer[] leg,
            int width,
            int height)
        {
            bool[] sockMask = RenderBinary(camera, new[] { sock }, width, height);
            bool[] legMask = RenderBinary(camera, leg, width, height);
            int sockPixels = 0;
            int legPixels = 0;
            int overlap = 0;
            for (int i = 0; i < sockMask.Length; i++)
            {
                if (sockMask[i])
                    sockPixels++;
                if (legMask[i])
                    legPixels++;
                if (sockMask[i] && legMask[i])
                    overlap++;
            }
            int total = width * height;
            if (sockPixels == 0 || legPixels == 0)
                return new MaskCoverage(
                    false, 0, sockPixels, legPixels, overlap, "empty semantic mask");
            if (sockPixels == total || legPixels == total)
                return new MaskCoverage(
                    false, 0, sockPixels, legPixels, overlap, "full-frame semantic mask");
            return new MaskCoverage(
                true,
                (float)overlap / legPixels,
                sockPixels,
                legPixels,
                overlap,
                string.Empty);
        }

        private static bool[] RenderBinary(
            Camera camera,
            Renderer[] targets,
            int width,
            int height)
        {
            Renderer[] renderers = UnityEngine.Object.FindObjectsOfType<Renderer>();
            var targetSet = new System.Collections.Generic.HashSet<Renderer>(targets);
            var enabled = new bool[renderers.Length];
            for (int i = 0; i < renderers.Length; i++)
            {
                enabled[i] = renderers[i].enabled;
                renderers[i].enabled = targetSet.Contains(renderers[i]);
            }
            RenderTexture previousTarget = camera.targetTexture;
            RenderTexture previousActive = RenderTexture.active;
            Color previousBackground = camera.backgroundColor;
            CameraClearFlags previousFlags = camera.clearFlags;
            var texture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            var readable = new Texture2D(width, height, TextureFormat.RGB24, false);
            try
            {
                camera.targetTexture = texture;
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.backgroundColor = Color.black;
                camera.Render();
                RenderTexture.active = texture;
                readable.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                readable.Apply(false);
                Color32[] pixels = readable.GetPixels32();
                var mask = new bool[pixels.Length];
                for (int i = 0; i < pixels.Length; i++)
                    mask[i] = pixels[i].r != 0 || pixels[i].g != 0 || pixels[i].b != 0;
                return mask;
            }
            finally
            {
                camera.targetTexture = previousTarget;
                camera.backgroundColor = previousBackground;
                camera.clearFlags = previousFlags;
                RenderTexture.active = previousActive;
                for (int i = 0; i < renderers.Length; i++)
                    renderers[i].enabled = enabled[i];
                UnityEngine.Object.Destroy(texture);
                UnityEngine.Object.Destroy(readable);
            }
        }

        private static byte[] CaptureCameraPng(
            Camera camera,
            int width,
            int height,
            Shader replacement)
        {
            RenderTexture previousTarget = camera.targetTexture;
            RenderTexture previousActive = RenderTexture.active;
            var texture = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            var readable = new Texture2D(width, height, TextureFormat.RGB24, false);
            try
            {
                camera.targetTexture = texture;
                if (replacement != null)
                    camera.SetReplacementShader(replacement, string.Empty);
                camera.Render();
                RenderTexture.active = texture;
                readable.ReadPixels(new Rect(0, 0, width, height), 0, 0);
                readable.Apply(false);
                return readable.EncodeToPNG();
            }
            finally
            {
                camera.ResetReplacementShader();
                camera.targetTexture = previousTarget;
                RenderTexture.active = previousActive;
                UnityEngine.Object.Destroy(texture);
                UnityEngine.Object.Destroy(readable);
            }
        }
    }
}
