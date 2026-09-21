using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace SockDressing
{
    public sealed class SockClothRuntime : MonoBehaviour
    {
        public const string ProtocolVersion = "sock-cloth-v1";

        [Header("Stable object IDs")]
        public int sockId = 1200;
        public int robotId = 1100;
        public int humanId = 2000;
        public int rgbCameraId = 1300;
        public int depthCameraId = 1302;
        public int instanceCameraId = 1303;

        [Header("Cloth contract")]
        public float stretchCompliance = 0.0005f;
        public float bendCompliance = 0.005f;
        public float maximumCircumferentialStretch = 1.5f;
        public float particleRadius = 0.008f;
        public float particleMass = 0.005f;
        public float collisionMargin = 0.002f;
        public float friction = 0.5f;
        public int substeps = 4;
        public int solverIterations = 8;
        public bool selfCollision = true;

        [Header("Scene references")]
        public Transform leftGripper;
        public Transform rightGripper;
        public Camera rgbCamera;
        public Camera maskCamera;
        public Renderer sockRenderer;
        public Renderer legRenderer;
        public Renderer[] legRenderers = Array.Empty<Renderer>();
        public MonoBehaviour obiAdapter;

        private ISockPhysicsBackend backend;
        private Vector3[] resetParticles = Array.Empty<Vector3>();
        private Quaternion resetRotation;
        private Vector3 resetPosition;
        private Vector3 leftResetPosition;
        private Quaternion leftResetRotation;
        private Vector3 rightResetPosition;
        private Quaternion rightResetRotation;

        public ISockPhysicsBackend Backend => backend;

        private void Awake()
        {
            Time.fixedDeltaTime = 0.02f;
            backend = obiAdapter as ISockPhysicsBackend;
            if (backend == null)
                backend = new UnavailableObiBackend();
            resetParticles = backend.Particles();
            resetPosition = transform.position;
            resetRotation = transform.rotation;
        }

        private void Start()
        {
            resetParticles = backend.Particles();
            resetPosition = transform.position;
            resetRotation = transform.rotation;
            if (leftGripper != null)
            {
                leftResetPosition = leftGripper.position;
                leftResetRotation = leftGripper.rotation;
            }
            if (rightGripper != null)
            {
                rightResetPosition = rightGripper.position;
                rightResetRotation = rightGripper.rotation;
            }
        }

        public Dictionary<string, object> Snapshot()
        {
            var snapshot = new Dictionary<string, object>
            {
                ["name"] = name,
                ["protocol_version"] = ProtocolVersion,
                ["particles"] = Vectors(backend.Particles()),
                ["particle_velocities"] = Vectors(backend.Velocities()),
                ["cloth_configuration"] = Configuration(),
                ["registered_obi_colliders"] = backend.RegisteredColliders(),
                ["grasp_state"] = new List<object>
                {
                    GraspMapping("left"), GraspMapping("right")
                },
                ["cloth_contacts"] = backend.Contacts(),
                ["coverage_observations"] = Coverage()
            };
            Renderer[] legs = ActiveLegRenderers();
            if (rgbCamera != null && maskCamera != null &&
                sockRenderer != null && legs.Length > 0)
            {
                const int width = 1280;
                const int height = 960;
                snapshot["rgb_png"] =
                    SynchronizedMaskCapture.CaptureRgbPng(rgbCamera, width, height);
                snapshot["depth_png"] =
                    SynchronizedMaskCapture.CaptureDepthPng(rgbCamera, width, height);
                snapshot["sock_mask_png"] =
                    SynchronizedMaskCapture.CaptureAmodalMaskPng(
                        maskCamera, sockRenderer, width, height);
                snapshot["leg_mask_png"] =
                    SynchronizedMaskCapture.CaptureAmodalMaskPng(
                        maskCamera, legs, width, height);
                snapshot["instance_mask_png"] =
                    SynchronizedMaskCapture.CaptureInstanceMaskPng(
                        maskCamera, width, height);
                snapshot["camera_frame"] = Time.frameCount;
            }
            return snapshot;
        }

        public Dictionary<string, object> Configuration()
        {
            return new Dictionary<string, object>
            {
                ["protocol_version"] = ProtocolVersion,
                ["obi_available"] = backend.Available,
                ["obi_version"] = backend.Version,
                ["settings_source"] = "SockClothRuntime serialized startup values",
                ["timestep_s"] = Time.fixedDeltaTime,
                ["substeps"] = substeps,
                ["solver_iterations"] = solverIterations,
                ["particle_radius_m"] = particleRadius,
                ["particle_mass_kg"] = particleMass,
                ["collision_margin_m"] = collisionMargin,
                ["friction"] = friction,
                ["stretch_compliance"] = stretchCompliance,
                ["bend_compliance"] = bendCompliance,
                ["maximum_circumferential_stretch"] = maximumCircumferentialStretch,
                ["self_collision"] = selfCollision
            };
        }

        public void Grasp(string side, float maxDistance)
        {
            Transform target = side == "left" ? leftGripper : rightGripper;
            if (target == null)
                throw new InvalidOperationException($"{side} gripper is not configured");
            backend.Grasp(side, target, maxDistance);
        }

        public void Release(string side)
        {
            backend.Release(side);
        }

        public void ResetSock()
        {
            backend.Release("left");
            backend.Release("right");
            transform.SetPositionAndRotation(resetPosition, resetRotation);
            if (leftGripper != null)
                leftGripper.SetPositionAndRotation(leftResetPosition, leftResetRotation);
            if (rightGripper != null)
                rightGripper.SetPositionAndRotation(rightResetPosition, rightResetRotation);
            backend.Reset(resetParticles);
        }

        private Dictionary<string, object> GraspMapping(string side)
        {
            SockGraspState state = backend.GraspState(side);
            return new Dictionary<string, object>
            {
                ["side"] = side,
                ["attached"] = state.Attached,
                ["particle_indices"] = state.ParticleIndices.Cast<object>().ToList(),
                ["constraint_error"] = state.ConstraintError
            };
        }

        private Dictionary<string, object> Coverage()
        {
            Renderer[] legs = ActiveLegRenderers();
            if (maskCamera == null || sockRenderer == null || legs.Length == 0)
                return InvalidCoverage("mask camera or semantic renderers are not configured");
            MaskCoverage observation = SynchronizedMaskCapture.Measure(
                maskCamera, sockRenderer, legs, 256, 256);
            return new Dictionary<string, object>
            {
                ["valid"] = observation.Valid,
                ["coverage"] = observation.Valid ? observation.Coverage : null,
                ["sock_pixels"] = observation.SockPixels,
                ["leg_pixels"] = observation.LegPixels,
                ["overlap_pixels"] = observation.OverlapPixels,
                ["reason"] = observation.Reason,
                ["simulation_frame"] = Time.frameCount
            };
        }

        private static Dictionary<string, object> InvalidCoverage(string reason)
        {
            return new Dictionary<string, object>
            {
                ["valid"] = false,
                ["coverage"] = null,
                ["reason"] = reason,
                ["simulation_frame"] = Time.frameCount
            };
        }

        private Renderer[] ActiveLegRenderers()
        {
            if (legRenderers != null && legRenderers.Length > 0)
                return legRenderers.Where(item => item != null).ToArray();
            return legRenderer != null ? new[] { legRenderer } : Array.Empty<Renderer>();
        }

        private static List<object> Vectors(IEnumerable<Vector3> vectors)
        {
            return vectors.Select(
                vector => (object)new List<object> { vector.x, vector.y, vector.z }).ToList();
        }
    }

    public readonly struct SockGraspState
    {
        public SockGraspState(bool attached, int[] particleIndices, float constraintError)
        {
            Attached = attached;
            ParticleIndices = particleIndices ?? Array.Empty<int>();
            ConstraintError = constraintError;
        }

        public bool Attached { get; }
        public int[] ParticleIndices { get; }
        public float ConstraintError { get; }
    }

    public interface ISockPhysicsBackend
    {
        bool Available { get; }
        string Version { get; }
        Vector3[] Particles();
        Vector3[] Velocities();
        List<object> RegisteredColliders();
        List<object> Contacts();
        void Grasp(string side, Transform target, float maxDistance);
        void Release(string side);
        SockGraspState GraspState(string side);
        void Reset(Vector3[] positions);
    }

    internal sealed class UnavailableObiBackend : ISockPhysicsBackend
    {
        public bool Available => false;
        public string Version => "unavailable";
        public Vector3[] Particles() => Array.Empty<Vector3>();
        public Vector3[] Velocities() => Array.Empty<Vector3>();
        public List<object> RegisteredColliders() => new();
        public List<object> Contacts() => new();
        public void Grasp(string side, Transform target, float maxDistance) =>
            throw new InvalidOperationException(
                "Licensed Obi package and an Obi backend are required for grasp");
        public void Release(string side) { }
        public SockGraspState GraspState(string side) =>
            new(false, Array.Empty<int>(), 0);
        public void Reset(Vector3[] positions) { }
    }
}
