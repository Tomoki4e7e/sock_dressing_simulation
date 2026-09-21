#if SOCKDRESSING_OBI
using System;
using System.Collections.Generic;
using System.Linq;
using Obi;
using UnityEngine;

namespace SockDressing
{
    /// <summary>
    /// Obi 6.x/7.x adapter. Enable SOCKDRESSING_OBI only after importing the
    /// licensed package and assign this component to SockClothRuntime.obiAdapter.
    /// Grasped particles are kinematically pinned to their selected gripper.
    /// </summary>
    [RequireComponent(typeof(ObiCloth))]
    public sealed class ObiSockBackend : MonoBehaviour, ISockPhysicsBackend
    {
        [SerializeField] private int[] openingParticleIndices = Array.Empty<int>();
        [SerializeField] private int particlesPerGrasp = 4;

        private ObiCloth actor;
        private readonly Dictionary<string, PinSet> pins = new();
        private readonly List<object> contacts = new();
        private Vector3[] initialLocalPositions = Array.Empty<Vector3>();

        public bool Available => actor != null && actor.isLoaded;
        public string Version =>
            typeof(ObiActor).Assembly.GetName().Version?.ToString() ?? "unknown";

        private void Awake()
        {
            actor = GetComponent<ObiCloth>();
            pins["left"] = new PinSet();
            pins["right"] = new PinSet();
        }

        private void Start()
        {
            initialLocalPositions = Particles()
                .Select(transform.InverseTransformPoint)
                .ToArray();
            if (actor.solver != null)
                actor.solver.OnCollision += SolverOnCollision;
        }

        private void OnDestroy()
        {
            if (actor != null && actor.solver != null)
                actor.solver.OnCollision -= SolverOnCollision;
        }

        private void FixedUpdate()
        {
            foreach (PinSet pin in pins.Values)
                Drive(pin);
        }

        public Vector3[] Particles()
        {
            if (!Available)
                return Array.Empty<Vector3>();
            var result = new Vector3[actor.activeParticleCount];
            for (int i = 0; i < result.Length; i++)
            {
                int solverIndex = actor.solverIndices[i];
                Vector4 position = actor.solver.positions[solverIndex];
                result[i] = actor.solver.transform.TransformPoint(position);
            }
            return result;
        }

        public Vector3[] Velocities()
        {
            if (!Available)
                return Array.Empty<Vector3>();
            var result = new Vector3[actor.activeParticleCount];
            for (int i = 0; i < result.Length; i++)
            {
                Vector4 velocity = actor.solver.velocities[actor.solverIndices[i]];
                result[i] = actor.solver.transform.TransformVector(velocity);
            }
            return result;
        }

        public List<object> RegisteredColliders()
        {
            return FindObjectsOfType<StableObjectId>()
                .Where(item => item.obiColliderExpected)
                .Select(item => (object)new Dictionary<string, object>
                {
                    ["object_id"] = item.objectId,
                    ["region"] = item.region,
                    ["enabled"] = item.isActiveAndEnabled &&
                        item.GetComponent<ObiCollider>() != null
                }).ToList();
        }

        public List<object> Contacts()
        {
            return new List<object>(contacts);
        }

        public void Grasp(string side, Transform target, float maxDistance)
        {
            if (!Available)
                throw new InvalidOperationException("Obi cloth is not loaded");
            Release(side);
            int[] candidates = openingParticleIndices.Length > 0
                ? openingParticleIndices
                : Enumerable.Range(0, actor.activeParticleCount).ToArray();
            int[] selected = candidates
                .Where(index => index >= 0 && index < actor.activeParticleCount)
                .Select(index => new
                {
                    Index = index,
                    Distance = Vector3.Distance(Particle(index), target.position)
                })
                .Where(item => item.Distance <= maxDistance)
                .OrderBy(item => item.Distance)
                .Take(Math.Max(1, particlesPerGrasp))
                .Select(item => item.Index)
                .ToArray();
            var other = pins[side == "left" ? "right" : "left"].ActorIndices;
            selected = selected.Except(other).ToArray();
            if (selected.Length == 0)
                return;
            var pin = pins[side];
            pin.Target = target;
            pin.ActorIndices = selected;
            pin.TargetLocalOffsets = selected
                .Select(index => target.InverseTransformPoint(Particle(index)))
                .ToArray();
            pin.OriginalInverseMasses = selected
                .Select(index => actor.solver.invMasses[actor.solverIndices[index]])
                .ToArray();
            Drive(pin);
        }

        public void Release(string side)
        {
            PinSet pin = pins[side];
            for (int i = 0; i < pin.ActorIndices.Length; i++)
            {
                int solverIndex = actor.solverIndices[pin.ActorIndices[i]];
                actor.solver.invMasses[solverIndex] = pin.OriginalInverseMasses[i];
            }
            pin.Clear();
        }

        public SockGraspState GraspState(string side)
        {
            PinSet pin = pins[side];
            return new SockGraspState(
                pin.ActorIndices.Length > 0,
                pin.ActorIndices.ToArray(),
                ConstraintError(pin));
        }

        public void Reset(Vector3[] positions)
        {
            if (!Available)
                return;
            Vector3[] source = positions.Length == actor.activeParticleCount
                ? positions
                : initialLocalPositions.Select(transform.TransformPoint).ToArray();
            for (int i = 0; i < Math.Min(source.Length, actor.activeParticleCount); i++)
            {
                int solverIndex = actor.solverIndices[i];
                actor.solver.positions[solverIndex] =
                    actor.solver.transform.InverseTransformPoint(source[i]);
                actor.solver.velocities[solverIndex] = Vector4.zero;
            }
        }

        private Vector3 Particle(int actorIndex)
        {
            Vector4 value = actor.solver.positions[actor.solverIndices[actorIndex]];
            return actor.solver.transform.TransformPoint(value);
        }

        private void Drive(PinSet pin)
        {
            if (pin.Target == null)
                return;
            for (int i = 0; i < pin.ActorIndices.Length; i++)
            {
                int solverIndex = actor.solverIndices[pin.ActorIndices[i]];
                Vector3 world = pin.Target.TransformPoint(pin.TargetLocalOffsets[i]);
                actor.solver.positions[solverIndex] =
                    actor.solver.transform.InverseTransformPoint(world);
                actor.solver.velocities[solverIndex] = Vector4.zero;
                actor.solver.invMasses[solverIndex] = 0;
            }
        }

        private float ConstraintError(PinSet pin)
        {
            if (pin.Target == null || pin.ActorIndices.Length == 0)
                return 0;
            float maximum = 0;
            for (int i = 0; i < pin.ActorIndices.Length; i++)
            {
                Vector3 expected = pin.Target.TransformPoint(pin.TargetLocalOffsets[i]);
                maximum = Mathf.Max(maximum, Vector3.Distance(expected, Particle(pin.ActorIndices[i])));
            }
            return maximum;
        }

        private void SolverOnCollision(
            object sender,
            ObiSolver.ObiCollisionEventArgs eventArgs)
        {
            contacts.Clear();
            var world = ObiColliderWorld.GetInstance();
            foreach (Oni.Contact contact in eventArgs.contacts)
            {
                if (contact.distance >= 0.01f)
                    continue;
                ObiColliderBase collider = world.colliderHandles[contact.bodyB].owner;
                StableObjectId stable = collider != null
                    ? collider.GetComponentInParent<StableObjectId>()
                    : null;
                if (stable == null)
                    continue;
                int solverParticle = actor.solver.simplices[contact.bodyA];
                int actorParticle = Array.IndexOf(actor.solverIndices, solverParticle);
                if (actorParticle < 0)
                    continue;
                Vector3 point = actor.solver.transform.TransformPoint(contact.pointA);
                contacts.Add(new Dictionary<string, object>
                {
                    ["particle_index"] = actorParticle,
                    ["collider_id"] = stable.objectId,
                    ["position"] = new List<object> { point.x, point.y, point.z },
                    ["normal"] = new List<object>
                    {
                        contact.normal.x, contact.normal.y, contact.normal.z
                    },
                    ["penetration_m"] = Mathf.Max(0, -contact.distance),
                    ["force_proxy"] = Mathf.Abs(contact.normalImpulse)
                });
            }
        }

        [Serializable]
        private sealed class PinSet
        {
            public Transform Target;
            public int[] ActorIndices = Array.Empty<int>();
            public Vector3[] TargetLocalOffsets = Array.Empty<Vector3>();
            public float[] OriginalInverseMasses = Array.Empty<float>();

            public void Clear()
            {
                Target = null;
                ActorIndices = Array.Empty<int>();
                TargetLocalOffsets = Array.Empty<Vector3>();
                OriginalInverseMasses = Array.Empty<float>();
            }
        }
    }
}
#endif
