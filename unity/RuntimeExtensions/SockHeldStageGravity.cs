#if OBI
using System;
using System.Linq;
using Obi;
using UnityEngine;

namespace RCareWorld.Attributes
{
    public partial class SockClothAttr
    {
        private float heldStageTailGravityScale = 1;
        private int[] heldStageFitTailParticles = new int[0];
        private bool[] heldStageFitTailMask;

        private Vector3 HeldStageEffectiveGravity(int particle)
        {
            bool tail = heldStageFitTailMask != null && particle < heldStageFitTailMask.Length && heldStageFitTailMask[particle];
            return solver.gravity*(tail ? heldStageTailGravityScale : 1);
        }

        [RFUAPI]
        public void ConfigureSockHeldStageFitTailGravity(float materialBoundary, float gravityScale)
        {
            if (!circumferenceFitActive || !float.IsFinite(gravityScale) || gravityScale <= 0 || gravityScale > 1 ||
                !float.IsFinite(materialBoundary) || materialBoundary < 0)
                throw new ArgumentException("Tail gravity requires an active fit and a scale in (0,1]");
            var blueprint = cloth.sourceBlueprint as ObiClothBlueprint;
            float minimum = circumferenceFitZ.Min();
            heldStageFitTailParticles = Enumerable.Range(0,cloth.particleCount)
                .Where(p => blueprint.topology.clusters[p].centroid.z*blueprint.scale.z-minimum > materialBoundary+1e-6f).ToArray();
            if (heldStageFitTailParticles.Length == 0 || heldStageFitTailParticles.Length == cloth.particleCount)
                throw new ArgumentException("Select only the free material beyond the anatomical toe");
            heldStageTailGravityScale = gravityScale;
            heldStageFitTailMask = new bool[cloth.particleCount];
            foreach (int p in heldStageFitTailParticles) heldStageFitTailMask[p] = true;
            solver.OnExternalSubstepStart -= ApplyHeldStageTailGravity;
            solver.OnExternalSubstepStart += ApplyHeldStageTailGravity;
            collectData.AddDataNextStep("sock_held_stage_tail_gravity",new System.Collections.Generic.Dictionary<string,object> {
                {"scale",gravityScale},{"material_boundary_m",materialBoundary},
                {"affected_particles",heldStageFitTailParticles.Length},
                {"wrapped_gravity_scale",1f},{"mass_or_material_modified",false},
                {"contact_gravity_model","force_consistent_tail_gravity_v2"}});
        }

        private void ApplyHeldStageTailGravity(ObiSolver active,float stepTime,float dt)
        {
            if (!circumferenceFitActive || heldStageTailGravityScale >= 1) return;
            Vector3 compensation = solver.transform.InverseTransformVector(-solver.gravity*((1-heldStageTailGravityScale)*dt));
            foreach (int p in heldStageFitTailParticles)
            {
                int i = cloth.GetParticleRuntimeIndex(p);
                if (solver.invMasses[i] > 0)
                    solver.velocities[i] += new Vector4(compensation.x,compensation.y,compensation.z,0);
            }
        }

        [RFUAPI]
        public void SetSockHeldStageGravityScale(float scale)
        {
            if (!sockPatternDrapeActive || !float.IsFinite(scale) || scale <= 0 || scale > 1)
                throw new ArgumentException("Drape gravity scale must be in (0,1]");
            solver.gravity = new Vector3(0,-9.81f*scale,0);
            solver.gravitySpace = Space.World;
            solver.PushSolverParameters();
        }

        [RFUAPI]
        public void BeginSockHeldStageCircumferenceFit()
        {
            BeginSockCircumferenceFit();
            // Use the contact ordering already used by A's toe preparation:
            // contact correction before friction/damping, not a second
            // velocity reconstruction after damping on the long wrapped part.
            solver.OnExternalSubstepFinalized -= SolveSockFitContact;
            solver.OnExternalSubstepEnd += SolveSockFitContact;
            SetHeldStageMaterialNeighbours();
        }

        [RFUAPI]
        public void SetSockHeldStageCircumferenceScale(float scale)
        {
            SetSockCircumferenceScale(scale);
            SetHeldStageMaterialNeighbours();
        }

        private void SetHeldStageMaterialNeighbours()
        {
            // A's toe preparation uses material coordinates for self-contact
            // neighbour exclusion. Bent world-space references would suppress
            // contacts between opposite sides of the foot/ankle fold in D/E.
            var blueprint = cloth.sourceBlueprint as ObiClothBlueprint;
            var distances = cloth.GetConstraintsByType(Oni.ConstraintType.Distance) as ObiConstraints<ObiDistanceConstraintsBatch>;
            var scales = new System.Collections.Generic.List<float>();
            foreach (var batch in distances.batches)
                for (int k = 0; k < batch.constraintCount; k++)
                {
                    int a = batch.particleIndices[2*k], b = batch.particleIndices[2*k+1];
                    Vector3 ra = Vector3.Scale(blueprint.scale, blueprint.topology.clusters[a].centroid);
                    Vector3 rb = Vector3.Scale(blueprint.scale, blueprint.topology.clusters[b].centroid);
                    if (Mathf.Abs(ra.z-rb.z) < 1e-6f && Vector3.Distance(ra,rb) > 1e-8f)
                        scales.Add(structuralRestLengths[StructuralEdgeKey(a,b)]/Vector3.Distance(ra,rb));
                }
            scales.Sort();
            float hoopScale = scales[scales.Count/2];
            for (int p = 0; p < cloth.particleCount; p++)
            {
                Vector3 rest = Vector3.Scale(blueprint.scale, blueprint.topology.clusters[p].centroid);
                rest.x *= hoopScale; rest.y *= hoopScale;
                solver.restPositions[cloth.GetParticleRuntimeIndex(p)] = new Vector4(rest.x,rest.y,rest.z,1);
            }
        }

        // The original A section-chain solver accepts only a short supported
        // length. C--E cover more of the foot: reuse that solver with their
        // measured material boundary, without changing A's API or defaults.
        [RFUAPI]
        public void BeginSockHeldStageGravityDrape(string referenceState, float pinnedLength)
        {
            var blueprint = cloth.sourceBlueprint as ObiClothBlueprint;
            float minimum = Enumerable.Range(0, cloth.particleCount)
                .Min(p => blueprint.topology.clusters[p].centroid.z * blueprint.scale.z);
            float maximum = Enumerable.Range(0, cloth.particleCount)
                .Max(p => blueprint.topology.clusters[p].centroid.z * blueprint.scale.z);
            if (!float.IsFinite(pinnedLength) || pinnedLength <= .03f || pinnedLength >= maximum-minimum-1e-6f)
                throw new ArgumentException("A stage drape requires a supported cuff and a nonempty free end");
            BeginSockPatternGravityDrape(referenceState, .075f);
            sockPatternPinnedLength = pinnedLength;
            for (int p = 0; p < cloth.particleCount; p++)
            {
                float depth = blueprint.topology.clusters[p].centroid.z * blueprint.scale.z - minimum;
                sockPatternPinned[p] = depth <= pinnedLength + 1e-6f;
            }
            // Use the material-section correspondence introduced for A's toe
            // tail, rather than binding a vertex to a nearby spatial section
            // on the bent foot. This also uses A's bounded tail curvature.
            sockToeHybridActive = true;
            BuildDrapeSections(blueprint);
            drapeBoundaryBlend = 4;
            sockToeTailFollowers = new bool[cloth.particleCount];
            sockToeTailSection = new int[cloth.particleCount];
            sockToeTailOffset = new Vector3[cloth.particleCount];
            for (int k = drapeLastFixed+1; k < drapeSections.Length; k++)
                foreach (int p in drapeSections[k].particles)
                {
                    sockToeTailFollowers[p] = true;
                    sockToeTailSection[p] = k;
                    sockToeTailOffset[p] = sockPatternHeldPositions[p]-drapeSections[k].center;
                }
            RestoreSockPatternHeld();
            GetSockPatternGravityDrape();
        }
    }
}
#endif
