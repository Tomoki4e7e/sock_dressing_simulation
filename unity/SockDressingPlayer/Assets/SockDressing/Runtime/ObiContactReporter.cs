#if SOCKDRESSING_OBI
using System.Collections.Generic;
using UnityEngine;

namespace SockDressing
{
    /// <summary>
    /// Version-neutral contact buffer. An Obi-version-specific collision
    /// callback calls Report for penetrating contacts before each Collect.
    /// </summary>
    public sealed class ObiContactReporter : MonoBehaviour
    {
        private readonly List<object> contacts = new();

        public void BeginSolverStep()
        {
            contacts.Clear();
        }

        public void Report(
            int particleIndex,
            int colliderObjectId,
            Vector3 position,
            Vector3 normal,
            float penetration,
            float forceProxy)
        {
            contacts.Add(new Dictionary<string, object>
            {
                ["particle_index"] = particleIndex,
                ["collider_id"] = colliderObjectId,
                ["position"] = new List<object> { position.x, position.y, position.z },
                ["normal"] = new List<object> { normal.x, normal.y, normal.z },
                ["penetration_m"] = penetration,
                ["force_proxy"] = forceProxy
            });
        }

        public List<object> Snapshot()
        {
            return new List<object>(contacts);
        }
    }
}
#endif
