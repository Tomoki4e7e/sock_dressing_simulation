#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp "${ROOT}/unity/RuntimeExtensions/SockCompoundReaching.cs" "${ROOT}/RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockCompoundReaching.cs"
cp "${ROOT}/unity/RuntimeExtensions/ObiConvexCollisionMesh.cs" "${ROOT}/RCareUnity/Assets/Paid Dependencies/Obi/Scripts/Common/Collisions/ColliderTrackers/Trackers3D/ObiConvexCollisionMesh.cs"
export SOCK_BUILD_OUTPUT="${ROOT}/Build/SockDressingPlayer/CompoundReachingV1"
export SOCK_BUILD_LOG_DIR="${ROOT}/artifacts/compound-reaching/unity"
exec bash "${ROOT}/scripts/build_custom_player.sh" development
