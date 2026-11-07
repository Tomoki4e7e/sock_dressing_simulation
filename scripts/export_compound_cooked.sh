#!/usr/bin/env bash
# Export baked convex bytes using a separate project and standard Unity modules.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAYLOAD="$(realpath "${1:?Usage: export_compound_cooked.sh PAYLOAD OUTPUT_DIRECTORY}")"
OUTPUT="$(realpath -m "${2:?Output directory is required}")"
PROJECT="${SOCK_COOK_PROJECT:-/tmp/sock-convex-cook-export}"
UNITY="${UNITY_EDITOR:-${ROOT}/.deps/unity/2022.3.34f1/Editor/Unity}"
mkdir -p "${PROJECT}/Assets" "${PROJECT}/Packages" "${PROJECT}/ProjectSettings" "${OUTPUT}"
cp "${ROOT}/unity/Diagnostics/ConvexCookExport.cs" "${PROJECT}/Assets/ConvexCookExport.cs"
cat > "${PROJECT}/Packages/manifest.json" <<'JSON'
{"dependencies":{"com.unity.modules.physics":"1.0.0","com.unity.modules.jsonserialize":"1.0.0","com.unity.modules.assetbundle":"1.0.0"}}
JSON
cat > "${PROJECT}/ProjectSettings/ProjectVersion.txt" <<'VERSION'
m_EditorVersion: 2022.3.34f1
VERSION
SOCK_COOK_PAYLOAD="${PAYLOAD}" SOCK_COOK_OUTPUT="${OUTPUT}" \
  "${UNITY}" -batchmode -nographics -quit -projectPath "${PROJECT}" \
  -executeMethod ConvexCookExport.Export -logFile "${OUTPUT}/export.log"
test -f "${OUTPUT}/cooked-convex"
cp "${ROOT}/unity/Diagnostics/ConvexCookExport.cs" "${OUTPUT}/ConvexCookExport.cs"
sha256sum "${OUTPUT}/cooked-convex" "${OUTPUT}/ConvexCookExport.cs" > "${OUTPUT}/sha256.txt"
