#!/usr/bin/env bash
# Build the isolated contact reproduction without RCareWorld or Obi.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${SOCK_REPRO_PROJECT:-/tmp/sock-physx-repro-project}"
OUTPUT="${SOCK_REPRO_OUTPUT:-/tmp/sock-physx-repro-player}"
UNITY="${UNITY_EDITOR:-${ROOT}/.deps/unity/2022.3.34f1/Editor/Unity}"
mkdir -p "${PROJECT}/Assets" "${PROJECT}/Packages" "${PROJECT}/ProjectSettings" "${OUTPUT}"
cp "${ROOT}/unity/Diagnostics/ConvexContactRepro.cs" "${PROJECT}/Assets/ConvexContactRepro.cs"
cat > "${PROJECT}/Packages/manifest.json" <<'JSON'
{"dependencies":{"com.unity.modules.physics":"1.0.0","com.unity.modules.jsonserialize":"1.0.0"}}
JSON
cat > "${PROJECT}/ProjectSettings/ProjectVersion.txt" <<'VERSION'
m_EditorVersion: 2022.3.34f1
VERSION
SOCK_REPRO_PLAYER="${OUTPUT}/Player.x86_64" SOCK_REPRO_DEVELOPMENT="${SOCK_REPRO_DEVELOPMENT:-1}" \
  "${UNITY}" -batchmode -nographics -quit -projectPath "${PROJECT}" \
  -executeMethod ConvexContactReproBuild.Build -logFile "${OUTPUT}/build.log"
test -x "${OUTPUT}/Player.x86_64"
cp "${ROOT}/unity/Diagnostics/ConvexContactRepro.cs" "${OUTPUT}/ConvexContactRepro.cs"
sha256sum "${OUTPUT}/UnityPlayer.so" "${OUTPUT}/ConvexContactRepro.cs" > "${OUTPUT}/sha256.txt"
echo "Built isolated PhysX reproduction at ${OUTPUT}/Player.x86_64"
