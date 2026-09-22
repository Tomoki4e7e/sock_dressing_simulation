#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${ROOT}/RCareUnity"
UNITY="${UNITY_EDITOR:-${ROOT}/.deps/unity/2022.3.34f1/Editor/Unity}"
MODE="${1:-development}"
MODE_DIR="${MODE^}"
OUTPUT="${ROOT}/Build/SockDressingPlayer/${MODE_DIR}"
LOG_DIR="${ROOT}/artifacts/unity"
IMPORT_LOG="${LOG_DIR}/import-${MODE}.log"
BUILD_LOG="${LOG_DIR}/build-${MODE}.log"

if [[ ! -x "${UNITY}" ]]; then
  echo "Unity 2022.3.34f1 was not found at ${UNITY}." >&2
  echo "Install Unity Editor with Linux Build Support or set UNITY_EDITOR." >&2
  exit 2
fi
if [[ "${MODE}" != "development" && "${MODE}" != "release" ]]; then
  echo "Usage: $0 [development|release]" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${OUTPUT}"

if [[ ! -f "${PROJECT}/Library/ArtifactDB" ]] ||
   [[ ! -d "${PROJECT}/Library/ScriptAssemblies" ]]; then
  echo "Completing Unity asset import and script compilation..."
  "${UNITY}" \
    -batchmode \
    -quit \
    -projectPath "${PROJECT}" \
    -logFile "${IMPORT_LOG}"
fi

if [[ ! -f "${PROJECT}/Library/ArtifactDB" ]] ||
   [[ ! -d "${PROJECT}/Library/ScriptAssemblies" ]]; then
  echo "Unity import did not complete. See ${IMPORT_LOG}." >&2
  exit 3
fi

ARGS=(
  -batchmode
  -quit
  -projectPath "${PROJECT}"
  -executeMethod SockDressing.Editor.SockDressingBuild.BuildFromCommandLine
  -logFile "${BUILD_LOG}"
)
if [[ "${MODE}" == "development" ]]; then
  ARGS+=(-sockDevelopment)
fi

SOCK_BUILD_OUTPUT="${OUTPUT}" "${UNITY}" "${ARGS[@]}"

PLAYER="${OUTPUT}/Player.x86_64"
if [[ ! -x "${PLAYER}" ]] ||
   [[ ! -d "${OUTPUT}/Player_Data" ]] ||
   [[ ! -f "${OUTPUT}/UnityPlayer.so" ]]; then
  echo "Unity exited without a complete Player. See ${BUILD_LOG}." >&2
  exit 4
fi

echo "Built ${MODE} Player at ${PLAYER}"
