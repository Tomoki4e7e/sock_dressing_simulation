#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${ROOT}/RCareUnity"
UNITY="${UNITY_EDITOR:-${ROOT}/.deps/unity/2022.3.34f1/Editor/Unity}"
MODE="${1:-development}"

if [[ ! -x "${UNITY}" ]]; then
  echo "Unity 2022.3.34f1 was not found at ${UNITY}." >&2
  echo "Install Unity Editor with Linux Build Support or set UNITY_EDITOR." >&2
  exit 2
fi
if [[ "${MODE}" != "development" && "${MODE}" != "release" ]]; then
  echo "Usage: $0 [development|release]" >&2
  exit 2
fi

ARGS=(
  -batchmode
  -quit
  -projectPath "${PROJECT}"
  -executeMethod SockDressing.Editor.SockDressingBuild.BuildFromCommandLine
  -logFile -
)
if [[ "${MODE}" == "development" ]]; then
  ARGS+=(-sockDevelopment)
fi

SOCK_BUILD_OUTPUT="${ROOT}/Build/SockDressingPlayer" "${UNITY}" "${ARGS[@]}"
