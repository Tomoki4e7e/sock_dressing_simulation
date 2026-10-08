#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${ROOT}/RCareUnity"
UNITY="${UNITY_EDITOR:-${ROOT}/.deps/unity/2022.3.34f1/Editor/Unity}"
MODE="${1:-development}"
MODE_DIR="${MODE^}"
OUTPUT="${SOCK_BUILD_OUTPUT:-${ROOT}/Build/SockDressingPlayer/${MODE_DIR}}"
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

python3 - "${ROOT}" "${OUTPUT}" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);player=Path(sys.argv[2]);archive=player/'runtime_sources';archive.mkdir(exist_ok=True)
paths=[root/'RCareUnity/Assets/RCareCommon/Scripts/Main/PlayerMain.cs',
       root/'RCareUnity/Assets/Paid Dependencies/Obi/Scripts/Common/Solver/ObiSolver.cs',
       root/'RCareUnity/Assets/Paid Dependencies/Obi/Scripts/Common/Backends/Burst/Solver/BurstSolverImpl.cs',
       root/'RCareUnity/Assets/SockDressing/Resources/FootSkinHullGeometry.json']
paths += [root/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi'/name for name in
          ['SockClothAttr.cs','SockPreparedState.cs','FootContactGeometry.cs','FootConvexSolid.cs','FootSurfaceHistory.cs','FootSkinHullCollider.cs','SockFootSkinHull.cs']]
manifest={}
for source in paths:
    if not source.exists():continue
    shutil.copyfile(source,archive/source.name)
    manifest[str(source.relative_to(root))]=hashlib.sha256(source.read_bytes()).hexdigest()
for name in ['RCareWorld.dll','Obi.dll']:
    manifest['compiled_'+name]=hashlib.sha256((player/'Player_Data/Managed'/name).read_bytes()).hexdigest()
(archive/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
echo "Built ${MODE} Player at ${PLAYER}"
