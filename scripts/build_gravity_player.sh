#!/usr/bin/env bash
set -euo pipefail
GRAVITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp "${GRAVITY_ROOT}/unity/RuntimeExtensions/SockGravityReference.cs" "${GRAVITY_ROOT}/RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockGravityReference.cs"
export SOCK_BUILD_OUTPUT="${GRAVITY_ROOT}/Build/SockDressingPlayer/RegisteredGravityV1"
export SOCK_BUILD_LOG_DIR="${GRAVITY_ROOT}/artifacts/gravity-calibration-v1/unity-build"
bash "${GRAVITY_ROOT}/scripts/build_custom_player.sh" development
python3 - "${GRAVITY_ROOT}" "${SOCK_BUILD_OUTPUT}" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2])/'runtime_sources'
for name in ('SockGravityReference.cs','SockLearningLighting.cs'):
 src=root/'unity/RuntimeExtensions'/name;shutil.copy2(src,out/name)
 p=out/'manifest.json';m=json.loads(p.read_text());m[str(src.relative_to(root))]=hashlib.sha256(src.read_bytes()).hexdigest();p.write_text(json.dumps(m,indent=2)+'\n')
PY
