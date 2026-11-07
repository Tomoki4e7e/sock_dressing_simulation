#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp "${ROOT}/unity/RuntimeExtensions/SockLearningLighting.cs" "${ROOT}/RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockLearningLighting.cs"
export SOCK_BUILD_OUTPUT="${ROOT}/Build/SockDressingPlayer/RegisteredLightingV1"
export SOCK_BUILD_LOG_DIR="${ROOT}/artifacts/reaching-shareset-lit-v3/unity-build"
bash "${ROOT}/scripts/build_custom_player.sh" development
python3 - "${ROOT}" "${SOCK_BUILD_OUTPUT}" <<'PY'
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]);out=Path(sys.argv[2])/'runtime_sources'
src=root/'unity/RuntimeExtensions/SockLearningLighting.cs';shutil.copy2(src,out/src.name)
p=out/'manifest.json';m=json.loads(p.read_text());m[str(src.relative_to(root))]=hashlib.sha256(src.read_bytes()).hexdigest();p.write_text(json.dumps(m,indent=2)+'\n')
PY
