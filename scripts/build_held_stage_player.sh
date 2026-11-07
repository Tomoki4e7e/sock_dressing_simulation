#!/usr/bin/env bash
set -euo pipefail
STAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cp "${STAGE_ROOT}/unity/RuntimeExtensions/SockHeldStageGravity.cs" \
  "${STAGE_ROOT}/RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockHeldStageGravity.cs"
python3 - "${STAGE_ROOT}" <<'PY'
import sys
from pathlib import Path
path=Path(sys.argv[1])/'RCareUnity/Assets/RCareCommon/Scripts/Attributes/Obi/SockCircumferenceFit.cs'
source=path.read_text()
old='Vector3 gravityForce=inverseMass>0 ? solver.gravity/inverseMass : Vector3.zero;'
new='Vector3 gravityForce=inverseMass>0 ? HeldStageEffectiveGravity(p)/inverseMass : Vector3.zero;'
if old in source:
    path.write_text(source.replace(old,new))
elif new not in source:
    raise SystemExit('Unexpected native fitter: cannot install tail-specific gravity consistently')
PY
export SOCK_BUILD_OUTPUT="${STAGE_ROOT}/Build/SockDressingPlayer/HeldStagesV1"
export SOCK_BUILD_LOG_DIR="${STAGE_ROOT}/artifacts/cde-tight-fit/unity-build"
bash "${STAGE_ROOT}/scripts/build_custom_player.sh" development
python3 - "${STAGE_ROOT}" "${SOCK_BUILD_OUTPUT}" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
root, player = map(Path, sys.argv[1:])
source = root/'unity/RuntimeExtensions/SockHeldStageGravity.cs'
archive = player/'runtime_sources'
shutil.copy2(source, archive/source.name)
path = archive/'manifest.json'
manifest = json.loads(path.read_text())
manifest[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
path.write_text(json.dumps(manifest, indent=2)+'\n')
PY
