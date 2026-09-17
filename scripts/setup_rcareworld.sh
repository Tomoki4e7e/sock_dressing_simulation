#!/usr/bin/env bash
set -euo pipefail

COMMIT="ae0900be3e450ae08d6137468970d0ac473a001b"
REPOSITORY="https://github.com/empriselab/RCareWorld.git"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-${PROJECT_ROOT}/.deps/RCareWorld}"

if [[ ! -d "${DESTINATION}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${REPOSITORY}" "${DESTINATION}"
fi

git -C "${DESTINATION}" fetch origin "${COMMIT}"
git -C "${DESTINATION}" checkout --detach "${COMMIT}"
chmod +x "${DESTINATION}/Build/Player.x86_64"
python3 -m pip install -e "${DESTINATION}/pyrcareworld"
"${PROJECT_ROOT}/scripts/setup_assimp_runtime.sh"

INSTALLED="$(git -C "${DESTINATION}" rev-parse HEAD)"
[[ "${INSTALLED}" == "${COMMIT}" ]] || {
  echo "RCareWorld checkout verification failed: ${INSTALLED}" >&2
  exit 1
}
echo "Installed pyrcareworld from ${INSTALLED}"

if ldd "${DESTINATION}/Build/Player.x86_64" 2>/dev/null | grep -q "not found"; then
  echo "Warning: the upstream checkout does not contain every Unity runtime library." >&2
  echo "Run 'python3 -m sock_dressing_simulation.cli doctor'; obtain UnityPlayer.so before live smoke." >&2
fi
