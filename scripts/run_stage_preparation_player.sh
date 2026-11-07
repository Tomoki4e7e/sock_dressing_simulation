#!/usr/bin/env bash
set -euo pipefail
STAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_PLAYER="RigidSkinV1"
if [[ "${SOCK_STAGE_GRAVITY:-0}" == "1" ]]; then STAGE_PLAYER="HeldStagesV1"; fi
exec "${STAGE_ROOT}/Build/SockDressingPlayer/${STAGE_PLAYER}/Player.x86_64" \
  -job-worker-count 2 -logFile "${SOCK_STAGE_PLAYER_LOG:-/tmp/sock-stage-preparation.log}" "$@"
