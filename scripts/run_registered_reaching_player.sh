#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPLAY_GRAPHICS_ARGS=()
if [[ "${SOCK_REGISTERED_BATCHMODE:-0}" == "1" ]]; then
  REPLAY_GRAPHICS_ARGS=(-batchmode -force-glcore)
fi
exec "${SOCK_REGISTERED_PLAYER:-${ROOT}/Build/SockDressingPlayer/RegisteredReachingV3/Player.x86_64}" -job-worker-count 2 -logFile "${SOCK_REGISTERED_PLAYER_LOG:-/tmp/sock-registered-reaching-player.log}" "${REPLAY_GRAPHICS_ARGS[@]}" "$@"
