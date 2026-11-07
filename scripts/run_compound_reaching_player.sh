#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT}/Build/SockDressingPlayer/CompoundReachingV1/Player.x86_64" -job-worker-count 2 -logFile /tmp/sock-compound-player.log "$@"
