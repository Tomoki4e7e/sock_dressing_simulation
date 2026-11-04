#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sock_reaching_port=unknown
for argument in "$@"; do
  case "$argument" in -port:*) sock_reaching_port="${argument#-port:}";; esac
done
exec "${ROOT}/Build/SockDressingPlayer/ArmReachingV1/Player.x86_64" -job-worker-count 2 -logFile "/tmp/sock-arm-reaching-${sock_reaching_port}.log" "$@"
