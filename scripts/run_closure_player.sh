#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sock_closure_port=unknown
for argument in "$@"; do
  case "$argument" in -port:*) sock_closure_port="${argument#-port:}";; esac
done
exec "${ROOT}/Build/SockDressingPlayer/ClosureMeasureV5/Player.x86_64" -job-worker-count "${SOCK_CLOSURE_WORKERS:-2}" -logFile "/tmp/sock-closure-player-${sock_closure_port}.log" "$@"
