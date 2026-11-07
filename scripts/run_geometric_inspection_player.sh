#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sock_inspection_port=unknown
for argument in "$@"; do
  case "$argument" in -port:*) sock_inspection_port="${argument#-port:}";; esac
done
exec "${ROOT}/Build/SockDressingPlayer/GeometricInspectionV1/Player.x86_64" -job-worker-count 2 -logFile "/tmp/sock-geometric-inspection-${sock_inspection_port}.log" "$@"
