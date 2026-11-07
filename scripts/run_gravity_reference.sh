#!/usr/bin/env bash
set -euo pipefail
GRAVITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${GRAVITY_ROOT}/artifacts/gravity-calibration-v1/native"
export PYTHONPATH="${GRAVITY_ROOT}/.deps/RCareWorld/pyrcareworld:/opt/ros/noetic/lib/python3/dist-packages${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=2
export LIBGL_ALWAYS_SOFTWARE=1 __GLX_VENDOR_LIBRARY_NAME=mesa SOCK_REGISTERED_BATCHMODE=1
export SOCK_REGISTERED_PLAYER="${GRAVITY_ROOT}/Build/SockDressingPlayer/RegisteredGravityV1/Player.x86_64"
export SOCK_REGISTERED_PLAYER_LOG="${GRAVITY_ROOT}/artifacts/gravity-calibration-v1/native/player.log"
exec "${SOCK_GRAVITY_PYTHON:-/tmp/sock-diversity-venv/bin/python}" -u "${GRAVITY_ROOT}/scripts/probe_gravity_reference.py" "$@"
