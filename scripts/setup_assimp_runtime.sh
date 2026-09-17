#!/usr/bin/env bash
set -euo pipefail

ASSIMP_COMMIT="80799bdbf90ce626475635815ee18537718a05b1"
REPOSITORY="https://github.com/assimp/assimp.git"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-${PROJECT_ROOT}/.deps/assimp-4.1.0}"
BUILD="${SOURCE}/build"

if [[ ! -d "${SOURCE}/.git" ]]; then
  git clone --filter=blob:none --no-checkout "${REPOSITORY}" "${SOURCE}"
fi

git -C "${SOURCE}" fetch origin "${ASSIMP_COMMIT}"
git -C "${SOURCE}" checkout --detach "${ASSIMP_COMMIT}"
cmake \
  -S "${SOURCE}" \
  -B "${BUILD}" \
  -G Ninja \
  -DASSIMP_BUILD_TESTS=OFF \
  -DASSIMP_BUILD_ASSIMP_TOOLS=OFF \
  -DASSIMP_BUILD_SAMPLES=OFF \
  -DASSIMP_BUILD_ZLIB=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD}" --parallel

test -f "${BUILD}/code/libassimp.so"
echo "Built Assimp 4.1 runtime compatibility library at ${BUILD}/code/libassimp.so"
