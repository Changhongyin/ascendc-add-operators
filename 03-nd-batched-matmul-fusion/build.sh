#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NPU_ARCH="${NPU_ARCH:-dav-2201}"
BUILD_DIR="${BUILD_DIR:-build}"

rm -rf "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -DNPU_ARCH="$NPU_ARCH" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel 4
echo "Built $BUILD_DIR/nd_bmm_fused for NPU_ARCH=$NPU_ARCH"
