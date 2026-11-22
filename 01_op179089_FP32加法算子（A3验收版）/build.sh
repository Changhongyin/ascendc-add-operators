#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
cmake -S . -B build -DNPU_ARCH="${NPU_ARCH:-dav-2201}" -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
