#!/usr/bin/env python3
"""Isolated cloud Atlas A3 / CANN CPU multi-dtype & multi-path matrix verifier for cannjudge_2524.

Builds a standalone verification binary in /tmp (leaving official main.asc / sample.py untouched),
runs all 5 dispatch paths x 4 dtypes on real CANN hardware, and optionally runs mssanitizer
(memcheck + racecheck) and msprof op.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent

MATRIX_MAIN_ASC = r'''#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t*
#endif
#ifndef TENSOR_GROUP_INFO_DEFINED
#define TENSOR_GROUP_INFO_DEFINED
struct TensorInfo {
    const int64_t* shape;
    int64_t numDims;
    int32_t dtype;
};
struct TensorGroupInfo {
    const TensorInfo* tensors;
    int64_t numTensors;
};
#endif
#ifdef ASCENDC_CPU_DEBUG
#include "cpu_debug_launch.h"
#endif
#include "kernel.asc"

namespace {
void CheckAcl(aclError code, const char* op) {
    if (code != ACL_SUCCESS) {
        throw std::runtime_error(std::string(op) + " failed: " + std::to_string(code));
    }
}

uint16_t FloatToFp16Bits(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    uint32_t exp = (bits >> 23) & 0xFFU;
    uint32_t mant = bits & 0x7FFFFFU;
    if (exp == 255) return static_cast<uint16_t>(sign | 0x7C00U | (mant ? 1U : 0U));
    int32_t ne = static_cast<int32_t>(exp) - 127 + 15;
    if (ne >= 31) return static_cast<uint16_t>(sign | 0x7C00U);
    if (ne <= 0) return sign;
    uint32_t r = mant >> 13;
    uint32_t rem = mant & 0x1FFFU;
    if (rem > 0x1000U || (rem == 0x1000U && (r & 1U))) {
        ++r;
        if (r == 0x0400U) { r = 0; ++ne; if (ne >= 31) return static_cast<uint16_t>(sign | 0x7C00U); }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(ne) << 10) | r);
}

float Fp16BitsToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16;
    uint32_t exp = (h >> 10) & 0x1FU;
    uint32_t mant = h & 0x03FFU;
    uint32_t fb = (exp == 0) ? sign : ((exp == 31) ? (sign | 0x7F800000U | (mant << 13))
                                                   : (sign | ((exp + 112U) << 23) | (mant << 13)));
    float out = 0.0f;
    std::memcpy(&out, &fb, 4);
    return out;
}

uint16_t FloatToBf16Bits(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    if (((bits >> 23) & 0xFFU) == 255 && (bits & 0x7FFFFFU)) return static_cast<uint16_t>((bits >> 16) | 0x0040U);
    return static_cast<uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

float Bf16BitsToFloat(uint16_t b) {
    uint32_t bits = static_cast<uint32_t>(b) << 16;
    float out = 0.0f;
    std::memcpy(&out, &bits, 4);
    return out;
}

std::vector<int64_t> BcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    size_t r = std::max(a.size(), b.size());
    std::vector<int64_t> o(r, 1);
    for (size_t i = 0; i < r; ++i) {
        int64_t da = (i + a.size() >= r) ? a[i + a.size() - r] : 1;
        int64_t db = (i + b.size() >= r) ? b[i + b.size() - r] : 1;
        o[i] = std::max(da, db);
    }
    return o;
}

uint64_t Prod(const std::vector<int64_t>& s) {
    uint64_t n = 1;
    for (int64_t d : s) n *= static_cast<uint64_t>(d);
    return n;
}

std::vector<uint64_t> Strides(const std::vector<int64_t>& s, size_t r) {
    std::vector<uint64_t> st(r, 0);
    uint64_t acc = 1;
    for (size_t k = 0; k < s.size(); ++k) {
        size_t i = s.size() - 1 - k;
        if (s[i] > 1) { st[r - 1 - k] = acc; acc *= static_cast<uint64_t>(s[i]); }
    }
    return st;
}

void RunCase(const char* name, int32_t dtype, const std::vector<int64_t>& s1,
             const std::vector<int64_t>& s2, int64_t cores, aclrtStream stream) {
    const auto sy = BcastShape(s1, s2);
    const uint64_t n1 = Prod(s1), n2 = Prod(s2), ny = Prod(sy);
    const size_t elemBytes = (dtype == 1 || dtype == 2) ? 2 : 4;
    const size_t guardBytes = 64;
    std::vector<uint8_t> h1(n1 * elemBytes), h2(n2 * elemBytes), hy(ny * elemBytes + guardBytes, 0xA5);

    auto fill = [&](std::vector<uint8_t>& buf, uint64_t n, int seed) {
        for (uint64_t i = 0; i < n; ++i) {
            float fv = (static_cast<int32_t>((i * 29 + static_cast<uint64_t>(seed) * 13) % 129) - 64) * 0.25f;
            if (dtype == 0) {
                std::memcpy(buf.data() + i * 4, &fv, 4);
            } else if (dtype == 5) {
                int32_t iv = static_cast<int32_t>(i * 7 - seed * 100);
                std::memcpy(buf.data() + i * 4, &iv, 4);
            } else if (dtype == 1) {
                uint16_t hv = FloatToFp16Bits(fv);
                std::memcpy(buf.data() + i * 2, &hv, 2);
            } else {
                uint16_t bv = FloatToBf16Bits(fv);
                std::memcpy(buf.data() + i * 2, &bv, 2);
            }
        }
    };
    fill(h1, n1, 0);
    fill(h2, n2, 1);

    void *d1 = nullptr, *d2 = nullptr, *dy = nullptr;
    CheckAcl(aclrtMalloc(&d1, std::max<size_t>(1, h1.size()), ACL_MEM_MALLOC_HUGE_FIRST), "malloc x1");
    CheckAcl(aclrtMalloc(&d2, std::max<size_t>(1, h2.size()), ACL_MEM_MALLOC_HUGE_FIRST), "malloc x2");
    CheckAcl(aclrtMalloc(&dy, hy.size(), ACL_MEM_MALLOC_HUGE_FIRST), "malloc y");
    CheckAcl(aclrtMemcpy(d1, h1.size(), h1.data(), h1.size(), ACL_MEMCPY_HOST_TO_DEVICE), "h2d x1");
    CheckAcl(aclrtMemcpy(d2, h2.size(), h2.data(), h2.size(), ACL_MEMCPY_HOST_TO_DEVICE), "h2d x2");
    CheckAcl(aclrtMemcpy(dy, hy.size(), hy.data(), hy.size(), ACL_MEMCPY_HOST_TO_DEVICE), "h2d y");

    TensorInfo ti1{s1.empty() ? nullptr : s1.data(), static_cast<int64_t>(s1.size()), dtype};
    TensorInfo ti2{s2.empty() ? nullptr : s2.data(), static_cast<int64_t>(s2.size()), dtype};
    TensorInfo tiy{sy.empty() ? nullptr : sy.data(), static_cast<int64_t>(sy.size()), dtype};
    TensorGroupInfo g1{&ti1, 1}, g2{&ti2, 1}, gy{&tiy, 1};

    run_kernel(reinterpret_cast<GM_ADDR>(d1), g1,
               reinterpret_cast<GM_ADDR>(d2), g2,
               reinterpret_cast<GM_ADDR>(dy), gy, cores, stream);
    CheckAcl(aclrtSynchronizeStream(stream), "sync");
    CheckAcl(aclrtMemcpy(hy.data(), hy.size(), dy, hy.size(), ACL_MEMCPY_DEVICE_TO_HOST), "d2h y");
    aclrtFree(d1); aclrtFree(d2); aclrtFree(dy);

    const auto st1 = Strides(s1, sy.size()), st2 = Strides(s2, sy.size());
    for (uint64_t idx = 0; idx < ny; ++idx) {
        uint64_t rem = idx, o1 = 0, o2 = 0;
        for (size_t k = 0; k < sy.size(); ++k) {
            size_t d = sy.size() - 1 - k;
            uint64_t dim = static_cast<uint64_t>(sy[d]);
            uint64_t c = rem % dim; rem /= dim;
            o1 += c * st1[d]; o2 += c * st2[d];
        }
        bool ok = false;
        if (dtype == 0) {
            float a, b, g;
            std::memcpy(&a, h1.data() + o1 * 4, 4);
            std::memcpy(&b, h2.data() + o2 * 4, 4);
            std::memcpy(&g, hy.data() + idx * 4, 4);
            ok = (g == a + b);
        } else if (dtype == 5) {
            int32_t a, b, g;
            std::memcpy(&a, h1.data() + o1 * 4, 4);
            std::memcpy(&b, h2.data() + o2 * 4, 4);
            std::memcpy(&g, hy.data() + idx * 4, 4);
            ok = (g == static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b)));
        } else if (dtype == 1) {
            uint16_t a, b, g;
            std::memcpy(&a, h1.data() + o1 * 2, 2);
            std::memcpy(&b, h2.data() + o2 * 2, 2);
            std::memcpy(&g, hy.data() + idx * 2, 2);
            ok = (g == FloatToFp16Bits(Fp16BitsToFloat(a) + Fp16BitsToFloat(b)));
        } else {
            uint16_t a, b, g;
            std::memcpy(&a, h1.data() + o1 * 2, 2);
            std::memcpy(&b, h2.data() + o2 * 2, 2);
            std::memcpy(&g, hy.data() + idx * 2, 2);
            ok = (g == FloatToBf16Bits(Bf16BitsToFloat(a) + Bf16BitsToFloat(b)));
        }
        if (!ok) {
            throw std::runtime_error(std::string("Mismatch on ") + name + " dtype=" + std::to_string(dtype));
        }
    }
    for (size_t b = ny * elemBytes; b < hy.size(); ++b) {
        if (hy[b] != 0xA5) throw std::runtime_error(std::string("Guard overwritten on ") + name);
    }
    std::printf("PASS %s dtype=%d N=%llu\n", name, dtype, static_cast<unsigned long long>(ny));
}
} // namespace

int main(int argc, char** argv) {
    const int device = (argc >= 2) ? std::atoi(argv[1]) : 0;
    const std::string filter = (argc >= 3) ? argv[2] : "";
    CheckAcl(aclInit(nullptr), "aclInit");
    CheckAcl(aclrtSetDevice(device), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
#ifdef ASCENDC_CPU_DEBUG
    int64_t cores = 1;
#else
    int64_t cores = 0;
    CheckAcl(aclrtGetDeviceInfo(device, ACL_DEV_ATTR_VECTOR_CORE_NUM, &cores), "getCores");
#endif

    struct Spec { const char* name; std::vector<int64_t> s1; std::vector<int64_t> s2; };
    const std::vector<Spec> specs = {
        {"same_unaligned", {8193}, {8193}},
        {"scalar_bcast", {4097}, {1}},
        {"row_bcast_k32", {512, 32}, {1, 32}},
        {"nd_both_contig", {3, 1, 5, 17}, {1, 4, 1, 17}},
        {"inner_scalar", {24, 70}, {24, 1}},
        {"packed_col_w3", {160, 3}, {160, 1}},
        {"packed_col_w31", {135, 31}, {135, 1}},
    };
    const int32_t dtypes[] = {0, 1, 2, 5};
    for (const auto& sp : specs) {
        if (!filter.empty() && filter != sp.name) continue;
        for (int32_t dt : dtypes) {
            RunCase(sp.name, dt, sp.s1, sp.s2, cores, stream);
        }
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(device);
    aclFinalize();
    return 0;
}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real CANN NPU/CPU multi-dtype & multi-path matrix for cannjudge_2524.")
    parser.add_argument("--mode", choices=("npu", "cpu"), default="npu")
    parser.add_argument("--npu-arch", default=os.environ.get("NPU_ARCH", "dav-2201"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--memcheck", action="store_true", help="Compile with sanitizer and run mssanitizer memcheck+racecheck")
    parser.add_argument("--profile", action="store_true", help="Run msprof op on row_bcast_k32")
    args = parser.parse_args()

    if "ASCEND_HOME_PATH" not in os.environ:
        print("ERROR: ASCEND_HOME_PATH not set. Source CANN set_env.sh first.", file=sys.stderr)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="cannjudge2524-matrix."))
    try:
        shutil.copy2(ROOT / "kernel.asc", tmp / "kernel.asc")
        shutil.copy2(ROOT / "CMakeLists.txt", tmp / "CMakeLists.txt")
        (tmp / "main.asc").write_text(MATRIX_MAIN_ASC, encoding="utf-8")

        build = tmp / f"build-{args.mode}"
        cmake_cmd = [
            "cmake", "-S", str(tmp), "-B", str(build),
            f"-DCMAKE_ASC_RUN_MODE={args.mode}",
            f"-DCMAKE_ASC_ARCHITECTURES={args.npu_arch}",
        ]
        if args.memcheck:
            cmake_cmd.append("-DCMAKE_ASC_FLAGS=--cce-enable-sanitizer -g")
        subprocess.run(cmake_cmd, check=True)
        subprocess.run(["cmake", "--build", str(build), "--parallel", "4"], check=True)

        binary = build / "sample"
        subprocess.run([str(binary), str(args.device_id)], check=True)

        if args.memcheck and args.mode == "npu":
            for tool in ("memcheck", "racecheck"):
                for target in ("inner_scalar", "packed_col_w3", "row_bcast_k32", "same_unaligned"):
                    print(f"=== mssanitizer --tool={tool} ({target}) ===")
                    res = subprocess.run(
                        ["mssanitizer", f"--tool={tool}", "--", str(binary), str(args.device_id), target],
                        capture_output=True, text=True, check=False,
                    )
                    print(res.stdout[-800:] + res.stderr[-400:])
                    if res.returncode != 0 or "No error detected" not in (res.stdout + res.stderr):
                        print(f"ERROR: mssanitizer {tool} reported issues on {target}", file=sys.stderr)
                        return 1

        if args.profile and args.mode == "npu":
            prof_out = ROOT / "reports" / "msprof_row_bcast_k32"
            prof_out.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "msprof", "op", f"--output={prof_out}",
                    "--aic-metrics=PipeUtilization,Memory",
                    str(binary), str(args.device_id), "row_bcast_k32",
                ],
                check=True,
            )
        print("NPU MATRIX VERIFICATION PASSED!")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
