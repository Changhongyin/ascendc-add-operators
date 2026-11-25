#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"

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

// Generated simulated kernel (triple-chevron launches lowered to multi-block CPU loop).
#include "kernel_sim.inc"

namespace {

std::vector<int64_t> BroadcastShape(const std::vector<int64_t>& a, const std::vector<int64_t>& b)
{
    const size_t rank = std::max(a.size(), b.size());
    std::vector<int64_t> out(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t da = (i + a.size() >= rank) ? a[i + a.size() - rank] : 1;
        const int64_t db = (i + b.size() >= rank) ? b[i + b.size() - rank] : 1;
        assert(da == db || da == 1 || db == 1);
        out[i] = std::max(da, db);
    }
    return out;
}

uint64_t NumElems(const std::vector<int64_t>& s)
{
    uint64_t n = 1;
    for (int64_t d : s) {
        n *= static_cast<uint64_t>(d);
    }
    return n;
}

std::vector<uint64_t> ComputeStrides(const std::vector<int64_t>& shape, size_t targetRank)
{
    std::vector<uint64_t> strides(targetRank, 0);
    uint64_t stride = 1;
    for (size_t k = 0; k < shape.size(); ++k) {
        const size_t idx = shape.size() - 1 - k;
        const size_t outIdx = targetRank - 1 - k;
        if (shape[idx] > 1) {
            strides[outIdx] = stride;
            stride *= static_cast<uint64_t>(shape[idx]);
        }
    }
    return strides;
}

std::string ShapeStr(const std::vector<int64_t>& s)
{
    if (s.empty()) {
        return "()";
    }
    std::string out = "(";
    for (size_t i = 0; i < s.size(); ++i) {
        if (i > 0) {
            out += ",";
        }
        out += std::to_string(s[i]);
    }
    out += ")";
    return out;
}

template <typename T>
float ToFloatRef(T v);
template <>
float ToFloatRef<float>(float v) { return v; }
template <>
float ToFloatRef<half>(half v) { return AscendCSim::HalfToFloat(v); }
template <>
float ToFloatRef<bfloat16_t>(bfloat16_t v) { return AscendCSim::Bf16ToFloat(v); }

template <typename T>
T FromFloatRef(float v);
template <>
float FromFloatRef<float>(float v) { return v; }
template <>
half FromFloatRef<half>(float v) { return AscendCSim::FloatToHalfRint(v); }
template <>
bfloat16_t FromFloatRef<bfloat16_t>(float v) { return AscendCSim::FloatToBf16Rint(v); }

template <typename T>
T MakeTestVal(uint64_t idx, int seed, bool specialEdges)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        if (specialEdges) {
            static const int32_t kEdgeI32[] = {
                std::numeric_limits<int32_t>::max(),
                std::numeric_limits<int32_t>::min(),
                1,
                -1,
                0,
                123456789,
                -987654321,
                2147483640,
            };
            if (idx < 8) {
                return kEdgeI32[(idx + static_cast<uint64_t>(seed)) % 8];
            }
        }
        return static_cast<int32_t>((static_cast<int64_t>(idx) * (seed == 0 ? 131 : -97) + seed * 17) % 100000 - 50000);
    } else {
        if (specialEdges && idx < 6) {
            static const float kEdgeF[] = {
                0.0f,
                -0.0f,
                std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                3.5f,
            };
            return FromFloatRef<T>(kEdgeF[(idx + static_cast<uint64_t>(seed)) % 6]);
        }
        const float v = (static_cast<int32_t>((idx * 37 + static_cast<uint64_t>(seed) * 19) % 257) - 128) * 0.125f;
        return FromFloatRef<T>(v);
    }
}

template <typename T>
T RefAdd(T a, T b)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        return static_cast<int32_t>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
    } else {
        return FromFloatRef<T>(ToFloatRef<T>(a) + ToFloatRef<T>(b));
    }
}

template <typename T>
bool EqualElem(T got, T exp)
{
    if constexpr (std::is_same_v<T, int32_t>) {
        return got == exp;
    } else {
        const float fg = ToFloatRef<T>(got);
        const float fe = ToFloatRef<T>(exp);
        if (std::isnan(fg) && std::isnan(fe)) {
            return true;
        }
        return fg == fe;
    }
}

template <typename T>
void RunTypedCase(const char* tag,
                  int32_t dtypeCode,
                  const std::vector<int64_t>& s1,
                  const std::vector<int64_t>& s2,
                  int64_t cores,
                  bool specialEdges = false)
{
    const std::vector<int64_t> sy = BroadcastShape(s1, s2);
    const uint64_t n1 = NumElems(s1);
    const uint64_t n2 = NumElems(s2);
    const uint64_t ny = NumElems(sy);
    constexpr size_t kGuardElems = 32;

    std::vector<T> x1(n1);
    std::vector<T> x2(n2);
    for (uint64_t i = 0; i < n1; ++i) {
        x1[i] = MakeTestVal<T>(i, 0, specialEdges);
    }
    for (uint64_t i = 0; i < n2; ++i) {
        x2[i] = MakeTestVal<T>(i, 1, specialEdges);
    }

    // Fill output + guard region with deterministic poison pattern.
    std::vector<uint8_t> yRaw((ny + kGuardElems) * sizeof(T), 0xA5);

    TensorInfo ti1{s1.empty() ? nullptr : s1.data(), static_cast<int64_t>(s1.size()), dtypeCode};
    TensorInfo ti2{s2.empty() ? nullptr : s2.data(), static_cast<int64_t>(s2.size()), dtypeCode};
    TensorInfo tiy{sy.empty() ? nullptr : sy.data(), static_cast<int64_t>(sy.size()), dtypeCode};
    TensorGroupInfo g1{&ti1, 1};
    TensorGroupInfo g2{&ti2, 1};
    TensorGroupInfo gy{&tiy, 1};

    run_kernel(reinterpret_cast<GM_ADDR>(x1.data()), g1,
               reinterpret_cast<GM_ADDR>(x2.data()), g2,
               reinterpret_cast<GM_ADDR>(yRaw.data()), gy,
               cores, nullptr);

    const auto* yOut = reinterpret_cast<const T*>(yRaw.data());
    const size_t rank = sy.size();
    const auto st1 = ComputeStrides(s1, rank);
    const auto st2 = ComputeStrides(s2, rank);

    for (uint64_t idx = 0; idx < ny; ++idx) {
        uint64_t rem = idx;
        uint64_t o1 = 0;
        uint64_t o2 = 0;
        for (size_t k = 0; k < rank; ++k) {
            const size_t d = rank - 1 - k;
            const uint64_t dim = static_cast<uint64_t>(sy[d]);
            const uint64_t coord = rem % dim;
            rem /= dim;
            o1 += coord * st1[d];
            o2 += coord * st2[d];
        }
        const T expected = RefAdd(x1[o1], x2[o2]);
        if (!EqualElem(yOut[idx], expected)) {
            std::fprintf(stderr, "FAIL [%s] dtype=%d %s+%s cores=%lld at idx=%llu\n",
                         tag, dtypeCode, ShapeStr(s1).c_str(), ShapeStr(s2).c_str(),
                         static_cast<long long>(cores), static_cast<unsigned long long>(idx));
            std::exit(1);
        }
    }

    // Verify guard bytes after logical output were untouched.
    for (size_t b = ny * sizeof(T); b < yRaw.size(); ++b) {
        if (yRaw[b] != 0xA5) {
            std::fprintf(stderr, "FAIL [%s] GUARD OVERWRITE dtype=%d %s+%s byte=%zu\n",
                         tag, dtypeCode, ShapeStr(s1).c_str(), ShapeStr(s2).c_str(), b);
            std::exit(1);
        }
    }
}

struct ShapeCase {
    const char* tag;
    std::vector<int64_t> s1;
    std::vector<int64_t> s2;
    bool specialEdges;
};

} // namespace

int main()
{
    const std::vector<ShapeCase> cases = {
        // Path 1: Same-shape contiguous (0D, 1D unaligned, repeat boundary, multi-tile, 4D)
        {"same_0d", {}, {}, false},
        {"same_1", {1}, {1}, false},
        {"same_7", {7}, {7}, true},
        {"same_65", {65}, {65}, false},
        {"same_4097", {4097}, {4097}, false},
        {"same_10000", {10000}, {10000}, false},
        {"same_4d", {1, 4, 2, 2}, {1, 4, 2, 2}, false},
        {"same_edges", {16}, {16}, true},

        // Path 2: Scalar broadcast (both orders, unaligned, multi-tile, edges)
        {"scalar_rhs", {4097}, {1}, false},
        {"scalar_lhs", {1}, {4097}, false},
        {"scalar_0d_rhs", {3, 19}, {}, true},
        {"scalar_0d_lhs", {}, {3, 19}, true},
        {"scalar_multi_tile", {10005}, {1}, false},

        // Path 2b: UB-cached & tiled 2D Row broadcast (small K, non-pow2 K, max K=4096, both orders)
        {"row_k8_4b", {256, 8}, {1, 8}, false},
        {"row_k16", {300, 16}, {1, 16}, false},
        {"row_k32", {256, 32}, {1, 32}, false},
        {"row_k32_lhs", {1, 32}, {256, 32}, false},
        {"row_k48", {100, 48}, {1, 48}, false},
        {"row_k128", {64, 128}, {1, 128}, false},
        {"row_k4096", {3, 4096}, {1, 4096}, false},
        {"row_3d_coalesced", {4, 16, 32}, {1, 1, 32}, false},

        // Path 3a: General ND broadcast (both inner contiguous, unaligned K, multi-dim, >4096 K)
        {"nd_both_unaligned", {17, 13}, {1, 13}, false},
        {"nd_both_3d", {5, 7, 19}, {1, 7, 19}, false},
        {"nd_both_4d_bidir", {3, 1, 5, 17}, {1, 4, 1, 17}, false},
        {"nd_both_large_k", {3, 5000}, {1, 5000}, false},
        {"nd_both_high_dim", {2, 1, 3, 1, 2, 1, 2, 9}, {1, 2, 1, 2, 1, 3, 1, 9}, false},
        // Path 3a high-D Batched Matrix Row-Reuse fast path (3D/4D/5D/8D/12D Transformer & tensor bias)
        {"nd_bmat_3d_rhs", {4, 64, 32}, {4, 1, 32}, false},
        {"nd_bmat_3d_lhs", {4, 1, 32}, {4, 64, 32}, false},
        {"nd_mha_4d_bias", {2, 4, 48, 32}, {1, 4, 1, 32}, false},
        {"nd_mha_4d_bias_lhs", {1, 4, 1, 64}, {2, 4, 33, 64}, false},
        {"nd_5d_tensor_bias", {2, 3, 2, 17, 16}, {2, 1, 2, 1, 16}, false},
        {"nd_12d_alternating", {2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 16}, {1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 16}, false},

        // Path 3b: Inner scalar broadcast (general + V->S stress across many rows)
        {"inner_scl_rhs", {17, 70}, {17, 1}, true},
        {"inner_scl_lhs", {17, 1}, {17, 70}, true},
        {"inner_scl_large_k", {3, 5, 4100}, {3, 1, 1}, false},
        {"inner_scl_bidir", {3, 1, 19}, {1, 5, 1}, false},
        {"inner_scl_many_rows", {40, 35}, {40, 1}, false},

        // Path 3b fast path: RunPackedColumn + SafeGather (various widths, odd widths, swapped orders)
        {"packed_col_w3", {160, 3}, {160, 1}, false},
        {"packed_col_w3_lhs", {160, 1}, {160, 3}, false},
        {"packed_col_w8", {150, 8}, {150, 1}, false},
        {"packed_col_w16", {200, 16}, {200, 1}, false},
        {"packed_col_w31", {135, 31}, {135, 1}, false},
        {"packed_col_w32", {180, 32}, {180, 1}, false},
        {"packed_col_unaligned_rows", {137, 7}, {137, 1}, false},
    };

    const int64_t coreCounts[] = {1, 7, 40, 48};
    uint64_t totalRuns = 0;

    for (const auto& c : cases) {
        for (int64_t cores : coreCounts) {
            RunTypedCase<float>(c.tag, AddOp::kDtypeFloat32, c.s1, c.s2, cores, c.specialEdges);
            RunTypedCase<half>(c.tag, AddOp::kDtypeFloat16, c.s1, c.s2, cores, c.specialEdges);
            RunTypedCase<bfloat16_t>(c.tag, AddOp::kDtypeBfloat16, c.s1, c.s2, cores, c.specialEdges);
            RunTypedCase<int32_t>(c.tag, AddOp::kDtypeInt32, c.s1, c.s2, cores, c.specialEdges);
            totalRuns += 4;
        }
    }

    // Verify RowBcast multi-row UB tiling drastically reduces DMA call count on narrow K=32 rows!
    AscendCSim::gDataCopyCalls = 0;
    RunTypedCase<float>("row_tiling_dma_check", AddOp::kDtypeFloat32, {512, 32}, {1, 32}, 1, false);
    const uint64_t dmaCalls = AscendCSim::gDataCopyCalls;
    assert(dmaCalls <= 24 && "RowBcast UB multi-row tiling did not coalesce DMA calls!");
    ++totalRuns;

    // Verify 4D Multi-Head Attention batched matrix row-bias reuse ([2,4,64,32] + [1,4,1,32]):
    // Without batched row reuse: 2*4*64 = 512 rows * 3 DMAs = 1536 calls!
    // With RunBatchedRowReuse: 8 batch slices, each 64*32=2048 elems (1 tile) -> ~80 calls!
    AscendCSim::gDataCopyCalls = 0;
    RunTypedCase<float>("mha_4d_row_reuse_dma_check", AddOp::kDtypeFloat32, {2, 4, 64, 32}, {1, 4, 1, 32}, 1, false);
    const uint64_t mhaDmaCalls = AscendCSim::gDataCopyCalls;
    assert(mhaDmaCalls <= 96 && "4D batched matrix row-reuse fast path did not coalesce DMA calls!");
    ++totalRuns;

    std::printf("PASS all %llu broadcast-add cases (4 dtypes x %zu shapes x 4 core counts + RowBcast=%llu / 4D-MHA=%llu DMAs)\n",
                static_cast<unsigned long long>(totalRuns), cases.size(),
                static_cast<unsigned long long>(dmaCalls),
                static_cast<unsigned long long>(mhaDmaCalls));
    return 0;
}
