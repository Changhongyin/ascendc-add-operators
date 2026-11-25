#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "kernel_operator.h"
#include "kernel_sim.inc"

namespace {

template <typename T>
float ToF32(T v);
template <> float ToF32<float>(float v) { return v; }
template <> float ToF32<half>(half v) { return AscendCSim::HalfToFloat(v); }
template <> float ToF32<bfloat16_t>(bfloat16_t v) { return AscendCSim::Bf16ToFloat(v); }

template <typename T>
T FromF32(float v);
template <> float FromF32<float>(float v) { return v; }
template <> half FromF32<half>(float v) { return AscendCSim::FloatToHalfRint(v); }
template <> bfloat16_t FromF32<bfloat16_t>(float v) { return AscendCSim::FloatToBf16Rint(v); }

uint64_t Prod(const std::vector<int64_t>& s)
{
    uint64_t n = 1;
    for (int64_t d : s) n *= static_cast<uint64_t>(d);
    return n;
}

float RefAct(float x, int32_t act)
{
    if (act == NdBmmOp::kActRelu) return x > 0.0f ? x : 0.0f;
    if (act == NdBmmOp::kActSilu) return x / (1.0f + std::exp(std::min(80.0f, -x)));
    if (act == NdBmmOp::kActGelu) {
        const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + std::tanh(u));
    }
    return x;
}

template <typename T>
void RunMatrixCase(const char* name,
                   int32_t dtype,
                   const std::vector<int64_t>& sA,
                   const std::vector<int64_t>& sB,
                   const std::vector<int64_t>& sC,
                   const std::vector<int64_t>& sY,
                   const NdBmmConfig& cfg,
                   int64_t cores,
                   float tol)
{
    const uint64_t nA = Prod(sA);
    const uint64_t nB = Prod(sB);
    const uint64_t nC = sC.empty() ? 0 : Prod(sC);
    const uint64_t nY = Prod(sY);
    constexpr size_t kGuard = 32;

    std::vector<T> hA(nA), hB(nB), hC(nC);
    for (uint64_t i = 0; i < nA; ++i) {
        hA[i] = FromF32<T>((static_cast<int32_t>((i * 17 + 3) % 31) - 15) * 0.0625f);
    }
    for (uint64_t i = 0; i < nB; ++i) {
        hB[i] = FromF32<T>((static_cast<int32_t>((i * 13 + 7) % 29) - 14) * 0.0625f);
    }
    for (uint64_t i = 0; i < nC; ++i) {
        hC[i] = FromF32<T>((static_cast<int32_t>((i * 11 + 5) % 23) - 11) * 0.125f);
    }

    std::vector<uint8_t> yRaw((nY + kGuard) * sizeof(T), 0xA5);

    NdBmmTensorDesc dA{sA.data(), static_cast<int64_t>(sA.size()), dtype};
    NdBmmTensorDesc dB{sB.data(), static_cast<int64_t>(sB.size()), dtype};
    NdBmmTensorDesc dC{sC.empty() ? nullptr : sC.data(), static_cast<int64_t>(sC.size()), dtype};
    NdBmmTensorDesc dY{sY.data(), static_cast<int64_t>(sY.size()), dtype};

    const bool ok = run_nd_bmm_fused(
        reinterpret_cast<GM_ADDR>(hA.data()), dA,
        reinterpret_cast<GM_ADDR>(hB.data()), dB,
        sC.empty() ? nullptr : reinterpret_cast<GM_ADDR>(hC.data()), sC.empty() ? nullptr : &dC,
        reinterpret_cast<GM_ADDR>(yRaw.data()), dY,
        cfg, cores, nullptr);
    if (!ok) {
        std::fprintf(stderr, "FAIL [%s] dispatch rejected dtype=%d\n", name, dtype);
        std::exit(1);
    }

    const uint64_t M = static_cast<uint64_t>(sA[sA.size() - 2]);
    const uint64_t K = static_cast<uint64_t>(sA[sA.size() - 1]);
    const uint64_t N = static_cast<uint64_t>(sY[sY.size() - 1]);
    const size_t yBatchRank = sY.size() - 2;
    const uint64_t batches = nY / (M * N);
    const bool rowBias = !sC.empty() && (sC.size() == 1 || sC[sC.size() - 2] == 1);

    auto batchStrides = [&](const std::vector<int64_t>& sh, uint64_t matElems) {
        std::vector<uint64_t> st(yBatchRank, 0);
        if (sh.empty() || sh.size() < 2) return st;
        const size_t br = sh.size() - 2;
        uint64_t acc = matElems;
        for (size_t k = 0; k < br; ++k) {
            const size_t idx = br - 1 - k;
            if (sh[idx] > 1) {
                st[yBatchRank - 1 - k] = acc;
                acc *= static_cast<uint64_t>(sh[idx]);
            }
        }
        return st;
    };

    const auto stA = batchStrides(sA, M * K);
    const auto stB = batchStrides(sB, K * N);
    const auto stC = batchStrides(sC, rowBias ? N : M * N);
    const auto* yOut = reinterpret_cast<const T*>(yRaw.data());

    for (uint64_t b = 0; b < batches; ++b) {
        uint64_t rem = b, aOff = 0, bOff = 0, cOff = 0;
        for (size_t k = 0; k < yBatchRank; ++k) {
            const size_t d = yBatchRank - 1 - k;
            const uint64_t dim = static_cast<uint64_t>(sY[d]);
            const uint64_t coord = rem % dim;
            rem /= dim;
            aOff += coord * stA[d];
            bOff += coord * stB[d];
            cOff += coord * stC[d];
        }
        for (uint64_t m = 0; m < M; ++m) {
            for (uint64_t n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (uint64_t k = 0; k < K; ++k) {
                    const float av = ToF32(hA[aOff + m * K + k]);
                    const float bv = cfg.transB ? ToF32(hB[bOff + n * K + k]) : ToF32(hB[bOff + k * N + n]);
                    acc += av * bv;
                }
                float val = cfg.alpha * acc;
                if (!sC.empty()) {
                    val += cfg.beta * ToF32(hC[cOff + (rowBias ? n : m * N + n)]);
                }
                val = RefAct(val, cfg.actMode);
                const float expRounded = ToF32(FromF32<T>(val));
                const float got = ToF32(yOut[b * M * N + m * N + n]);
                const float diff = std::fabs(got - expRounded);
                if (diff > tol + tol * std::fabs(expRounded)) {
                    std::fprintf(stderr, "FAIL [%s] dtype=%d cores=%lld b=%llu m=%llu n=%llu exp=%g got=%g\n",
                                 name, dtype, static_cast<long long>(cores),
                                 static_cast<unsigned long long>(b),
                                 static_cast<unsigned long long>(m),
                                 static_cast<unsigned long long>(n), expRounded, got);
                    std::exit(1);
                }
            }
        }
    }

    for (size_t i = nY * sizeof(T); i < yRaw.size(); ++i) {
        if (yRaw[i] != 0xA5) {
            std::fprintf(stderr, "FAIL [%s] GUARD OVERWRITE dtype=%d byte=%zu\n", name, dtype, i);
            std::exit(1);
        }
    }
}

struct MatrixCase {
    const char* name;
    std::vector<int64_t> sA;
    std::vector<int64_t> sB;
    std::vector<int64_t> sC;
    std::vector<int64_t> sY;
    NdBmmConfig cfg;
};

} // namespace

int main()
{
    const std::vector<MatrixCase> cases = {
        // 2D GEMM (aligned, prime unaligned, multi-tile M/N/K, transB, bias, all activations)
        {"2d_tiny_prime", {7, 13}, {13, 9}, {}, {7, 9}, {false, NdBmmOp::kActNone, 1.0f, 0.0f}},
        {"2d_gemm_relu", {19, 73}, {73, 70}, {70}, {19, 70}, {false, NdBmmOp::kActRelu, 1.0f, 1.0f}},
        {"2d_linear_transb_silu", {33, 65}, {71, 65}, {1, 71}, {33, 71}, {true, NdBmmOp::kActSilu, 0.75f, 0.5f}},
        {"2d_residual_gelu", {17, 64}, {64, 65}, {17, 65}, {17, 65}, {false, NdBmmOp::kActGelu, 1.0f, 1.0f}},

        // 3D Batched MatMul (same batch + broadcast batch)
        {"3d_bmm_basic", {4, 17, 33}, {4, 33, 29}, {}, {4, 17, 29}, {false, NdBmmOp::kActNone, 1.0f, 0.0f}},
        {"3d_bmm_bcast_b_silu", {3, 18, 66}, {1, 66, 65}, {1, 1, 65}, {3, 18, 65}, {false, NdBmmOp::kActSilu, 1.0f, 1.0f}},

        // 4D Multi-Head Attention Q @ K^T * scale + broadcast mask [B, 1, S, S]
        {"4d_mha_qkt_mask", {2, 3, 19, 35}, {2, 3, 21, 35}, {2, 1, 19, 21}, {2, 3, 19, 21},
         {true, NdBmmOp::kActNone, 0.125f, 1.0f}},
        {"4d_mha_av_gelu", {2, 3, 19, 21}, {2, 3, 21, 66}, {1, 3, 1, 66}, {2, 3, 19, 66},
         {false, NdBmmOp::kActGelu, 1.0f, 1.0f}},

        // 5D Grouped-Query Attention (GQA): [B, G, H_g, S, D] x [B, G, 1, S, D]^T
        {"5d_gqa_qkt", {2, 2, 3, 17, 33}, {2, 2, 1, 19, 33}, {1, 2, 1, 17, 19}, {2, 2, 3, 17, 19},
         {true, NdBmmOp::kActRelu, 0.25f, 0.5f}},

        // 6D & 8D high-dimensional bidirectional broadcasted tensor contractions
        {"6d_bidir_bcast_silu", {2, 1, 3, 1, 9, 17}, {1, 2, 1, 2, 17, 15}, {1, 2, 3, 1, 1, 15},
         {2, 2, 3, 2, 9, 15}, {false, NdBmmOp::kActSilu, 1.0f, 1.0f}},
        {"8d_tensor_contraction", {2, 1, 2, 1, 2, 1, 5, 11}, {1, 2, 1, 2, 1, 2, 7, 11}, {7},
         {2, 2, 2, 2, 2, 2, 5, 7}, {true, NdBmmOp::kActGelu, 1.0f, 1.0f}},
    };

    const int64_t coreCounts[] = {1, 7, 40, 48};
    uint64_t totalRuns = 0;

    for (const auto& c : cases) {
        for (int64_t cores : coreCounts) {
            RunMatrixCase<float>(c.name, NdBmmOp::kDtypeFloat32, c.sA, c.sB, c.sC, c.sY, c.cfg, cores, 1e-4f);
            RunMatrixCase<half>(c.name, NdBmmOp::kDtypeFloat16, c.sA, c.sB, c.sC, c.sY, c.cfg, cores, 2e-2f);
            RunMatrixCase<bfloat16_t>(c.name, NdBmmOp::kDtypeBfloat16, c.sA, c.sB, c.sC, c.sY, c.cfg, cores, 4e-2f);
            totalRuns += 3;
        }
    }

    // Negative dispatch validation checks (must cleanly return false)
    {
        int64_t badA[2] = {4, 5}, badB[2] = {6, 4}, badY[2] = {4, 4};
        NdBmmTensorDesc dA{badA, 2, 0}, dB{badB, 2, 0}, dY{badY, 2, 0};
        uint8_t dummy[64]{};
        assert(!run_nd_bmm_fused(dummy, dA, dummy, dB, nullptr, nullptr, dummy, dY, {}, 40, nullptr));
        int64_t badRank[9] = {1, 1, 1, 1, 1, 1, 1, 2, 2};
        NdBmmTensorDesc d9{badRank, 9, 0};
        assert(!run_nd_bmm_fused(dummy, d9, dummy, d9, nullptr, nullptr, dummy, d9, {}, 40, nullptr));
        ++totalRuns;
    }

    std::printf("PASS all %llu high-D batched matmul fusion cases (2D..8D x 3 dtypes x 4 core counts + negative checks)\n",
                static_cast<unsigned long long>(totalRuns));
    return 0;
}
