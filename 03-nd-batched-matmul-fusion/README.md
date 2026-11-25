# `03-nd-batched-matmul-fusion`：高维广播批量矩阵乘融合算子（Ascend C）

面向华为昇腾 **CANN 9.0 / Atlas A2·A3·A5** 的高维（2D–8D）广播批量矩阵乘法与后处理融合算子：

$$Y[b_0,\dots,b_{r-1}, m, n] = \operatorname{Act}\!\left(\alpha \cdot A[b_A, m, :] \cdot B^{T?}[b_B, :, n] + \beta \cdot C[b_C, m/1, n]\right)$$

## 核心能力

- **高维广播张量矩阵计算**：支持 **2D–8D**（最多 6 个 Batch 维度 + 2 个矩阵维度）的 NumPy 风格跨 Batch 广播，天然覆盖：
  - 2D GEMM + Bias + 激活（FFN / SwiGLU / Linear）
  - 4D Multi-Head Attention `Q @ K^T * scale + attn_mask`（支持 `transB` 与 `[B, 1, S, S]` Mask 广播）
  - 5D Grouped-Query Attention（GQA `[B, G, H_g, S, D] × [B, G, 1, S, D]^T`）
  - 6D–8D 任意高维张量缩并（Tensor Contraction）
- **多精度支持**：`float32` / `float16` / `bfloat16`，片上统一 **FP32 累加** + `CAST_RINT` 回转。
- **片上融合 Epilogue（零额外 GM 往返）**：支持 `alpha` 缩放、`beta` 乘加（无偏置 / 1D & 高维行偏置 `[..., 1, N]` / 完整或广播残差矩阵 `[..., M, N]`）及 `NONE` / `RELU` / `SILU` / `GELU(tanh)` 激活。
- **分块与安全设计**：
  - `16 × 64 × 64` 分块外积流水，`kTileN = 64` FP32 严格对齐 256 B 向量 Repeat，UB 总占用 `36–52 KB`（远低于 192 KB 上限）。
  - 按 `(batch, tileM, tileN)` 细粒度均匀切分到全部 AI Core（40/48/64 核均可负载均衡）。
  - 非对齐尾块使用 `DataCopyPad` + 片上零填充，显式 `HardEvent::V_S` / `S_V` 与 `ReleaseEventID`，满足 `memcheck` / `racecheck` / `initcheck` 规范。

## 本地离线验证（无需 NPU）

```bash
python3 tests/test_local_all.py
```

## Atlas A3 真机编译与验证

```bash
NPU_ARCH=dav-2201 bash build.sh
./build/nd_bmm_fused
./build/nd_bmm_fused --benchmark
python3 tests/run_npu_suite.py --npu-arch dav-2201 --memcheck --profile
```
