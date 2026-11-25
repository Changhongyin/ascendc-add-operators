# Ascend C 算子整理汇总

三个基于华为昇腾 **Ascend C** 的向量/矩阵算子工程，覆盖从同形 FP32 逐元素加法、多类型高维广播加法到 **2D–8D 高维广播批量矩阵乘融合算子（BMM + Bias/Residual + Activation）**，并附带**真机诊断、验收与全仓离线/云端自动化测试套件**。

| 子项目 | 内容 | 状态 |
|---|---|---|
| `01-op179089-fp32-add` | 同形连续 FP32 逐元素加法，1D–8D | Atlas A3 真机 **100% 验收通过**（20/20 SHA-256 校验通过） |
| `02-cannjudge-2524-broadcast-add` | 通用多类型、任意维 NumPy 广播加法（含 3D–12D 批量矩阵行偏置 UB 复用快路径） | 竞态修复 + 行广播/高维矩阵偏置 UB 平铺优化；本地 **706 组**多类型/多路径回归全通过 |
| `03-nd-batched-matmul-fusion` | **2D–8D 高维广播批量矩阵乘融合算子**（`alpha·A@B^T + beta·C -> ReLU/SiLU/GELU`） | 支持 FP32/FP16/BF16、MHA/GQA/8D 张量缩并；本地 **133 组**严格模拟回归全通过 |

> 本仓库是个人学习与测试过程的整理归档，用于记录 Ascend C 向量算子的排错、优化与验证方法，**不是生产级算子库**。

---

## 目录结构

```text
.
├── README.md                                  # 本文件
├── run_all_local_tests.py                     # 全仓离线一键验证入口（子项目一 + 子项目二）
├── 算子整合总结文档.md                          # 项目全景、技术总结、性能数据、溯源清单
├── 01-op179089-fp32-add/                      # FP32 逐元素 Add（Atlas A3 验收版）
│   ├── op179089.asc                           # 验收版源码（主入口 + AddKernel）
│   ├── CMakeLists.txt / build.sh              # CANN 9.0 构建脚本
│   ├── baseline/ next/ op179089.patch         # 基线、最终版、统一补丁
│   ├── SHA256SUMS.txt                         # 20 项核心文件校验和
│   ├── scripts/                               # 回归套件、msprof 解析、Tile 建模、安全合入
│   ├── tests/                                 # Mock 头文件 + 5 阶段本地回归
│   └── reports/                               # 诊断报告、A3 验收报告、memcheck 原始日志、msprof 数据
├── 02-cannjudge-2524-broadcast-add/           # 通用多类型广播 Add（CANN Judge #2524 优化测试版）
│   ├── kernel.asc                             # 多类型通用广播 Add 内核（含 3D–26D 批量矩阵行偏置 UB 复用）
│   ├── main.asc                               # ACL 运行入口与张量 I/O（保持官方模板不变）
│   ├── CMakeLists.txt / run.sh                # NPU / CPU 双模式编译调测（保持官方模板不变）
│   ├── sample.py / reference.py               # 造数、NumPy 参考实现与校验（保持官方模板不变）
│   └── tests/                                 # 本地 706 组模拟回归（含 V->S 竞态/EventID/UB 断言）与云端 NPU 矩阵脚本
├── 03-nd-batched-matmul-fusion/               # 2D–8D 高维广播批量矩阵乘融合算子（BMM + Epilogue）
│   ├── kernel.asc                             # 16x64x64 分块外积 GEMM + 6D Batch 广播 + 融合 Bias/SiLU/GELU/ReLU
│   ├── main.asc                               # ACL 运行入口与 GFLOP/s 基准测试
│   ├── CMakeLists.txt / build.sh              # CANN 9.0 构建脚本（dav-2201 / dav-3510）
│   ├── README.md                              # 算子规格、UB 预算与运行指南
│   └── tests/                                 # 本地 133 组 2D–8D 严格模拟回归与云端 NPU/memcheck/msprof 脚本
└── 99-archive/                                # 全部原始压缩包与交接文档
```

---

## 子项目一：`op179089` FP32 逐元素加法

- 支持 1D–8D 同形连续张量，元素总数上限 `33,554,432`（单张量 128 MB）。
- Tile `4096` 元素，UB 双缓冲 `3 × 2 × 4096 × 4B = 96 KB`。
- 多核按 32 B 对齐组均分，尾核单独处理。

### 解决的核心问题

CANN 9.0.0 下 `AscendC::Add(z, x, y, count)` 走 Counter-Mode `vadd`。当 `count > 64` 且 `count % 64 != 0` 时，`mssanitizer --tool=memcheck` 会报 `illegal read/write on UB` —— 尽管数值结果正确。

修复方式是在 UB 内按 256 B（64 元素）repeat 边界把向量指令拆成两段：

```cpp
const uint32_t wholeRepeats = count - (count % kVectorRepeatElements);
if (wholeRepeats != 0) {
    AscendC::Add(z, x, y, wholeRepeats);
}
const uint32_t tailElements = count - wholeRepeats;
if (tailElements != 0) {
    AscendC::Add(z[wholeRepeats], x[wholeRepeats], y[wholeRepeats], tailElements);
}
```

零额外 DMA、零额外 Tile，拆分偏移天然满足 256 B 对齐，性能无回退。

### 验收结果（Atlas A3, CANN 9.0.0, dav-2201, 40 核）

- 本地回归 `tests/test_local_all.py`：**5/5 阶段通过**
- 真机 `mssanitizer`：**11/11 用例显式输出 `No error detected.`**
- 真机全量回归：25/25 功能、11/11 非法参数、4/4 边界、10/10 基准全部 PASS

| 元素数 | 流内 Median | 有效带宽 | msprof Task Duration | `aiv_mte2_ratio` |
|---:|---:|---:|---:|---:|
| 16,777,216 | 184.981 µs | 1088.36 GB/s | 142.517 µs | 98.09% |
| 33,554,432 | 372.981 µs | 1079.56 GB/s | 337.253 µs | 98.95% |

*(有效带宽按两读一写 `3 × N × 4 B / median` 计算；独立复测 16M 达 181.151 µs / 1111.37 GB/s。)*

### 构建与验证

```bash
cd "01-op179089-fp32-add"

shasum -a 256 -c SHA256SUMS.txt          # 完整性校验

NPU_ARCH=dav-2201 bash build.sh          # Release 构建
./build/op179089 --shape 2x2x2x2x2x2x2x3
python3 scripts/run_npu_suite.py --binary build/op179089 --profile

python3 tests/test_local_all.py          # 无 NPU 也可跑的本地回归

cmake -S . -B build-memcheck -DNPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=Release \
      "-DCMAKE_ASC_FLAGS=--cce-enable-sanitizer -g"
cmake --build build-memcheck --parallel
mssanitizer --tool=memcheck -- ./build-memcheck/op179089 8193
```

---

## 子项目二：`cannjudge_2524` 通用 Add 广播算子

在官方模板基础上，把 `kernel.asc` 从 36 行 TODO 骨架实现为 **1392 行** 多路径算子：

- **数据类型**：`float32` / `float16` / `bfloat16` / `int32`
- **广播语义**：NumPy 风格任意高维，Host 侧 Shape Coalescing 后最多 26 维，元素上限 `2^26`
- **5 条调度路径**：
  1. 同形连续 —— 4096 元素双缓冲，打满 HBM 带宽
  2. 标量广播 —— 单侧 1 元素时提前 `Duplicate`，每 Tile 仅 1 读 1 写
  3. 2D 行/偏置广播 —— 行偏置常驻 UB，消除重复读
  4. 通用 ND 双连续广播 —— 按 32 B 对齐切段流式搬运
  5. 内维标量广播 —— 细长列场景走 `RunPackedColumn` 打包列 + `Gather` 展开
- **安全原语与竞态修复**：`SafeAdd4B` / `SafeCast` / `SafeGather` 统一按 256 B Repeat 边界拆分；`ReadScalar2BAsFloat` 使用显式 `HardEvent::V_S` + `HardEvent::S_V` 同步并释放 EventID，消除 2B 标量读取竞态与事件池耗尽风险；`RowBcast` 对小 `K` 偏置在 UB 内对数倍增平铺（如 `(512,32)+(1,32)` 的 DMA 次数由 1025 降至 16）；通用广播热循环采用 O(1) 里程表步进消除逐行 64 位除法
- **精度**：FP16/BF16 在 UB 内升精度到 FP32 计算，再以 `CAST_RINT` 回转

### 构建与调测

```bash
# 全仓离线一键验证（无需 NPU，同时校验子项目一 SHA-256、子项目二 706 组与子项目三 133 组回归）
python3 run_all_local_tests.py

cd "02-cannjudge-2524-broadcast-add"
python3 tests/test_local_all.py    # 子项目二离线 706 组回归（含 32B/UB/Repeat/EventID/V->S 竞态断言）

source /path/to/cann/set_env.sh
bash run.sh npu                    # 官方模板单样例真机验证
bash run.sh cpu                    # 无设备时的 CANN CPU 调测模式（不可测性能）
python3 tests/run_npu_matrix.py --mode npu --memcheck --profile   # 云端多类型/多路径 + memcheck/racecheck + msprof
```

---

## 子项目三：`03-nd-batched-matmul-fusion` 高维广播批量矩阵乘融合算子

将算子核心能力从逐元素加法进一步扩展至**复杂高维矩阵计算与融合后处理**（覆盖 2D GEMM、3D BMM、4D Multi-Head Attention `QK^T + Mask`、5D Grouped-Query Attention GQA 及 6D–8D 高维张量缩并）：

- **公式**：`Y = Act(alpha * (A @ B^{T?}) + beta * C)`，其中 `A/B/C` 支持最高 **8D（6 个 Batch 维 + 2 个矩阵维）** 的任意 NumPy 跨 Batch 广播。
- **多精度与数值安全**：支持 `float32` / `float16` / `bfloat16`，片上统一 FP32 累加 + `CAST_RINT` 回转；支持 `NONE` / `RELU` / `SILU` / `GELU(tanh)` 四种片上融合激活及无偏置 / 行偏置 `[..., 1, N]` / 完整或广播残差矩阵 `[..., M, N]`。
- **分块与 UB 设计**：`16 × 64 × 64` 外积分块（`kTileN = 64` 严格对齐 256 B Repeat），UB 占用 `36–52 KB`（远低于 192 KB），按 `(batch, tileM, tileN)` 细粒度切分至全核，显式 `V_S`/`S_V` 同步与非对齐尾块零填充。

```bash
cd "03-nd-batched-matmul-fusion"
python3 tests/test_local_all.py    # 离线 133 组 2D–8D 严格模拟回归
NPU_ARCH=dav-2201 bash build.sh && ./build/nd_bmm_fused
python3 tests/run_npu_suite.py --npu-arch dav-2201 --memcheck --profile
```

---

## 环境要求

- CANN Toolkit **9.0.0**（Ascend C）
- 昇腾 Atlas A3，`NPU_ARCH=dav-2201`（子项目二可用 CPU 调测模式）
- CMake ≥ 3.16、Python 3.8+、NumPy

## 文档索引

| 文档 | 说明 |
|---|---|
| [`算子整合总结文档.md`](算子整合总结文档.md) | 项目全景、横向对比、性能数据、原始文件溯源 |
| [`reports/0001-ub-access-diagnosis.md`](01-op179089-fp32-add/reports/0001-ub-access-diagnosis.md) | UB 越界报错的根因推导过程 |
| [`reports/0002-atlas-a3-verification-report.md`](01-op179089-fp32-add/reports/0002-atlas-a3-verification-report.md) | Atlas A3 真机验收报告 |
| [`reports/npu-suite/report.md`](01-op179089-fp32-add/reports/npu-suite/report.md) | msprof / 回归套件汇总 |

## 说明

- 所有真机数据采自个人 Atlas A3 环境，`reports/` 中的 `.bin` / `.o` 为 `msprof` 原始产物，仅作证据留存。
- 子项目二的 `main.asc`、`CMakeLists.txt`、`run.sh`、`sample.py`、`reference.py`、`requirements.txt` 与官方题目模板保持一致，仅 `kernel.asc` 为本人实现。

## License

[MIT](LICENSE)
