# Ascend C Add 算子整理汇总

两个基于华为昇腾 **Ascend C** 的 Add 算子工程，以及配套的**真机诊断、验收与性能数据**。

| 子项目 | 内容 | 状态 |
|---|---|---|
| `01-op179089-fp32-add` | 同形连续 FP32 逐元素加法，1D–8D | Atlas A3 真机 **100% 验收通过** |
| `02-cannjudge-2524-broadcast-add` | 通用多类型、任意维 NumPy 广播加法（CANN Judge #2524 提交版） | 源码就绪，支持 NPU / CPU 双模式编译 |

> 本仓库是个人学习与测试过程的整理归档，用于记录 Ascend C 向量算子的排错、优化与验证方法，**不是生产级算子库**。

---

## 目录结构

```text
.
├── README.md                                  # 本文件
├── 算子整合总结文档.md                          # 项目全景、技术总结、性能数据、溯源清单
├── 01-op179089-fp32-add/                      # FP32 逐元素 Add（Atlas A3 验收版）
│   ├── op179089.asc                           # 验收版源码（主入口 + AddKernel）
│   ├── CMakeLists.txt / build.sh              # CANN 9.0 构建脚本
│   ├── baseline/ next/ op179089.patch         # 基线、最终版、统一补丁
│   ├── SHA256SUMS.txt                         # 20 项核心文件校验和
│   ├── scripts/                               # 回归套件、msprof 解析、Tile 建模、安全合入
│   ├── tests/                                 # Mock 头文件 + 5 阶段本地回归
│   └── reports/                               # 诊断报告、A3 验收报告、memcheck 原始日志、msprof 数据
├── 02-cannjudge-2524-broadcast-add/           # 通用多类型广播 Add（CANN Judge #2524）
│   ├── kernel.asc                             # 1392 行多类型通用广播 Add 内核 + Host 分发
│   ├── main.asc                               # ACL 运行入口与张量 I/O
│   ├── CMakeLists.txt / run.sh                # NPU / CPU 双模式编译调测
│   └── sample.py / reference.py               # 造数、NumPy 参考实现、精度校验
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
- **安全原语**：`SafeAdd4B` / `SafeCast` 复用子项目一的 64 元素边界拆分经验；`CopyInTile` / `CopyOutTile` 自动在 `DataCopy` 与 `DataCopyPad` 间切换
- **精度**：FP16/BF16 在 UB 内升精度到 FP32 计算，再以 `CAST_RINT` 回转

### 构建与调测

```bash
cd "02-cannjudge-2524-broadcast-add"
source /path/to/cann/set_env.sh

bash run.sh npu    # 昇腾真机
bash run.sh cpu    # 无设备时的 CANN CPU 调测模式（不可测性能）
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
