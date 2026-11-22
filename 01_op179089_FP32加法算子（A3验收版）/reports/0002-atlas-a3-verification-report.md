# op179089 — Atlas A3 真机 UB 根因确认、零额外 DMA 算法升级与全套验收报告

日期：2026-11-22 · 环境：Atlas A3 (`dav-2201`, 40 Vector Cores), CANN 9.0.0 (`bisheng` / `mssanitizer` / `msprof`)

> **验收结论**：全部真机与本地验收项 **100% PASS**。
> - 插桩 `mssanitizer --tool=memcheck`：**11/11 用例显式报告 `No error detected.`**（含全部 6 组必测用例 `1, 7, 9, 8193, 40961, --shape 16x257` 以及 5 组扩展判别探针 `72, 513, 1023, 8192, --shape 32x64x129`）。
> - 全量 NPU 回归套件：`25/25` 功能、`11/11` 非法参数、`4/4` 形状与长度边界（含 8D 与 33,554,432）、`10/10` 流内基准、`msprof`（16M & 32M）、`mssanitizer` 全部 **PASS**。
> - 性能无回退：`msprof` 测得 16M Task Duration **142.517 µs**（`aiv_mte2_ratio` 98.09%）、32M Task Duration **337.253 µs**（`aiv_mte2_ratio` 98.95%）；流内基准 16M **184.981 µs / 1088.36 GB/s**（独立复测 181.151 µs / 1111.37 GB/s）、32M **372.981 µs / 1079.56 GB/s**。

---

## 1. 真机复现与精确根因定位

### 1.1 升级前 11 组 `memcheck` 探针实测结果（原始日志见 `reports/memcheck-before/`）

| 用例 | Blocks | 路径与每核元素数 | 升级前 `memcheck` | 报错数 & 首条报错摘要 |
|---|---:|---|---|---|
| `1, 7, 9` | 1 | pad (`1, 7, 9`) | CLEAN | 0 |
| `72` | 1 | aligned (`72 = 64+8`) | CLEAN | 0 |
| `16x257` (`4112`) | 5 | aligned (`824, 816`) | CLEAN | 0 |
| `8192` | 8 | aligned (`1024`) | CLEAN | 0 |
| `513` | 1 | pad (`513 = 8×64+1`) | **FAIL** | 2 (`illegal write/read of size 4` at UB `0xc000`/`0x8000`, `op179089.asc:95` ← `:69`) |
| `1023` | 1 | pad (`1023 = 15×64+63`) | **FAIL** | 584 (`illegal write of size 4` at UB `0xdf80`, `:95` ← `:69`) |
| `8193` | 9 | aligned (`912`) + pad (`897`) | **FAIL** | 770（仅 `aiv(8)` 报错，`385` 读 + `385` 写 size=4，`:95` ← `:69`） |
| `40961` | 40 | aligned (`1032, 1024`) + pad (`1017`) | **FAIL** | 1140（`aiv(0)` aligned 130 条 size=32 + `aiv(39)` pad 1010 条 size=4，`:95` ← `:67`/`:69`） |
| `32x64x129` (`264192`) | 40 | aligned 双缓冲（尾 `2512, 2504`） | **FAIL** | 684（`aiv(0-23)` 186 条 size=64 + `aiv(24-39)` 498 条 size=32，`:95` ← `:67`） |

### 1.2 根因确认（实证自 CANN 9.0.0 底层头文件 + UB 地址分析）

检查 CANN 9.0.0 实际展开代码 `/home/developer/Ascend/cann-9.0.0/aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_vec_binary_impl.h`（第 188–201 行）：

```cpp
// Add::Level 2
template <typename T>
__aicore__ inline void AddImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, const int32_t& count)
{
    if ASCEND_IS_AIV {
        set_mask_count();
        set_vector_mask(0, count);
        vadd(dst, src0, src1, 1, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE, DEFAULT_BLK_STRIDE,
            DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
    }
}
```

结合真实报错的 UB 偏移与数量规律（`x=[0x0000,0x4000)`、`y=[0x4000,0x8000)`、`z=[0x8000,0xc000)`，单缓冲各 16 KB = `0x4000`）：
- `N=513`：`513 - 512 = 1`，在 `z + 0x4000 = 0xc000` 与 `y + 0x4000 = 0x8000` 各报 1 次 `size=4` 越界（共 2 条）。
- `N=8193`（`aiv(8)`, `count=897`）：`897 - 512 = 385` 次读 + `385` 次写 `size=4` 越界（共 770 条）。
- `N=40961`（`aiv(0)`, `count=1032`, aligned）：`1032 / 8 - 64 = 65` 次读 + `65` 次写 `size=32` 越界（共 130 条）；`aiv(39)`（`count=1017`, pad）：`1017 - 512 = 505` 读 + `505` 写（共 1010 条）。
- `32x64x129`（`aiv(0-23)`, `count=2512`, aligned）：`2512 / 16 - 64 = 93` 读 + `93` 写 `size=64`（共 186 条）；`aiv(24-39)`（`count=2504`, aligned）：`2504 / 8 - 64 = 249` 读 + `249` 写 `size=32`（共 498 条）。

**根本原因**：Level-2 `AscendC::Add(z, x, y, count)` 使用 **Counter Mode** `vadd`。当 `count > 64` 且不能被 `kVectorRepeatElements = 64`（256 B / `sizeof(float)`）整除时，CANN 9.0.0 的 `mssanitizer` 插桩将 Counter-Mode `vadd` 按子块粒度与固定跨步展开校验，在 `count` 足够大时访问地址超出 `16 KB` `LocalTensor` 边界，触发 `illegal read/write on UB`（即使算子数值计算与 CPU 参考结果完全一致）。

---

## 2. 算法升级方案选型（零额外 DMA）

在 Atlas A3 真机上对比验证了两种修复方案：

| 对比维度 | 候选 A：Tile 级分拆（`tail-repeat-split`） | **最终采用：UB 内 repeat 边界分拆** |
|---|---|---|
| 改动位置 | `AddKernel::Run`（将尾 Tile 拆为两次 `ProcessTile`） | `AddKernel::ProcessTile`（在 UB 内按 64 元素边界拆为两次 `Add`） |
| 额外 GM↔UB DMA | 每核多 1 对 DMA（16M/32M +40~48 Tile） | **0（零额外 DMA、零额外 Tile）** |
| 11 组 A3 `memcheck` | 11/11 CLEAN (0 errors) | **11/11 CLEAN (0 errors)** |
| A3 独立基准 16M | 185.058 µs / 1087.91 GB/s | **181.151 µs / 1111.37 GB/s** |
| A3 独立基准 8193 / 40961 | 6.639 µs / 6.587 µs | **6.544 µs / 6.562 µs** |

最终采用 **UB 内 repeat 边界分拆**（`op179089.asc`）：
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
每次 `AscendC::Add` 要么是 `64` 的整数倍（完整 256 B repeat），要么是 `< 64` 的单 repeat 尾段；偏移 `wholeRepeats * 4` 严格 256 B（32 B 的 8 倍）对齐，且不引入任何额外 GM↔UB 搬运。

---

## 3. 升级后 Atlas A3 真机全套验收结果

### 3.1 插桩 `mssanitizer --tool=memcheck` 回归（原始日志见 `reports/memcheck-after/`）

11/11 全部通过且显式输出 `No error detected.` 与 `PASS`：
- 必测 6 组：`N=1`, `N=7`, `N=9`, `N=8193`, `N=40961`, `--shape 16x257` — **全部 0 错误**
- 扩展判别 5 组：`N=72`, `N=513`, `N=1023`, `N=8192`, `--shape 32x64x129` — **全部 0 错误**

### 3.2 Release 完整 NPU 套件与性能指标（原始日志见 `reports/npu-suite/`）

- `scripts/run_npu_suite.py --binary build/op179089 --profile --sanitize --sanitizer-binary build-memcheck/op179089 --out-dir reports/npu-suite`：**退出码 0（全绿）**
  - Validation：`25/25` PASS
  - Invalid-Arg：`11/11` PASS（exit 2）
  - Shape & Length Boundaries：`4/4` PASS（含 8D `2x2x2x2x2x2x2x3`、`33554431`、`2x2x2x2x2x2x2x262144`）
  - `mssanitizer` 集成项（`8193`, `16x257`）：PASS

#### 流内基准测试（`--benchmark`）与 `msprof op` 硬件指标汇总

| 元素数 N | Blocks | 流内 Median (µs) | 流内有效带宽 (GB/s) | `msprof` Task Duration (µs) | `aiv_mte2_ratio` | `aiv_mte3_ratio` | `aiv_vec_ratio` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `16,777,216` (16M) | 40 | 184.981 *(独立复测 181.151)* | 1088.36 *(1111.37)* | **142.517** *(核内等效 1412.69 GB/s)* | **98.09%** | 72.70% | 7.54% |
| `33,554,432` (32M) | 40 | 372.981 | 1079.56 | **337.253** *(核内等效 1193.93 GB/s)* | **98.95%** | 74.90% | 6.34% |

*(注：流内带宽按两读一写 `3 × N × 4 B / median_us` 计算；参考候选版前基线为 16M `182.018 µs / 1106.08 GB/s`、32M `370.943 µs / 1085.49 GB/s`，性能持平且 `aiv_mte2_ratio` 已达 98%–99% 饱和。)*

---

## 4. 变更文件清单

- [op179089.asc](../op179089.asc) 与 [next/op179089.asc](../next/op179089.asc)：在 `ProcessTile` 内按 `kVectorRepeatElements (64)` 边界分拆 `AscendC::Add`（零额外 DMA），并新增 `static_assert`。
- [op179089.patch](../op179089.patch)：从 `baseline/op179089.asc` 到 `next/op179089.asc` 重新生成的统一补丁（LF 换行）。
- [tests/mock_headers/kernel_operator.h](../tests/mock_headers/kernel_operator.h)：为本地模拟 `LocalTensor<T>` 增加 32B 对齐校验的 `operator[]` 视图及 `const LocalTensor<T>&` 签名，对齐真实 Ascend C 接口。
- [tests/test_local_all.py](../tests/test_local_all.py)：将 `AddKernel` 回归守卫更新为精确校验 UB 内 repeat 分拆改动，并固定 `op179089.patch` 写出 LF 换行。
- [README.md](../README.md)、[SHA256SUMS.txt](../SHA256SUMS.txt)、`reports/memcheck-before/`、`reports/memcheck-after/`、`reports/npu-suite/` 与本报告。
- `baseline/`、原始输入包及云端 git 仓库 `/mnt/workspace/gitCode/cann/cann-learning-hub/op179089` 均保持原样未修改。
