# op179089 — UB 访存报告诊断（N=8193 / N=40961）

日期：2026-11-21 · 工作目录：`op179089-current/op179089`（`baseline/` 未改动）

> **交付状态**：算子源码 `op179089.asc` **未修改**。Atlas A3 不可用，无法复现 UB 访存报告，
> 因此没有把未经硬件确认的内核改动写进交付源码。本报告给出根因假设、判别实验、
> 以及一份已在本地验证过的候选补丁（`reports/candidates/`），待 A3 一次 memcheck 即可定论。

---

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 复现 UB 报告 | **未能复现** — 本机无 CANN / NPU / mssanitizer，无 A3 通路（见第 6 节） |
| 根因 | **主假设 A 已高度收敛（5/5 吻合），尚未经硬件确认** |
| 源码改动 | 无（`op179089.asc`、`next/`、`op179089.patch`、`SHA256SUMS.txt` 中算子源码部分不变） |
| 新增文件 | `scripts/tile_model.py`、`reports/candidates/*`、本报告、本地回归日志 |
| 本地回归 | **全部通过**：25 功能 / 11 非法参数 / 4 形状与长度边界（见第 5 节） |
| 性能测量 | **未完成** — 16M / 32M 的 median latency 与有效带宽需真机（见第 6 节） |

## 2. 证据：失败与通过用例的唯一结构差异

`AddKernel::Run` 的分块与分 tile 算术已用 `scripts/tile_model.py` 完整复算
（该脚本逐行镜像 `op179089.asc` 的 `Run()`，并断言 tile 恰好覆盖 `[0, N)` 一次）。

定义 **性质 P（partial repeat）**：某个 tile 的元素数 `count` 满足
`count >= 64 且 count % 64 != 0` —— 即一次**跨多个 256 B 向量 repeat、而最后一个 repeat 不完整**的
`Add`。（fp32 下一个 repeat = 256 B / 4 B = 64 lane。）

已插桩的 A3 用例与 P 完全对应：

| N | blocks | tile 构成 | P | A3 memcheck |
|---|---|---|---|---|
| 1 | 1 | 1 个 pad tile，count=1 | 无 | **通过** |
| 7 | 1 | 1 个 pad tile，count=7 | 无 | **通过** |
| 9 | 1 | 1 个 pad tile，count=9 | 无 | **通过** |
| 8193 | 9 | 8 × aligned count=912 + 1 × pad count=897 | **9 个 tile 全部有 P** | **报错** |
| 40961 | 41 | 37 × 1000 + 3 × 992（aligned）+ 1 × pad 985 | **41 个 tile 全部有 P** | **报错** |

`python3 scripts/tile_model.py 8193` 可复现上表。要点：

- 通过的 3 个用例是**唯一**只含「单个不足一个 repeat 的 tile」（count < 64）的用例：
  掩码只覆盖一次 repeat，不存在"后续 repeat"。
- 失败的 2 个用例里，**连 aligned 路径也有 P**（912 / 1000 / 992 都是 32 B 对齐、但不是 64 的倍数），
  说明问题不局限于非对齐尾块。
- 相邻的 N=8192、40960 全部 tile 恰好 1024 元素 = 16 个完整 repeat（无 P）；
  这正好解释了"只有 ...93 / ...61 这类长度报错"。

## 3. 主假设 A：多 repeat 向量操作的不完整尾 repeat 读到了 DMA 未填充的 UB

把 `AddKernel` 的两个版本并排看（`baseline/` → 当前源码，内核内唯一改动）：

```c
- AscendC::Add(z, x, y, vectorCount);   // baseline：按 32 B 向上取整的元素数
+ AscendC::Add(z, x, y, count);         // 当前：按真实元素数
```

- `vectorCount` 版本：`Add` 覆盖 `[0, vectorCount)`，在 N=1/7/9 上被 memcheck 报错；
  改成 `count` 后这三例转为通过 —— 说明**检查器确实对 `[count, vectorCount)` 区间的 UB 访问敏感**，
  即它关心的是「访问的 UB 是否由前序 DMA 真正填充过」，而不只是是否越过缓冲区边界
  （UB 缓冲区恒为 `4096 × 4 B`，两版本都远未越界）。
- 但 `count` 版本只消除了 `count < 64` 这一类：当 `count > 64` 且 `count % 64 != 0` 时，
  `Add` 被拆成若干完整 repeat 加一个不完整尾 repeat；尾 repeat 的 lane 地址覆盖到
  `ceil(count/64)*64`，而 `DataCopy` / `DataCopyPad` 只填充到 `count`（pad 路径填到 32 B 边界，
  即 `vectorCount`）。中间的 lane 从未被写入 → 命中 memcheck。

该假设同时解释了：为什么 1/7/9 通过、8193/40961 报错、以及为什么"调整向量掩码"只修好了小用例。

**待确认的前提**（无法本地验证，需 CANN 文档或真机）：CANN 9.0 的 `Add(dst, src0, src1, calCount)`
在 `calCount > 64` 时尾 repeat 的访存是否按整 repeat 粒度寻址。
这正是需要 memcheck 首条错误记录（类型 / 大小 / UB 偏移 / 报告 block）来定论的部分。

## 4. 判别实验（A3 上 4 次 memcheck 即可定论）

假设 A 之外，已知 5 个用例也同样被「blocks > 1」区分开（1/7/9 都是单 block），
所以必须用下面的交叉用例把两者分开。`--shape 32x64x129` 是 48 block 但全部 tile 无 P 的关键对照。

| 探针 | blocks | P | 假设 A 预测 | 假设 B（分块/多核相关）预测 |
|---|---|---|---|---|
| `513` | **1** | **有** | 报错 | 干净 |
| `1023` | **1** | **有** | 报错 | 干净 |
| `8192` | **8** | 无 | 干净 | 报错 |
| `--shape 32x64x129`（264192） | **48** | 无 | 干净 | 报错 |
| `--shape 16x257`（4112） | 5 | 有 | 报错 | 报错 |

```bash
cmake -S . -B build-memcheck -DNPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=Release \
      "-DCMAKE_ASC_FLAGS=--cce-enable-sanitizer -g"
cmake --build build-memcheck --parallel
for c in 513 1023 8192 8193 40961; do
  mssanitizer --tool=memcheck -- ./build-memcheck/op179089 "$c" 2>&1 | tee "reports/memcheck-$c.log"
done
mssanitizer --tool=memcheck -- ./build-memcheck/op179089 --shape 32x64x129 2>&1 | tee reports/memcheck-32x64x129.log
mssanitizer --tool=memcheck -- ./build-memcheck/op179089 --shape 16x257     2>&1 | tee reports/memcheck-16x257.log
```

判读：进程退出码不足为据，须确认日志里算子确实执行、且 sanitizer 显式报告 0 错误；
若报错，请保留**首条**错误记录（读/写、长度、UB 偏移、调用栈、报告 block），它直接区分
"尾 repeat 读未填充 UB"（偏移落在 `[count, ceil(count/64)*64)`）与其他机制。

次要假设 C（已降级）：pad 路径的 GM 侧 32 B 粒度回读会越过 `x`/`y` 的 `aclrtMalloc` 末尾
（`z` 已有 8 元素 guard，`x`/`y` 没有）。它无法解释 N=1/7/9 通过（同样越界），
但探针极廉价：把 host 侧 `x`/`y` 的申请字节数向上取整到 32 B，若报告消失即为此因。

## 5. 本地验证结果（已完成）

`python3 tests/test_local_all.py` 等价运行，全部通过：

```
[1/5] clang++ -Wall -Wextra -Werror 编译干净；AddKernel tail mask 断言通过
[2/5] 25/25 功能、11/11 非法参数（exit 2）、4/4 形状与长度边界（含 8D 与 33,554,432）
      msprof / mssanitizer 工具失败传播（exit 17 → suite exit 1）
[3/5] 40 / 48 / 64 核扩展、OP179089_DEVICE_ID 覆盖、BENCH bw_gbs 格式
[4/5] parse_msprof.py 日志 + CSV + baseline 加速比
[5/5] apply_to_cloud.py 校验和守卫、dry-run、备份、原子替换
```

原始日志：`reports/local-validation-20261121.log`。

本机为 Windows 11 ARM64，为跑通套件做了 4 处**仅限宿主环境**的替代，均不涉及算子逻辑：

1. 无 clang++ → 用一个转发到 `C:\mingw64\bin\g++.exe` 的 `clang++.exe` 包装器（g++ 16.2.0，同样 `-Wall -Wextra -Werror` 干净）。
2. 套件用 `#!/bin/sh` 脚本伪造 msprof / mssanitizer，Windows 无法执行 → 换成返回 17 的 `.exe`，断言不变。
3. `scripts/apply_to_cloud.py` 用了 POSIX 专有的 `os.fchmod`，在 Windows 上必然抛 `AttributeError`
   → 测试期间用 `sitecustomize.py` 打了桩。**这是脚本的真实可移植性缺陷**，在 Linux 云端主机上不触发；
   如需 Windows 可用，应改为 `os.chmod(tmp.name, mode)`。
4. 控制台 cp1252 无法输出报告中的中文 → `PYTHONUTF8=1`。

注意：`tests/test_local_all.py` 第 1 步会**重新生成** `op179089.patch`。在 Windows 上它会被写成 CRLF，
从而改变该交付文件的校验和。本次已从输入 ZIP 恢复原始字节，`sha256sum -c SHA256SUMS.txt` 全部 OK。

### 候选补丁的本地验证

`reports/candidates/op179089-tail-repeat-split.asc`（补丁 `…-tail-repeat-split.patch`，3 个 hunk）
把每个 block 的尾部切在向量 repeat 边界上，使任何一次 `Add` 要么是整数个 repeat、
要么是单个不足一个 repeat 的操作（即 N=1/7/9 那种已被硬件接受的形态）：

```c
uint32_t tail = blockLength - offset;
const uint32_t wholeRepeats = tail - tail % kVectorRepeatElements;
if (wholeRepeats != 0) {
    ProcessTile<true>(offset, wholeRepeats);   // 64 的倍数 ⇒ 同时满足 32 B 对齐
    offset += wholeRepeats;
    tail -= wholeRepeats;
}
```

已验证（本地）：

- `clang++ -std=c++17 -O2 -Wall -Wextra -Werror` 编译干净；
- 同一套模拟回归 **25/25、11/11、4/4** 通过，40/48/64 核扩展与 `16x257` 基准格式通过；
- 用 `scripts/tile_model.py` 对全部 27 个用例建模：性质 P 的 tile 数由最多 48 降为 **0**，
  且 tile 仍恰好覆盖 `[0, N)` 一次（脚本内断言）；
- 代价：每个 block 最多多一个 tile。16M 时 4128 → 4176 个 tile（+1.2%），
  32M 时 8208 → 8256（+0.6%），即多一对 DMA；4096 元素 tile、32 M 上限、1D–8D、CLI 均不变。

保留项：若硬件确认假设 A，再把该补丁并入 `op179089.asc` / `next/` / `op179089.patch` / `SHA256SUMS.txt`，
并把 `tests/test_local_all.py` 里 `AddKernel` 的那条等式断言从"只允许掩码改动"更新为
"掩码改动 + 尾部 repeat 切分"（建议断言切分后的三段文本，保持其仍能挡住无意的内核改动）。

## 6. 尚未完成的硬件检查（Atlas A3 不可用）

本机核查结果：无 `npu-smi` / `msprof` / `mssanitizer` / Ascend 编译器，无 `/usr/local/Ascend` 等 CANN 安装，
无 Ascend 相关环境变量，无 WSL，无 `~/.ssh` 配置或云端凭据 —— 没有任何 A3 通路。

因此以下全部**待办**：

1. **复现**：A3 + CANN 9.0.0 + dav-2201，插桩构建后对 N=8193、40961 跑 memcheck，
   抓取首条错误记录（类型、长度、UB 偏移、调用栈、报告 block）。这是定根因的决定性证据。
2. **判别**：第 4 节的 4 个探针（513、1023、8192、`--shape 32x64x129`）+ `16x257`，用于在假设 A / B 间定论。
3. **回归**：N=1、7、9、8193、40961 与 `--shape 16x257` 六例 memcheck 显式零错误。
4. **候选补丁验收**：若假设 A 成立，对打上补丁的版本重跑第 3 项，并重跑完整 NPU 套件
   （25 功能 / 11 非法参数 / 4 边界）确认无功能回归。
5. **性能**：release 构建后 `scripts/run_npu_suite.py --profile`，记录 16,777,216 与 33,554,432 元素的
   median latency 与有效带宽（两读一写口径），与候选版前的参考值
   （16M：182.018 µs / 1106.08 GB/s，32M：370.943 µs / 1085.49 GB/s）对比，确认多出的那对 DMA 影响在噪声范围内。
6. 原始 memcheck 日志与 profile 报告留存到 `reports/`。
