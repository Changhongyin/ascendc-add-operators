# op179089 算子当前版本（Atlas A3 真机验收通过）

这是两个同形连续 FP32 张量的逐元素加法。支持 1D–8D，元素总数上限为 33,554,432。当前默认 Tile 为 4096 个元素。

- `op179089.asc` 为当前已验收源码，可配合 `CMakeLists.txt` 和 `build.sh` 在 CANN 9.0 环境构建。
- `baseline/`、`next/` 和 `op179089.patch` 保留本地交接基线及统一补丁。交接基线校验和与当前云端工程不同，应用前应核对实际文件。
- `scripts/` 提供回归、性能采集、日志解析、Tile 建模及安全安装脚本。
- `reports/` 留存诊断分析（`0001-ub-access-diagnosis.md`）、Atlas A3 真机验收报告（`0002-atlas-a3-verification-report.md`）、升级前后全部 `mssanitizer` 原始日志（`memcheck-before/`、`memcheck-after/`）以及 `run_npu_suite.py` / `msprof` 原始与解析报告（`npu-suite/`）。

## 构建和运行

载入 CANN 9.0 环境后执行：

```bash
NPU_ARCH=dav-2201 bash build.sh
./build/op179089 --shape 2x2x2x2x2x2x2x3
python3 scripts/run_npu_suite.py --binary build/op179089 --profile
```

插桩内存检查：

```bash
cmake -S . -B build-memcheck -DNPU_ARCH=dav-2201 -DCMAKE_BUILD_TYPE=Release "-DCMAKE_ASC_FLAGS=--cce-enable-sanitizer -g"
cmake --build build-memcheck --parallel
mssanitizer --tool=memcheck -- ./build-memcheck/op179089 8193
```

## 验证状态（Atlas A3, CANN 9.0.0, dav-2201, 40 核）

- **本地模拟回归**：`python3 tests/test_local_all.py` 5/5 阶段通过（25 功能 / 11 非法参数 / 4 边界 / 多核扩展 / `parse_msprof` / `apply_to_cloud`）。
- **真机插桩 `mssanitizer --tool=memcheck`**：11/11 用例（`1, 7, 9, 72, 513, 1023, 8192, 8193, 40961, --shape 16x257, --shape 32x64x129`）全部显式报告 `No error detected.`（0 错误）。
- **真机全量回归与性能**：25/25 功能、11/11 非法参数、4/4 边界与 10/10 基准全通过；16M 元素流内中位数 `184.981 µs / 1088.36 GB/s`（`msprof` Task Duration `142.517 µs`，`aiv_mte2_ratio` 98.09%），32M 元素流内中位数 `372.981 µs / 1079.56 GB/s`（`msprof` Task Duration `337.253 µs`，`aiv_mte2_ratio` 98.95%）。
