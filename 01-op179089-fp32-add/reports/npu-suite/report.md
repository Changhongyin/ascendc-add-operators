# op179089 自动化验证与基准报告

- **NPU_ARCH**: `dav-2201`
- **Binary**: `/mnt/workspace/data/op179089-verify/op179089-claude-out/build/op179089`
- **Validation PASS**: `25/25` (PASS)
- **Invalid-Arg Checks**: `11/11` (PASS)
- **Shape and length boundaries**: `4/4` (PASS)
- **Benchmark**: PASS

## 流内基准测试结果

| 元素数 N | Blocks | Median (µs) | Min (µs) | Max (µs) | 有效带宽 (GB/s) |
|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 7.924 | 7.863 | 8.028 | 0.02 |
| 1,024 | 1 | 7.946 | 7.881 | 8.044 | 1.55 |
| 8,193 | 9 | 7.979 | 6.697 | 7.992 | 12.32 |
| 65,536 | 40 | 7.923 | 7.846 | 7.985 | 99.27 |
| 262,144 | 40 | 7.919 | 7.093 | 8.045 | 397.22 |
| 1,048,575 | 40 | 7.906 | 7.420 | 8.063 | 1591.47 |
| 1,048,576 | 40 | 7.928 | 6.573 | 8.049 | 1587.09 |
| 4,194,304 | 40 | 11.370 | 11.363 | 11.380 | 4426.58 |
| 16,777,216 | 40 | 184.981 | 184.695 | 185.115 | 1088.36 |
| 33,554,432 | 40 | 372.981 | 372.723 | 373.038 | 1079.56 |

## msprof 采集状态

- `N=16777216` PASS, exit=0, out=`/mnt/workspace/data/op179089-verify/op179089-claude-out/reports/npu-suite/msprof_16777216.20bswdhs`
- `N=33554432` PASS, exit=0, out=`/mnt/workspace/data/op179089-verify/op179089-claude-out/reports/npu-suite/msprof_33554432.fnxr5t56`

## mssanitizer 内存检测

- PASS, exit=0, target=`8193`
- PASS, exit=0, target=`--shape 16x257`
