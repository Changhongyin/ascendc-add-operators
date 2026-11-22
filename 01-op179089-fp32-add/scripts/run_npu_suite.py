#!/usr/bin/env python3
"""Automated build, regression, benchmark, msprof, and mssanitizer suite for op179089."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

BENCH_RE = re.compile(
    r"BENCH N=(\d+) blocks=(\d+) median_us=([0-9.]+) min_us=([0-9.]+) max_us=([0-9.]+)(?: bw_gbs=([0-9.]+))?"
)
PASS_RE = re.compile(r"^PASS N=(\d+)", re.MULTILINE)
SANITIZER_ERROR_RE = re.compile(r"\bERROR:|\bERROR SUMMARY:\s*[1-9]", re.IGNORECASE)

EXPECTED_DEFAULT_PASS_COUNT = 25  # 20 original 1D + 1 large-N (4M) + 4 multi-dim shapes
BENCHMARK_LENGTHS = [16, 1024, 8193, 65536, 262144, 1048575, 1048576, 4194304, 16777216, 33554432]
INVALID_ARG_CASES: Tuple[Tuple[str, ...], ...] = (
    ("0",),
    ("-1",),
    ("33554433",),
    ("12x",),
    ("--benchmark", "0"),
    ("1", "2"),
    ("--shape",),
    ("--shape", "0x4"),
    ("--shape", "12x"),
    ("--shape", "1x2x3x4x5x6x7x8x9"),
    ("--shape", "100000x100000"),
)


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as error:
        return subprocess.CompletedProcess(cmd, 127, "", str(error))
    except subprocess.TimeoutExpired as error:
        stdout = (error.stdout or b"").decode(errors="replace")
        stderr = (error.stderr or b"").decode(errors="replace")
        return subprocess.CompletedProcess(cmd, 124, stdout, f"{stderr}\n{error}")


def parse_bench_lines(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in BENCH_RE.finditer(text):
        row: Dict[str, Any] = {
            "N": int(match.group(1)),
            "blocks": int(match.group(2)),
            "median_us": float(match.group(3)),
            "min_us": float(match.group(4)),
            "max_us": float(match.group(5)),
        }
        if match.group(6) is not None:
            row["bw_gbs"] = float(match.group(6))
        rows.append(row)
    return rows


def render_report(report: Dict[str, Any]) -> str:
    lines = [
        "# op179089 自动化验证与基准报告",
        "",
        f"- **NPU_ARCH**: `{report['npu_arch']}`",
        f"- **Binary**: `{report['binary']}`",
        f"- **Validation PASS**: `{report['validation']['pass_count']}/{report['validation']['expected_count']}` ({report['validation']['status']})",
        f"- **Invalid-Arg Checks**: `{report['invalid_args']['passed']}/{report['invalid_args']['total']}` ({report['invalid_args']['status']})",
        f"- **Shape and length boundaries**: `{report['shape_checks']['passed']}/{report['shape_checks']['total']}` ({report['shape_checks']['status']})",
        f"- **Benchmark**: {report['benchmark']['status']}",
        "",
    ]
    bench = report.get("benchmark", {}).get("rows", [])
    if bench:
        lines.append("## 流内基准测试结果")
        lines.append("")
        lines.append("| 元素数 N | Blocks | Median (µs) | Min (µs) | Max (µs) | 有效带宽 (GB/s) |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for r in bench:
            bw = f"{r['bw_gbs']:.2f}" if "bw_gbs" in r else "-"
            lines.append(
                f"| {r['N']:,} | {r['blocks']} | {r['median_us']:.3f} | {r['min_us']:.3f} | {r['max_us']:.3f} | {bw} |"
            )
        lines.append("")
    if report.get("msprof"):
        lines.append("## msprof 采集状态")
        lines.append("")
        for item in report["msprof"]:
            lines.append(f"- `N={item['N']}` {item['status']}, exit={item['returncode']}, out=`{item['output_dir']}`")
        lines.append("")
    if report.get("mssanitizer"):
        san = report["mssanitizer"]
        lines.append("## mssanitizer 内存检测")
        lines.append("")
        for item in san:
            lines.append(f"- {item['status']}, exit={item['returncode']}, target=`{item['target']}`")
        lines.append("")
    return "\n".join(lines)


def build_in_workspace(project_dir: Path, npu_arch: str, isolated: bool, env: Dict[str, str]) -> Tuple[Path, Optional[Path]]:
    work_dir = project_dir
    temp_root: Optional[Path] = None
    if isolated:
        temp_root = Path(tempfile.mkdtemp(prefix="op179089-verify."))
        work_dir = temp_root / "project"
        work_dir.mkdir()
        for fname in ("op179089.asc", "build.sh", "CMakeLists.txt"):
            src = project_dir / fname
            if not src.is_file():
                raise FileNotFoundError(f"Missing required project file: {src}")
            shutil.copy2(src, work_dir / fname)

    build_env = {**env, "NPU_ARCH": npu_arch}
    res = run_cmd(["bash", "build.sh"], cwd=work_dir, env=build_env, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(f"build.sh failed (exit {res.returncode}):\n{res.stdout}\n{res.stderr}")
    binary = work_dir / "build" / "op179089"
    if not binary.is_file():
        raise FileNotFoundError(f"Expected binary not produced: {binary}")
    return binary, temp_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full op179089 verification, benchmark, and profiling suite.")
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(), help="op179089 project directory")
    parser.add_argument("--binary", type=Path, default=None, help="Use an already-built binary instead of running build.sh")
    parser.add_argument("--npu-arch", default=os.environ.get("NPU_ARCH", "dav-2201"), help="Target NPU_ARCH (dav-2201 or dav-3510)")
    parser.add_argument("--device-id", default=None, help="Optional OP179089_DEVICE_ID override")
    parser.add_argument("--isolated-build", action="store_true", help="Build in /tmp copy without modifying project build/")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip --benchmark sweep")
    parser.add_argument("--profile", action="store_true", help="Run msprof op (PipeUtilization,Memory) on 16M and 32M")
    parser.add_argument("--sanitize", action="store_true", help="Run mssanitizer check on unaligned 8193 & 2D shape")
    parser.add_argument("--sanitizer-binary", type=Path, help="Use a separately instrumented binary for --sanitize")
    parser.add_argument("--out-dir", type=Path, default=Path("./op179089_reports"), help="Output directory for reports/profiles")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()

    env = dict(os.environ)
    if args.device_id is not None:
        env["OP179089_DEVICE_ID"] = str(args.device_id)

    temp_root: Optional[Path] = None
    try:
        if args.binary is not None:
            binary = args.binary.resolve()
        else:
            binary, temp_root = build_in_workspace(args.project_dir.resolve(), args.npu_arch, args.isolated_build, env)

        cwd = binary.parent
        # 1. Default validation suite (25 cases)
        val_res = run_cmd([str(binary)], cwd=cwd, env=env, timeout=180)
        pass_count = len(PASS_RE.findall(val_res.stdout))
        val_ok = (val_res.returncode == 0) and (pass_count == EXPECTED_DEFAULT_PASS_COUNT)

        # 2. Invalid argument checks (exit code 2)
        invalid_passed = 0
        invalid_failures = []
        for bad_args in INVALID_ARG_CASES:
            res = run_cmd([str(binary), *bad_args], cwd=cwd, env=env, timeout=15)
            if res.returncode == 2:
                invalid_passed += 1
            else:
                invalid_failures.append({"args": list(bad_args), "returncode": res.returncode, "stderr": res.stderr.strip()})
        invalid_ok = invalid_passed == len(INVALID_ARG_CASES)

        # 3. Targeted N-D and maximum-length checks
        shape_cases = [
            (["--shape", "32x64x129"], "PASS N=264192 shape=32x64x129 "),
            (["--shape", "2x2x2x2x2x2x2x3"], "PASS N=384 shape=2x2x2x2x2x2x2x3 "),
            (["33554431"], "PASS N=33554431 "),
            (["--shape", "2x2x2x2x2x2x2x262144"], "PASS N=33554432 shape=2x2x2x2x2x2x2x262144 "),
        ]
        shape_passed = 0
        shape_failures = []
        for cli_args, expected_pass in shape_cases:
            res = run_cmd([str(binary), *cli_args], cwd=cwd, env=env, timeout=30)
            if res.returncode == 0 and any(line.startswith(expected_pass) for line in res.stdout.splitlines()):
                shape_passed += 1
            else:
                shape_failures.append({"args": cli_args, "returncode": res.returncode, "stdout": res.stdout.strip()})
        shape_ok = shape_passed == len(shape_cases)

        # 4. Benchmark sweep
        bench_rows: List[Dict[str, Any]] = []
        bench_ok = True
        if not args.skip_benchmark:
            bench_res = run_cmd([str(binary), "--benchmark"], cwd=cwd, env=env, timeout=240)
            bench_rows = parse_bench_lines(bench_res.stdout)
            bench_ok = bench_res.returncode == 0 and [row["N"] for row in bench_rows] == BENCHMARK_LENGTHS

        # 5. Optional msprof op
        msprof_results = []
        if args.profile:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for prof_n in (16777216, 33554432):
                prof_dir = Path(tempfile.mkdtemp(prefix=f"msprof_{prof_n}.", dir=args.out_dir))
                res = run_cmd(
                    [
                        "msprof",
                        "op",
                        f"--output={prof_dir}",
                        "--aic-metrics=PipeUtilization,Memory",
                        str(binary),
                        str(prof_n),
                    ],
                    cwd=cwd,
                    env=env,
                    timeout=300,
                )
                (prof_dir / "msprof.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
                has_profile = all(
                    any(path.stat().st_size for path in prof_dir.rglob(name))
                    for name in ("OpBasicInfo.csv", "Memory.csv", "PipeUtilization.csv")
                )
                has_pass = str(prof_n) in PASS_RE.findall(res.stdout + "\n" + res.stderr)
                msprof_results.append({
                    "N": prof_n,
                    "status": "PASS" if res.returncode == 0 and has_profile and has_pass else "FAIL",
                    "returncode": res.returncode,
                    "output_dir": str(prof_dir),
                })

        # 6. Optional mssanitizer
        san_results = []
        if args.sanitize:
            san_binary = args.sanitizer_binary.resolve() if args.sanitizer_binary else binary
            for target, length in ((["8193"], "8193"), (["--shape", "16x257"], "4112")):
                res = run_cmd(
                    ["mssanitizer", "--tool=memcheck", "--", str(san_binary), *target],
                    cwd=san_binary.parent,
                    env=env,
                    timeout=180,
                )
                tool_output = res.stdout + "\n" + res.stderr
                has_pass = length in PASS_RE.findall(tool_output)
                verified = "No error detected." in tool_output and not SANITIZER_ERROR_RE.search(tool_output)
                san_results.append({
                    "target": " ".join(target),
                    "status": "PASS" if res.returncode == 0 and has_pass and verified else "FAIL",
                    "returncode": res.returncode,
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                })

        report: Dict[str, Any] = {
            "npu_arch": args.npu_arch,
            "binary": str(binary),
            "validation": {
                "status": "PASS" if val_ok else "FAIL",
                "returncode": val_res.returncode,
                "pass_count": pass_count,
                "expected_count": EXPECTED_DEFAULT_PASS_COUNT,
                "stderr": val_res.stderr,
            },
            "invalid_args": {
                "status": "PASS" if invalid_ok else "FAIL",
                "passed": invalid_passed,
                "total": len(INVALID_ARG_CASES),
                "failures": invalid_failures,
            },
            "shape_checks": {
                "status": "PASS" if shape_ok else "FAIL",
                "passed": shape_passed,
                "total": len(shape_cases),
                "failures": shape_failures,
            },
            "benchmark": {
                "status": "SKIPPED" if args.skip_benchmark else ("PASS" if bench_ok else "FAIL"),
                "rows": bench_rows,
            },
            "msprof": msprof_results,
            "mssanitizer": san_results,
        }

        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        md = render_report(report)
        (args.out_dir / "report.md").write_text(md, encoding="utf-8")
        print(md)
        tools_ok = all(item["status"] == "PASS" for item in msprof_results + san_results)
        return 0 if val_ok and invalid_ok and shape_ok and bench_ok and tools_ok else 1
    finally:
        if temp_root is not None and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
