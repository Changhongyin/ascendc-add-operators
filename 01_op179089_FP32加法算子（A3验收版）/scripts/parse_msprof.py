#!/usr/bin/env python3
"""Parse CANN 9.0 msprof op logs and CSV exports into JSON & Markdown tables."""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Dict, List, Optional

TASK_DURATION_RE = re.compile(r"Task Duration\(us\):\s*([0-9]+(?:\.[0-9]+)?)")
BLOCK_DIM_RE = re.compile(r"Block Dim:\s*([0-9]+)")
KERNEL_NAME_RE = re.compile(r"Op Name:\s*(\S+)")

PIPE_COLUMNS = (
    "aic_vec_ratio",
    "aiv_vec_ratio",
    "vec_ratio",
    "aiv_mte2_ratio",
    "mte2_ratio",
    "aiv_mte3_ratio",
    "mte3_ratio",
    "aiv_scalar_ratio",
    "scalar_ratio",
)
MEM_COLUMNS = (
    "aiv_read_main_memory_datas(KB)",
    "aiv_write_main_memory_datas(KB)",
    "aiv_ub_read_bw(GB/s)",
    "aiv_ub_write_bw(GB/s)",
    "aiv_mte2_bw(GB/s)",
    "aiv_mte3_bw(GB/s)",
    "Task Duration(us)",
    "Duration(us)",
)


def parse_log_text(text: str) -> Dict[str, Any]:
    durations = [float(x) for x in TASK_DURATION_RE.findall(text)]
    block_dims = [int(x) for x in BLOCK_DIM_RE.findall(text)]
    kernel_names = KERNEL_NAME_RE.findall(text)
    result: Dict[str, Any] = {
        "samples": len(durations),
        "durations_us": durations,
        "block_dim": block_dims[-1] if block_dims else None,
        "kernel_name": kernel_names[-1] if kernel_names else None,
    }
    if durations:
        result["median_us"] = round(statistics.median(durations), 3)
        result["min_us"] = round(min(durations), 3)
        result["max_us"] = round(max(durations), 3)
    return result


def try_float(val: str) -> Optional[float]:
    val = val.strip()
    if not val or val.upper() in ("N/A", "NA", "NONE", "-"):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_csv_files(root: Path) -> Dict[str, Any]:
    metrics: Dict[str, List[float]] = {}
    csv_files = sorted(root.rglob("*.csv"))
    for csv_file in csv_files:
        try:
            with csv_file.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                interesting = [c for c in reader.fieldnames if c in PIPE_COLUMNS or c in MEM_COLUMNS]
                if not interesting:
                    continue
                for row in reader:
                    for col in interesting:
                        num = try_float(row.get(col, ""))
                        if num is not None:
                            metrics.setdefault(col, []).append(num)
        except OSError:
            continue
    return {
        col: {
            "count": len(vals),
            "median": round(statistics.median(vals), 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
        }
        for col, vals in sorted(metrics.items())
    }


def collect_target(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {"source": str(path)}
    if path.is_file():
        data["log"] = parse_log_text(path.read_text(encoding="utf-8", errors="ignore"))
        data["csv_metrics"] = {}
        return data

    log_candidates = sorted(path.rglob("msprof*.log"))
    combined_log = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in log_candidates)
    data["log"] = parse_log_text(combined_log)
    data["csv_metrics"] = parse_csv_files(path)
    return data


def render_markdown(summary: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> str:
    lines = ["### msprof op 汇总报告", ""]
    log = summary.get("log", {})
    if log.get("samples", 0) > 0:
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---:|")
        if log.get("kernel_name"):
            lines.append(f"| Kernel | `{log['kernel_name']}` |")
        if log.get("block_dim") is not None:
            lines.append(f"| Block Dim | {log['block_dim']} |")
        lines.append(f"| 样本数 | {log['samples']} |")
        lines.append(f"| Task Duration 中位数 (µs) | {log['median_us']:.3f} |")
        lines.append(f"| Task Duration 范围 (µs) | {log['min_us']:.3f} ~ {log['max_us']:.3f} |")
        if baseline and baseline.get("log", {}).get("median_us"):
            base_med = baseline["log"]["median_us"]
            speedup = base_med / log["median_us"] if log["median_us"] > 0 else 0.0
            lines.append(f"| 对比基线中位数 (µs) | {base_med:.3f} → {log['median_us']:.3f} ({speedup:.2f}x) |")
        lines.append("")
    else:
        lines.append("> 未在日志中检测到 `Task Duration(us)` 记录。")
        lines.append("")

    csv_metrics = summary.get("csv_metrics", {})
    if csv_metrics:
        lines.append("| CSV 硬件指标 | 中位数 | 最小值 | 最大值 | 样本数 |")
        lines.append("|---|---:|---:|---:|---:|")
        for col, stats in csv_metrics.items():
            lines.append(
                f"| `{col}` | {stats['median']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse CANN 9.0 msprof op logs and CSV metrics.")
    parser.add_argument("input", type=Path, help="msprof log file or output directory")
    parser.add_argument("--baseline-json", type=Path, default=None, help="Optional baseline JSON for comparison")
    parser.add_argument("--json-out", type=Path, default=None, help="Write parsed JSON to file")
    parser.add_argument("--md-out", type=Path, default=None, help="Write Markdown table to file")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    summary = collect_target(args.input)
    baseline = None
    if args.baseline_json and args.baseline_json.is_file():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))

    md = render_markdown(summary, baseline)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
