#!/usr/bin/env python3
"""Cloud Atlas A3 build, regression, benchmark, mssanitizer, and msprof runner for 03-nd-batched-matmul-fusion."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Atlas A3 verification suite for 03-nd-batched-matmul-fusion.")
    parser.add_argument("--npu-arch", default=os.environ.get("NPU_ARCH", "dav-2201"))
    parser.add_argument("--device-id", default=os.environ.get("DEVICE_ID", "0"))
    parser.add_argument("--memcheck", action="store_true", help="Run mssanitizer memcheck + racecheck")
    parser.add_argument("--profile", action="store_true", help="Run msprof op PipeUtilization,Memory")
    args = parser.parse_args()

    env = {**os.environ, "NPU_ARCH": args.npu_arch, "DEVICE_ID": str(args.device_id)}
    tmp = Path(tempfile.mkdtemp(prefix="nd-bmm-npu."))
    try:
        for fname in ("kernel.asc", "main.asc", "CMakeLists.txt", "build.sh"):
            shutil.copy2(ROOT / fname, tmp / fname)
        build_dir = tmp / "build"
        cmake_cmd = ["cmake", "-S", str(tmp), "-B", str(build_dir), f"-DNPU_ARCH={args.npu_arch}", "-DCMAKE_BUILD_TYPE=Release"]
        if args.memcheck:
            cmake_cmd.append("-DCMAKE_ASC_FLAGS=--cce-enable-sanitizer -g")
        subprocess.run(cmake_cmd, env=env, check=True)
        subprocess.run(["cmake", "--build", str(build_dir), "--parallel", "4"], env=env, check=True)

        binary = build_dir / "nd_bmm_fused"
        subprocess.run([str(binary)], env=env, check=True)
        subprocess.run([str(binary), "--benchmark"], env=env, check=True)

        if args.memcheck:
            for tool in ("memcheck", "racecheck"):
                res = subprocess.run(["mssanitizer", f"--tool={tool}", "--", str(binary)],
                                     env=env, capture_output=True, text=True, check=False)
                print(res.stdout[-800:] + res.stderr[-400:])
                if res.returncode != 0 or "No error detected" not in (res.stdout + res.stderr):
                    print(f"ERROR: mssanitizer {tool} failed", file=sys.stderr)
                    return 1

        if args.profile:
            out_dir = ROOT / "reports" / "msprof"
            out_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["msprof", "op", f"--output={out_dir}", "--aic-metrics=PipeUtilization,Memory", str(binary)],
                           env=env, check=True)

        print("03-ND-BATCHED-MATMUL-FUSION NPU SUITE PASSED!")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
