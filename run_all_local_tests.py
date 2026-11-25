#!/usr/bin/env python3
"""Top-level local verification runner for both Ascend C Add operator projects."""

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parent
OP1_DIR = ROOT / "01-op179089-fp32-add"
OP2_DIR = ROOT / "02-cannjudge-2524-broadcast-add"
OP3_DIR = ROOT / "03-nd-batched-matmul-fusion"
TEMPLATE_ZIP = ROOT / "99-archive" / "add_problem_2524_template.zip"

OFFICIAL_TEMPLATE_FILES = (
    "CMakeLists.txt",
    "main.asc",
    "run.sh",
    "sample.py",
    "reference.py",
    "requirements.txt",
    "README.md",
)


def run(cmd, cwd: Path):
    print(f"\n>>> [{cwd.name}] {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(cwd), check=False)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def verify_template_integrity():
    print("\n>>> [02-cannjudge-2524-broadcast-add] Verifying official template files remain untouched ...")
    with zipfile.ZipFile(TEMPLATE_ZIP, "r") as zf:
        by_name = {Path(info.filename).name: info.filename for info in zf.infolist() if not info.is_dir()}
        for fname in OFFICIAL_TEMPLATE_FILES:
            assert fname in by_name, f"Template zip missing {fname}"
            expected_sha = hashlib.sha256(zf.read(by_name[fname])).hexdigest()
            actual_sha = hashlib.sha256((OP2_DIR / fname).read_bytes()).hexdigest()
            assert actual_sha == expected_sha, f"Template file modified: {fname}"
    print(f"    OK: {len(OFFICIAL_TEMPLATE_FILES)}/{len(OFFICIAL_TEMPLATE_FILES)} official template files byte-identical to archive")


def run_op1_tests_isolated():
    print("\n>>> [01-op179089-fp32-add] Running 5-stage local test suite in isolated /tmp copy (preserving op179089.patch SHA256) ...")
    tmp = Path(tempfile.mkdtemp(prefix="op179089-local."))
    try:
        copy_dir = tmp / "01-op179089-fp32-add"
        shutil.copytree(OP1_DIR, copy_dir)
        run([sys.executable, "tests/test_local_all.py"], cwd=copy_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("==================================================================")
    print("  AscendC算子整理汇总 - 全仓离线自动化验证 (Project 01 + 02 + 03)")
    print("==================================================================")
    run(["shasum", "-a", "256", "-c", "SHA256SUMS.txt"], cwd=OP1_DIR)
    run_op1_tests_isolated()
    verify_template_integrity()
    run([sys.executable, "tests/test_local_all.py"], cwd=OP2_DIR)
    run([sys.executable, "tests/test_local_all.py"], cwd=OP3_DIR)
    print("\n==================================================================")
    print("  ALL REPOSITORY LOCAL TESTS & INTEGRITY CHECKS PASSED!")
    print("==================================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
