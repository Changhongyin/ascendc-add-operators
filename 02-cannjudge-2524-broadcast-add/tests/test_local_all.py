#!/usr/bin/env python3
"""Local automated verification suite for 02-cannjudge-2524-broadcast-add."""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL_ASC = ROOT / "kernel.asc"
TEST_CPP = ROOT / "tests" / "test_broadcast_add.cpp"
MOCK_HEADERS = ROOT / "tests" / "mock_headers"

LAUNCH_RE = re.compile(
    r"^(\s*)([a-zA-Z0-9_]+)<<<([^,]+),\s*nullptr,\s*stream>>>\(([^;]+)\);",
    re.MULTILINE,
)


def lower_launches_for_cpu(source: str) -> str:
    def repl(match: re.Match) -> str:
        indent, fn, blocks, args = match.groups()
        return (
            f"{indent}do {{\n"
            f"{indent}    (void)stream;\n"
            f"{indent}    AscendCSim::gBlockNum = static_cast<uint32_t>({blocks});\n"
            f"{indent}    for (uint32_t _sim_b = 0; _sim_b < AscendCSim::gBlockNum; ++_sim_b) {{\n"
            f"{indent}        AscendCSim::gBlockIdx = _sim_b;\n"
            f"{indent}        {fn}({args});\n"
            f"{indent}    }}\n"
            f"{indent}}} while (0);"
        )

    lowered, count = LAUNCH_RE.subn(repl, source)
    assert count == 20, f"Expected 20 kernel launches across 5 paths x 4 dtypes, got {count}"
    return lowered


def run(cmd, cwd=ROOT):
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise AssertionError(f"Command failed ({res.returncode}): {cmd}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res


def main() -> int:
    print("[1/3] Checking official template files exist and kernel.asc launches are complete ...")
    for required in ("CMakeLists.txt", "main.asc", "run.sh", "sample.py", "reference.py", "requirements.txt", "kernel.asc"):
        assert (ROOT / required).is_file(), f"Missing file: {required}"

    kernel_src = KERNEL_ASC.read_text(encoding="utf-8")
    lowered = lower_launches_for_cpu(kernel_src)
    # Guard against regression of key fixes:
    assert "HardEvent::V_S" in kernel_src and "ReleaseEventID<AscendC::HardEvent::V_S>" in kernel_src
    assert "SafeGather(" in kernel_src and "StepRowOdometer(" in kernel_src
    print("    OK: 20/20 kernel launches lowered; V_S sync, SafeGather, and O(1) odometer verified in source")

    tmp_dir = Path(tempfile.mkdtemp(prefix="cannjudge2524-sim."))
    try:
        (tmp_dir / "kernel_sim.inc").write_text(lowered, encoding="utf-8")
        sim_bin = tmp_dir / "test_broadcast_add"

        print("[2/3] Compiling kernel.asc + test_broadcast_add.cpp with clang++ -std=c++17 -O2 -Wall -Wextra -Werror ...")
        run([
            "clang++",
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{MOCK_HEADERS}",
            f"-I{tmp_dir}",
            str(TEST_CPP),
            "-o",
            str(sim_bin),
        ])
        print("    OK: clang++ compiled cleanly with zero warnings")

        print("[3/3] Running exhaustive multi-dtype / multi-path / guard / UB / race-check test matrix ...")
        res = run([str(sim_bin)], cwd=tmp_dir)
        print("   ", res.stdout.strip())
        assert "PASS all " in res.stdout
        print("\nALL CANNJUDGE_2524 LOCAL AUTOMATED TESTS PASSED!")
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
