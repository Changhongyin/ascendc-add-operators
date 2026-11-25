#!/usr/bin/env python3
"""Local automated verification suite for 03-nd-batched-matmul-fusion."""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL_ASC = ROOT / "kernel.asc"
TEST_CPP = ROOT / "tests" / "test_nd_bmm.cpp"
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
    assert count == 3, f"Expected 3 kernel launches (fp32/fp16/bf16), got {count}"
    return lowered


def run(cmd, cwd=ROOT):
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise AssertionError(f"Command failed ({res.returncode}): {cmd}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res


def main() -> int:
    print("[1/3] Checking 03-nd-batched-matmul-fusion files & lowering kernel launches ...")
    for f in ("kernel.asc", "main.asc", "CMakeLists.txt", "build.sh", "README.md"):
        assert (ROOT / f).is_file(), f"Missing {f}"
    lowered = lower_launches_for_cpu(KERNEL_ASC.read_text(encoding="utf-8"))

    tmp = Path(tempfile.mkdtemp(prefix="nd-bmm-sim."))
    try:
        (tmp / "kernel_sim.inc").write_text(lowered, encoding="utf-8")
        sim_bin = tmp / "test_nd_bmm"

        print("[2/3] Compiling kernel.asc + test_nd_bmm.cpp with clang++ -std=c++17 -O2 -Wall -Wextra -Werror ...")
        run([
            "clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
            f"-I{MOCK_HEADERS}", f"-I{tmp}", str(TEST_CPP), "-o", str(sim_bin),
        ])
        print("    OK: clang++ compiled cleanly with zero warnings")

        print("[3/3] Running 2D..8D batched matmul + transB + bias + ReLU/SiLU/GELU matrix ...")
        res = run([str(sim_bin)], cwd=tmp)
        print("   ", res.stdout.strip())
        assert "PASS all " in res.stdout
        print("\nALL 03-ND-BATCHED-MATMUL-FUSION LOCAL TESTS PASSED!")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
