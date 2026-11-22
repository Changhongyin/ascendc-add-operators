#!/usr/bin/env python3
"""Local automated verification suite for op179089 next-stage code and cloud scripts."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
BASELINE_ASC = ROOT / "baseline" / "op179089.asc"
NEXT_ASC = ROOT / "next" / "op179089.asc"
PATCH_FILE = ROOT / "op179089.patch"
MOCK_HEADERS = ROOT / "tests" / "mock_headers"
SCRIPTS_DIR = ROOT / "scripts"


def run(cmd, cwd=ROOT, env=None, check=True):
    res = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)
    if check and res.returncode != 0:
        raise AssertionError(f"Command failed ({res.returncode}): {cmd}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    return res


def test_patch_and_compile():
    print("[1/5] Generating unified diff & compiling next/op179089.asc with clang++ -Wall -Wextra -Werror ...")
    diff_res = run(["diff", "-u", "baseline/op179089.asc", "next/op179089.asc"], check=False)
    assert diff_res.returncode == 1, "Expected diff between baseline and next"
    PATCH_FILE.write_bytes(diff_res.stdout.replace("\r\n", "\n").encode("utf-8"))

    # Only the repeat-aligned in-UB vector Add split is permitted inside AddKernel.
    base_txt = BASELINE_ASC.read_text(encoding="utf-8")
    next_txt = NEXT_ASC.read_text(encoding="utf-8")
    k_start = "class AddKernel {"
    k_end = "\n};"
    base_kernel = base_txt[base_txt.index(k_start):base_txt.index(k_end, base_txt.index(k_start)) + len(k_end)]
    next_kernel = next_txt[next_txt.index(k_start):next_txt.index(k_end, next_txt.index(k_start)) + len(k_end)]
    expected_add = (
        "const uint32_t wholeRepeats = count - (count % kVectorRepeatElements);\n"
        "        if (wholeRepeats != 0) {\n"
        "            AscendC::Add(z, x, y, wholeRepeats);\n"
        "        }\n"
        "        const uint32_t tailElements = count - wholeRepeats;\n"
        "        if (tailElements != 0) {\n"
        "            AscendC::Add(z[wholeRepeats], x[wholeRepeats], y[wholeRepeats], tailElements);\n"
        "        }"
    )
    assert base_kernel.replace("AscendC::Add(z, x, y, vectorCount);", expected_add) == next_kernel

    tmp_dir = Path(tempfile.mkdtemp(prefix="op179089-sim."))
    sim_cpp = tmp_dir / "op179089_sim.cpp"
    sim_bin = tmp_dir / "op179089_sim"
    launch_line = "    op179089<<<blocks, nullptr, stream>>>(inputs, output);"
    assert next_txt.count(launch_line) == 1
    sim_launch = (
        "    (void)stream;\n"
        "    AscendCSim::gBlockNum = blocks;\n"
        "    for (uint32_t b = 0; b < blocks; ++b) {\n"
        "        AscendCSim::gBlockIdx = b;\n"
        "        op179089(inputs, output);\n"
        "    }"
    )
    sim_cpp.write_text(next_txt.replace(launch_line, sim_launch, 1), encoding="utf-8")

    run([
        "clang++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{MOCK_HEADERS}",
        str(sim_cpp),
        "-o",
        str(sim_bin),
    ])
    print("    OK: clang++ compiled cleanly with -Wall -Wextra -Werror; AddKernel tail mask verified")
    return tmp_dir, sim_bin


def test_simulated_suite(tmp_dir: Path, sim_bin: Path):
    print("[2/5] Running scripts/run_npu_suite.py against simulated op179089 binary ...")
    out_dir = tmp_dir / "suite_reports"
    res = run([
        sys.executable,
        str(SCRIPTS_DIR / "run_npu_suite.py"),
        "--binary",
        str(sim_bin),
        "--skip-benchmark",
        "--out-dir",
        str(out_dir),
    ])
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["validation"]["status"] == "PASS"
    assert summary["validation"]["pass_count"] == 25
    assert summary["invalid_args"]["status"] == "PASS"
    assert summary["invalid_args"]["passed"] == 11
    assert summary["shape_checks"]["status"] == "PASS"
    assert summary["shape_checks"]["passed"] == 4

    tools_dir = tmp_dir / "tools"
    tools_dir.mkdir()
    for tool in ("msprof", "mssanitizer"):
        executable = tools_dir / tool
        executable.write_text("#!/bin/sh\nexit 17\n", encoding="utf-8")
        executable.chmod(0o755)
    env = {**os.environ, "PATH": os.pathsep.join((str(tools_dir), os.environ["PATH"]))}
    failed_dir = tmp_dir / "failed_tools"
    failed = run([
        sys.executable,
        str(SCRIPTS_DIR / "run_npu_suite.py"),
        "--binary", str(sim_bin),
        "--skip-benchmark", "--profile", "--sanitize",
        "--out-dir", failed_dir.name,
    ], cwd=tmp_dir, env=env, check=False)
    assert failed.returncode == 1, failed.stderr
    failure_report = json.loads((failed_dir / "summary.json").read_text(encoding="utf-8"))
    assert [item["returncode"] for item in failure_report["msprof"]] == [17, 17]
    assert [item["returncode"] for item in failure_report["mssanitizer"]] == [17, 17]
    print("    OK: 25/25 validation, 11/11 invalid-arg (exit 2), 4/4 shape and length boundary checks PASSED")

    print("[3/5] Testing multi-core scaling (40 / 48 / 64 cores), OP179089_DEVICE_ID, and --benchmark output ...")
    for cores, expected_blocks in ((40, 40), (48, 48), (64, 64), (96, 64)):
        env = {**os.environ, "SIM_VECTOR_CORES": str(cores), "OP179089_DEVICE_ID": "1"}
        r = run([str(sim_bin), "131072"], env=env)
        assert f"PASS N=131072 (blocks={expected_blocks}," in r.stdout, r.stdout

    # Test single-case benchmark (fast N=1024 and N-D 16x257)
    r_bench = run([str(sim_bin), "--benchmark", "1024"])
    assert re.search(r"BENCH N=1024 blocks=1 median_us=[0-9.]+ min_us=[0-9.]+ max_us=[0-9.]+ bw_gbs=[0-9.]+", r_bench.stdout), r_bench.stdout
    r_bench_nd = run([str(sim_bin), "--benchmark", "--shape", "16x257"])
    assert "PASS N=4112 shape=16x257 (blocks=5," in r_bench_nd.stdout, r_bench_nd.stdout
    assert re.search(r"BENCH N=4112 blocks=5 .* bw_gbs=[0-9.]+", r_bench_nd.stdout), r_bench_nd.stdout
    print("    OK: A2/A3/A5 core scaling (40/48/64), device override, and BENCH bw_gbs format PASSED")


def test_parse_msprof(tmp_dir: Path):
    print("[4/5] Testing scripts/parse_msprof.py on CANN 9.0 msprof log & CSV fixtures ...")
    prof_dir = tmp_dir / "mock_msprof"
    prof_dir.mkdir()
    (prof_dir / "msprof.log").write_text(
        "\n".join([
            "[INFO] Op Name: op179089",
            "[INFO] Block Dim: 40",
            "[INFO] Task Duration(us): 4.210",
            "[INFO] Task Duration(us): 4.180",
            "[INFO] Task Duration(us): 4.250",
            "[INFO] Task Duration(us): 4.190",
            "[INFO] Task Duration(us): 4.220",
        ]) + "\n",
        encoding="utf-8",
    )
    (prof_dir / "PipeUtilization.csv").write_text(
        "Op Name,aiv_vec_ratio,aiv_mte2_ratio,aiv_mte3_ratio,aiv_ub_read_bw(GB/s)\n"
        "op179089,0.28,0.82,0.74,1120.5\n"
        "op179089,0.30,0.85,0.76,1150.0\n",
        encoding="utf-8",
    )
    base_json = tmp_dir / "baseline_prof.json"
    base_json.write_text(json.dumps({"log": {"median_us": 25.0}}), encoding="utf-8")

    json_out = tmp_dir / "parsed_prof.json"
    md_out = tmp_dir / "parsed_prof.md"
    run([
        sys.executable,
        str(SCRIPTS_DIR / "parse_msprof.py"),
        str(prof_dir),
        "--baseline-json",
        str(base_json),
        "--json-out",
        str(json_out),
        "--md-out",
        str(md_out),
    ])
    parsed = json.loads(json_out.read_text(encoding="utf-8"))
    assert parsed["log"]["samples"] == 5
    assert parsed["log"]["median_us"] == 4.21
    assert parsed["log"]["block_dim"] == 40
    assert parsed["csv_metrics"]["aiv_mte2_ratio"]["median"] == 0.835
    md_txt = md_out.read_text(encoding="utf-8")
    assert "25.000 → 4.210" in md_txt
    print("    OK: parse_msprof.py log + CSV + baseline speedup comparison PASSED")


def test_apply_to_cloud(tmp_dir: Path):
    print("[5/5] Testing scripts/apply_to_cloud.py backup, checksum guard, and atomic install ...")
    proj = tmp_dir / "mock_cloud_proj"
    proj.mkdir()
    shutil.copy2(BASELINE_ASC, proj / "op179089.asc")
    base_sha = hashlib.sha256(BASELINE_ASC.read_bytes()).hexdigest()
    next_sha = hashlib.sha256(NEXT_ASC.read_bytes()).hexdigest()

    # Checksum mismatch must fail with exit 3
    bad = run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "apply_to_cloud.py"),
            "--project-dir",
            str(proj),
            "--source",
            str(NEXT_ASC),
            "--expected-baseline-sha256",
            "0" * 64,
        ],
        check=False,
    )
    assert bad.returncode == 3, bad.returncode

    # Dry-run must succeed without modifying target
    run([
        sys.executable,
        str(SCRIPTS_DIR / "apply_to_cloud.py"),
        "--project-dir",
        str(proj),
        "--source",
        str(NEXT_ASC),
        "--expected-baseline-sha256",
        base_sha,
        "--dry-run",
    ])
    assert hashlib.sha256((proj / "op179089.asc").read_bytes()).hexdigest() == base_sha

    # Real install must create backup and match next_sha
    run([
        sys.executable,
        str(SCRIPTS_DIR / "apply_to_cloud.py"),
        "--project-dir",
        str(proj),
        "--source",
        str(NEXT_ASC),
        "--expected-baseline-sha256",
        base_sha,
    ])
    assert hashlib.sha256((proj / "op179089.asc").read_bytes()).hexdigest() == next_sha
    backups = list((proj / ".backups").glob("op179089.asc.*.bak"))
    assert len(backups) == 1
    assert hashlib.sha256(backups[0].read_bytes()).hexdigest() == base_sha
    print("    OK: apply_to_cloud.py checksum guard, dry-run, backup, and atomic replace PASSED")


def main() -> int:
    tmp_dir, sim_bin = test_patch_and_compile()
    try:
        test_simulated_suite(tmp_dir, sim_bin)
        test_parse_msprof(tmp_dir)
        test_apply_to_cloud(tmp_dir)
        print("\nALL LOCAL AUTOMATED TESTS PASSED!")
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
