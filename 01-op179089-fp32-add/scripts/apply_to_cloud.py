#!/usr/bin/env python3
"""Safe, atomic installer for op179089.asc on the cloud Ascend C project."""

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomically install updated op179089.asc with automatic backup.")
    parser.add_argument("--project-dir", type=Path, required=True, help="Path to cloud op179089 directory")
    parser.add_argument("--source", type=Path, required=True, help="Path to new op179089.asc")
    parser.add_argument("--expected-baseline-sha256", default="", help="Abort if target SHA-256 differs")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and checksums without writing")
    args = parser.parse_args()

    target = args.project_dir / "op179089.asc"
    if not args.source.is_file():
        print(f"ERROR: source file not found: {args.source}", file=sys.stderr)
        return 2
    if not target.is_file():
        print(f"ERROR: target file not found: {target}", file=sys.stderr)
        return 2

    old_bytes = target.read_bytes()
    new_bytes = args.source.read_bytes()
    old_hash = sha256_bytes(old_bytes)
    new_hash = sha256_bytes(new_bytes)

    if args.expected_baseline_sha256 and old_hash != args.expected_baseline_sha256:
        print(
            f"ERROR: target SHA-256 mismatch.\n  expected: {args.expected_baseline_sha256}\n  actual:   {old_hash}",
            file=sys.stderr,
        )
        return 3

    print(f"Target : {target}")
    print(f"Old SHA: {old_hash} ({len(old_bytes)} bytes)")
    print(f"New SHA: {new_hash} ({len(new_bytes)} bytes)")

    if old_bytes == new_bytes:
        print("No changes needed; target already matches source.")
        return 0

    if args.dry_run:
        print("[DRY-RUN] Validation passed. No files modified.")
        return 0

    backup_dir = args.project_dir / ".backups"
    backup_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"op179089.asc.{stamp}.bak"
    shutil.copy2(target, backup_path)
    print(f"Backup : {backup_path}")

    mode = target.stat().st_mode & 0o777
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.project_dir, prefix=".op179089.asc.", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(new_bytes)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.fchmod(tmp.fileno(), mode)
        os.replace(tmp_path, target)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    if sha256_bytes(target.read_bytes()) != new_hash:
        print("ERROR: post-install checksum verification failed!", file=sys.stderr)
        return 4

    print("Installed and verified op179089.asc successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
