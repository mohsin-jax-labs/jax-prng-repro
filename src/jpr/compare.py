"""Compare two JSON outputs for equality of the model hash (and optionally metrics)."""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("file_a", type=Path)
    p.add_argument("file_b", type=Path)
    p.add_argument("--check-metrics", action="store_true", help="Also require exact equality of loss/acc floats.")
    a = p.parse_args()

    ja = json.loads(a.file_a.read_text())
    jb = json.loads(a.file_b.read_text())

    ok = ja.get("hash") == jb.get("hash")
    if a.check_metrics:
        ok = ok and (ja.get("final") == jb.get("final"))

    if ok:
        print("✅ MATCH: hashes (and metrics if requested) are equal.")
        sys.exit(0)
    print("❌ MISMATCH: results differ.")
    print("A:", ja)
    print("B:", jb)
    sys.exit(1)


if __name__ == "__main__":
    main()
