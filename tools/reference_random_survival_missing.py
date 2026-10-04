"""Verify pinned RF-SRC sources and generate deterministic missing-data ledgers."""

from __future__ import annotations

import argparse
import hashlib
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_BLOBS = {
    "src/randomForestSRC.c": "e9c6e896c4f93c6eb1b85bdf37992964d74376ea",
    "R/rfsrc.R": "5be0607d365a5422e4b1d58e03c3e0dfe92c8045",
    "R/utilities.R": "25b40cd53edc9489f53e7799383d5db74494f6ba",
    "R/utilities.data.R": "bac4cfb07e838cbcae029120d0bf5307c02d9b17",
    "man/rfsrc.Rd": "6f441a8d027d20764ce75d26a3aefcc54188ee82",
}


def git_blob(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=ROOT.parent / "mdanderson-stats/research/raw/randomForestSRC",
        help="ignored cache for the pinned randomForestSRC source bundle",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "tests/fixtures", help="ledger output directory"
    )
    parser.add_argument("--rscript", default="Rscript", help="Rscript executable")
    args = parser.parse_args()

    for relative, expected in PINNED_BLOBS.items():
        path = args.source_root / relative
        actual = git_blob(path)
        if actual != expected:
            raise SystemExit(f"pinned RF-SRC blob mismatch for {relative}: {actual}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    subprocess.run(
        [
            args.rscript,
            str(ROOT / "tools/reference_random_survival_missing.R"),
            str(args.output_dir),
        ],
        check=True,
        timeout=30,
        cwd=ROOT,
    )
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_rss_bytes = int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss * 1024)
    print(
        f"reference wall={time.perf_counter() - started:.3f}s; "
        f"child peak RSS={peak_rss_bytes} bytes; child swaps={usage.ru_nswap}"
    )


if __name__ == "__main__":
    main()
