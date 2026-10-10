"""Regenerate references from the checksum-verified original CI-IIV2 archive.

Original source stays a local research input. Only this authored harness and
synthetic factual outputs ship. A recording RNG wrapper observes base-R draws;
original function bodies are unchanged and production Python is not imported.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

ARCHIVE_SHA256 = "e69f045d6c15595cc67e5055c94cc8ad119230e07e334a2def11708a0f1712f1"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "tests/fixtures/interaction-index-native"


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("original source archive checksum mismatch")
    with tempfile.TemporaryDirectory(prefix="interaction-index-native-") as temporary:
        root = Path(temporary)
        with zipfile.ZipFile(args.archive) as archive:
            for name in (
                "CI_IIV2.SSC",
                "Simulation1_3_drugsV2.SSC",
                "Simulation2_fixed_ratioV2.SSC",
            ):
                (root / name).write_bytes(archive.read(name))
        output = root / "outputs"
        result = subprocess.run(
            [
                args.rscript,
                str(ROOT / "tools/reference_interaction_index_native.R"),
                str(root),
                str(output),
            ],
            env=os.environ,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        files = sorted(output.glob("*.csv"))
        expected = {
            f"{stem}-{case}.csv"
            for stem in ("draws", "inputs", "reference", "observed")
            for case in (1, 2)
        } | {"source-truth.csv"}
        if {file.name for file in files} != expected:
            raise ValueError("original R execution did not produce all nine reference tables")
        if args.check:
            for file in files:
                if file.read_bytes() != (DESTINATION / file.name).read_bytes():
                    raise ValueError(f"original numerical reference changed: {file.name}")
            print(f"Verified {len(files)} original-function numerical tables")
        else:
            DESTINATION.mkdir(exist_ok=True)
            for file in files:
                (DESTINATION / file.name).write_bytes(file.read_bytes())
            print(f"Wrote {len(files)} synthetic original-function numerical tables")


if __name__ == "__main__":
    main()
