"""Generate bounded, original-instruction reference fixtures from a supplied DLL."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from arand_common import library_path

FAMILIES = (
    "x86",
    "scheduler",
    "outcome",
    "pending",
    "status",
    "mincriteria",
    "minmax",
    "final",
    "groupfinal",
    "allocation",
    "predictive",
    "integrated",
    "tte_stats",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    os.environ["ARAND_REFERENCE_LIBRARY"] = str(args.library.resolve())
    library_path()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "families": {},
    }
    here = Path(__file__).resolve().parent
    for family in FAMILIES:
        proc = subprocess.run(
            [sys.executable, str(here / f"arand_{family}_reference.py")],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        output = (
            args.output_dir
            / f"arand-{('floor' if family == 'x86' else family.replace('_', '-'))}-fixtures.jsonl"
        )
        output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        manifest["families"][family] = {
            "count": len(rows),
            "fixture_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "families": len(FAMILIES),
                "fixtures": sum(x["count"] for x in manifest["families"].values()),
                "output_dir": str(args.output_dir),
            }
        )
    )


if __name__ == "__main__":
    main()
