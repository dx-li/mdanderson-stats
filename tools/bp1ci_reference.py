"""Generate output fixtures from a locally compiled, unmodified BP1CI 2.0.

Usage: uv run python tools/bp1ci_reference.py PATH_TO_EXECUTABLE PATH_TO_ARCHIVE
Only numerical output is committed; the original archive is not redistributed.
"""

import hashlib
import json
import re
import subprocess
import sys
from decimal import Decimal
from pathlib import Path


def main() -> None:
    executable, archive = map(Path, sys.argv[1:])
    cases = []
    for confidence in (0.8, 0.95, 0.99):
        for distribution in ("binomial", "poisson"):
            data = (
                [(0, 30), (1, 30), (12, 30), (30, 30)]
                if distribution == "binomial"
                else [(0.5, None), (1, None), (10, None), (10000, None)]
            )
            for k, n in data:
                if distribution == "binomial":
                    user_input = f"\n1\n{confidence * 100}\nb\n2\n2\n{k} {n}\n0\n"
                else:
                    user_input = f"\n1\n{confidence * 100}\np\n2\n{k}\n0\n"
                process = subprocess.run(
                    [str(executable.resolve())],
                    input=user_input,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                matches = re.findall(
                    r"Low Bound:\s*([\d.E+-]+).*High Bound:\s*([\d.E+-]+)", process.stdout
                )
                if len(matches) != 1:
                    raise RuntimeError(f"Expected one interval: {process.stdout}\n{process.stderr}")
                tokens = matches[0]
                # Half a unit in the last printed place, not an arbitrary tolerance.
                tolerance = max(
                    float(Decimal(10) ** Decimal(v).as_tuple().exponent) / 2 for v in tokens
                )
                cases.append(
                    {
                        "distribution": distribution,
                        "events": k,
                        "trials": n,
                        "confidence": confidence,
                        "bounds": list(map(float, tokens)),
                        "atol": tolerance + 1e-12,
                    }
                )
    result = {
        "source_url": "https://biostatistics.mdanderson.org/SoftwareDownload/"
        "SoftwareFiles/BP1CI/BP1CI_V2.0x.zip",
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "build": "make F90=gfortran F90FLAGS='-O2 -std=legacy'",
        "cases": cases,
    }
    target = Path("tests/fixtures/bp1ci.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Recorded {len(cases)} reference intervals")


if __name__ == "__main__":
    main()
