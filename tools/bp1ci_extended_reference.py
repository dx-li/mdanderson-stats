"""Check fractional counts and interface limits against the unmodified executable."""

import hashlib
import json
import re
import subprocess
from decimal import Decimal
from pathlib import Path


def main():
    executable = Path("research/raw/bp1ci/bp1ci/SOURCE/bp1ci").resolve()
    cases = []
    for level in (0.1, 95, 99.9999):
        for entry, first, second in [
            ("trials", 1.5, 30.5),
            ("trials", 29.5, 30.5),
            ("failures", 1.25, 2.75),
            ("failures", 0, 2.5),
            ("failures", 2.5, 0),
            ("trials", 1, 1e10),
            ("trials", 1e10 - 1, 1e10),
            ("failures", 1e10, 1e10),
        ]:
            text = f"\n1\n{level}\nb\n{1 if entry == 'failures' else 2}\n2\n{first} {second}\n0\n"
            process = subprocess.run(
                [str(executable)],
                input=text,
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
            bounds = re.findall(
                r"Low Bound:\s*([\d.E+-]+).*High Bound:\s*([\d.E+-]+)", process.stdout
            )
            if len(bounds) != 1:
                raise RuntimeError(process.stdout + process.stderr)
            tokens = bounds[0]
            tolerance = max(
                float(Decimal(10) ** Decimal(v).as_tuple().exponent) / 2 for v in tokens
            )
            cases.append(
                dict(
                    distribution="binomial",
                    entry=entry,
                    events=first,
                    second=second,
                    confidence_percent=level,
                    bounds=list(map(float, tokens)),
                    atol=tolerance + 1e-12,
                )
            )
        for first in (1e-6, 1.25, 1e6, 1e10):
            text = f"\n1\n{level}\np\n2\n{first}\n0\n"
            process = subprocess.run(
                [str(executable)],
                input=text,
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
            bounds = re.findall(
                r"Low Bound:\s*([\d.E+-]+).*High Bound:\s*([\d.E+-]+)", process.stdout
            )
            if len(bounds) != 1:
                raise RuntimeError(process.stdout + process.stderr)
            tokens = bounds[0]
            tolerance = max(
                float(Decimal(10) ** Decimal(v).as_tuple().exponent) / 2 for v in tokens
            )
            cases.append(
                dict(
                    distribution="poisson",
                    entry="failures",
                    events=first,
                    second=None,
                    confidence_percent=level,
                    bounds=list(map(float, tokens)),
                    atol=tolerance + 1e-12,
                )
            )
    for case in cases:
        if case["confidence_percent"] == 0.1 and case["events"] == 1.25 and case["second"] == 2.75:
            case["source_issue"] = (
                "Native lower bound disagrees with independent beta-tail evaluation"
            )
    source = Path("research/raw/bp1ci/bp1ci/SOURCE/bp1ci.f90")
    record = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        notes="Unmodified BP1CI executable, native menu percentage limits and both binomial "
        "entry modes. Printed precision determines tolerance; very narrow/extreme bounds "
        "may be indistinguishable in source output.",
        cases=cases,
    )
    Path("tests/fixtures/bp1ci_extended.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{len(cases)} native intervals")


if __name__ == "__main__":
    main()
