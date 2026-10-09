"""Regenerate independent Multc99 C and arbitrary-precision references.

Run with system Python, GCC and mpmath 1.3.0. Those are reference-generation
dependencies only. --native-only omits mpmath; --check never rewrites fixtures.
Source bytes are checksum-verified and are not edited or bundled at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
URL = "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/Multc99/Multc99_V2.1.zip"
SHA256 = "0fb107937ca619681539a5eee9db09cd549f42ba8ccbe7961cc7794691af6f6c"
FIXTURES = ROOT / "tests/fixtures"


def native_reference() -> dict:
    archive = ROOT / "research/raw/source-recovery-2026-10-08/Multc99_V2.1.zip"
    if not archive.exists():
        with urlopen(URL, timeout=30) as response:
            data = response.read(2 * 1024 * 1024)
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Multc99 archive integrity check failed")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(data)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Multc99 archive integrity check failed")
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        filenames = (
            "header.h",
            "cdflib.h",
            "ranlib.h",
            "zrorj.h",
            "compute.c",
            "boundary.c",
            "cdflib.c",
            "zrorj.c",
            "ranlib.c",
        )
        with zipfile.ZipFile(archive) as source:
            for filename in filenames:
                (folder / filename).write_bytes(source.read("Source/" + filename))
        executable = folder / "reference"
        subprocess.run(
            [
                "gcc",
                "-std=c99",
                "-fcommon",
                "-ffunction-sections",
                "-fdata-sections",
                "-O0",
                "-I",
                str(folder),
                str(ROOT / "tools/reference_multc99_kernel.c"),
                *(
                    str(folder / file)
                    for file in ("compute.c", "boundary.c", "cdflib.c", "zrorj.c", "ranlib.c")
                ),
                "-Wl,--gc-sections",
                "-lm",
                "-o",
                str(executable),
            ],
            check=True,
            timeout=30,
        )
        result = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True, timeout=30
        )
        return json.loads(result.stdout)


def precise_reference(native: dict) -> dict:
    import mpmath as mp

    mp.mp.dps = 45
    cases = []
    for row in native["probabilities"]:
        a, b, c, d, delta = [mp.mpf(str(row[k])) for k in ("aS", "bS", "aE", "bE", "margin")]
        c += row["x"]
        d += row["n"] - row["x"]
        low, high = max(mp.mpf(0), -delta), min(mp.mpf(1), 1 - delta)

        def integrand(p):
            q = p + delta
            if q >= 1:
                return mp.mpf(0)
            density = p ** (a - 1) * (1 - p) ** (b - 1) / mp.beta(a, b)
            if q <= 0:
                return density
            return density * mp.betainc(c, d, q, 1, regularized=True)

        value = mp.betainc(a, b, 0, low, regularized=True) + mp.quad(
            integrand, [low, (low + high) / 2, high]
        )
        case = {k: v for k, v in row.items() if k != "probability"}
        case.update(probability=float(value), probability_decimal=mp.nstr(value, 40))
        cases.append(case)
    return {
        "generator": "mpmath 1.3.0; 45 decimal digits; direct historical Beta-density integral",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    native = native_reference()
    results = [("multc99-native-reference.json", native)]
    if not args.native_only:
        results.append(("multc99-high-precision-reference.json", precise_reference(native)))
    for filename, result in results:
        path = FIXTURES / filename
        if args.check:
            if result != json.loads(path.read_text()):
                raise ValueError(f"Reference differs: {filename}")
        else:
            path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Verified {filename}" if args.check else f"Generated {filename}")


if __name__ == "__main__":
    main()
