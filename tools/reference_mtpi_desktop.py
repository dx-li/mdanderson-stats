"""Regenerate synthetic mTPI references from user-supplied official assets.

Original workbook/XLL/R files are not redistributed. Requires pefile, unicorn,
SciPy and Rscript. The controller probe substitutes SciPy for the beta CDF; the
RNG/ranking probes execute original instructions. Cached workbook results are
original saved outputs, not a fresh Excel execution.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as E
import zipfile
from pathlib import Path

HASHES = {
    "mTPI.xlsm": "7bfa42bc88f1fe1aaf437d5b902770aef53590b488526ce5d320b410530ced19",
    "mTPI.xll": "c946b1bc48baafe1f6477767a6d4f64f97ca7ff95398c9b29ed9159f79d0c858",
    "mTPI-middle.R": "2629c28a1bf607329b125d7fca6715d8d1bfd0615162ebc4d091b2c603f5f860",
}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--rscript", default="Rscript")
    args = parser.parse_args()
    for name, expected in HASHES.items():
        if hashlib.sha256((args.assets / name).read_bytes()).hexdigest() != expected:
            raise ValueError("original asset hash mismatch: " + name)
    args.output.mkdir(parents=True, exist_ok=True)
    drivers = Path(__file__).parent / "mtpi_desktop_reference"
    for driver, suffix in (
        ("decisions", "xll-decisions.json"),
        ("rng", "uniforms.json"),
        ("rank", "rank.jsonl"),
    ):
        result = subprocess.run(
            [sys.executable, str(drivers / (driver + ".py")), str(args.assets / "mTPI.xll")],
            check=True,
            capture_output=True,
            text=True,
        )
        (args.output / ("mtpi-desktop-" + suffix)).write_text(result.stdout)
    subprocess.run(
        [
            args.rscript,
            "--vanilla",
            str(drivers / "trials.R"),
            str(args.assets / "mTPI-middle.R"),
            str(args.output / "mtpi-desktop-r-trials.csv"),
        ],
        check=True,
    )
    with zipfile.ZipFile(args.assets / "mTPI.xlsm") as archive:
        shared = ["".join(n.itertext()) for n in E.fromstring(archive.read("xl/sharedStrings.xml"))]

        def cells(sheet):
            root = E.fromstring(archive.read(f"xl/worksheets/sheet{sheet}.xml"))
            return {
                c.attrib["r"]: (
                    shared[int(c.findtext("m:v", namespaces=NS))]
                    if c.attrib.get("t") == "s"
                    else c.findtext("m:v", namespaces=NS)
                )
                for c in root.findall(".//m:c", NS)
                if c.find("m:v", NS) is not None
            }

        table = cells(1)

        def column(n):
            value = ""
            while n:
                n, r = divmod(n - 1, 26)
                value = chr(65 + r) + value
            return value

        rows = [
            dict(n=n, y=y, action=table[column(n + 3) + str(y + 4)])
            for n in range(1, 31)
            for y in range(n + 1)
        ]
        data = dict(
            target=float(table["B1"]), eps1=float(table["C1"]), eps2=float(table["D1"]), rows=rows
        )
        (args.output / "mtpi-desktop-cached-table.json").write_text(
            json.dumps(data, indent=2) + "\n"
        )
        sim = cells(2)
        S, D = int(sim["G1"]), int(sim["C1"])
        data = dict(
            trials=int(sim["A1"]),
            target=float(sim["B1"]),
            epsilon_lower=float(sim["H1"]),
            epsilon_upper=float(sim["I1"]),
            maximum_patients=int(sim["D1"]),
            cohort_size=int(sim["E1"]),
            starting_dose=int(sim["F1"]),
            scenarios=[
                [float(sim[column(j + 2) + str(i + 3)]) for j in range(D)] for i in range(S)
            ],
        )
        expected = dict(
            selection=[
                [float(sim[column(j + 2) + str(25 + 6 * i)]) for j in range(D + 1)]
                for i in range(S)
            ],
            mean_patients=[
                [float(sim[column(j + 2) + str(26 + 6 * i)]) for j in range(D)] for i in range(S)
            ],
            pooled_toxicity=[float(sim["B" + str(28 + 6 * i)]) for i in range(S)],
        )
        (args.output / "mtpi-desktop-cached-simulation.json").write_text(
            json.dumps(dict(study=data, expected=expected), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
