"""Evaluate SINGLE's original main-loop support predicates with native Fortran."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/SINGLE/source/single/single.f")
    original = source.read_text()
    zero = re.search(r"^      IF \(\.NOT\. \(grpcnt\(j,1,1\).*?GO TO 620$", original, re.M)
    overlap = re.search(r"^      DO 650 i = 1,kurgrp - 1.*?^  650 CONTINUE$", original, re.M | re.S)
    if zero is None or overlap is None:
        raise RuntimeError("Original main-loop support predicates not found")
    directory = Path("research/raw/reference/single_support").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f"
    driver.write_text(
        "      PROGRAM reference\n"
        "      IMPLICIT NONE\n"
        "      INTEGER kurgrp,i,j,igrp,haszero\n"
        "      DOUBLE PRECISION dose(10,4,2),grpcnt(10,4,2)\n"
        "      READ(*,*) kurgrp\n"
        "      READ(*,*) dose(1:kurgrp,1,1)\n"
        "      READ(*,*) grpcnt(1:kurgrp,1,1)\n"
        "      haszero=0\n"
        "      DO j=1,kurgrp\n" + zero.group() + "\n"
        "      haszero=1\n"
        "  620 CONTINUE\n"
        "      END DO\n"
        "      igrp=kurgrp\n" + overlap.group() + "\n"
        "      WRITE(*,*) haszero,kurgrp-igrp\n"
        "      END\n"
    )
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(driver), "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    inputs = [
        ([-1, 1], [50, 50]),
        ([-1, 0, 1], [50, 0, 50]),
        ([-1, 0, 1], [50, 1e-5, 50]),
        ([-1, 0, 1], [50, 0.9999999e-5, 50]),
        ([0, 0.001], [50, 50]),
        ([0, 0.00100000002], [50, 50]),
        ([0, 0.0010000001], [50, 50]),
        ([0, 0], [50, 50]),
        ([100, 100.1], [50, 50]),
        ([100, 100.10001], [50, 50]),
        ([-100, -99.9], [50, 50]),
        ([1, 1, 2], [50, 0, 50]),
        ([99.9, 100], [50, 50]),
    ]
    cases = []
    for doses, subjects in inputs:
        result = subprocess.run(
            [str(executable)],
            input=f"{len(doses)}\n"
            + " ".join(map(str, doses))
            + "\n"
            + " ".join(map(str, subjects))
            + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        zero_count, overlaps = map(int, result.stdout.split())
        cases.append(
            dict(
                doses=doses,
                subjects=subjects,
                stop_reason="zero_subjects" if zero_count else "dose_overlap" if overlaps else None,
            )
        )
    fixture = dict(
        source=(
            "Unchanged main-loop count predicate and pairwise overlap loop; "
            "independent driver, not the full optimizer"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/single_support.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
