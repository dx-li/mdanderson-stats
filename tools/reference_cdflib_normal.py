"""Original CDFLIB90 normal forward and inverse reference cases."""

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use cdf_normal_mod
implicit none
integer which,status
real(8) p,q,x,mean,sd
read(*,*) which,p,q,x,mean,sd
call cdf_normal(which,cum=p,ccum=q,x=x,mean=mean,sd=sd,status=status)
write(*,'(I5,5ES26.17E3)') status,p,q,x,mean,sd
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    work = Path("research/raw/reference/cdflib-normal").resolve()
    work.mkdir(parents=True, exist_ok=True)
    names = [
        "biomath_constants_mod",
        "biomath_strings_mod",
        "biomath_sort_mod",
        "biomath_interface_mod",
        "biomath_mathlib_mod",
        "zero_finder",
        "cdf_aux_mod",
        "cdf_normal_mod",
    ]
    paths = [source / f"{name}.f90" for name in names]
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    executable = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, paths),
        str(driver),
        "-o",
        str(executable),
    ]
    subprocess.run(command, cwd=work, check=True)

    def run(values):
        result = subprocess.run(
            [str(executable)],
            input=" ".join(map(str, values)) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        fields = result.stdout.split()
        if len(fields) != 6:
            raise RuntimeError(f"unexpected native output: {result.stdout}")
        return int(fields[0]), [float(v) for v in fields[1:]]

    cases = []
    for z, mean, sd in itertools.product(
        (-5.0, -2.0, -0.3, 0.0, 0.8, 3.0, 5.0), (-2.0, 0.0, 5.0), (0.1, 1.0, 3.0)
    ):
        x = mean + sd * z
        inputs = [1, 0.0, 0.0, x, mean, sd]
        status, values = run(inputs)
        if status:
            raise RuntimeError("unexpected native forward error")
        cases.append({"input": inputs, "status": status, "result": values})
        for which in (2, 3) if z == 0 else (2, 3, 4):
            inputs = [which, values[0], values[1], x, mean, sd]
            status, inverted = run(inputs)
            cases.append({"input": inputs, "status": status, "result": inverted})
    failures = []
    for inputs in (
        [4, 0.5, 0.5, 0.0, 0.0, 1.0],
        [4, 0.9, 0.1, -1.0, 0.0, 1.0],
        [4, 0.5, 0.5, 1.0, 0.0, 1.0],
    ):
        status, values = run(inputs)
        failures.append(
            {
                "input": inputs,
                "status": status,
                "result": [v if math.isfinite(v) else str(v) for v in values],
            }
        )
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    output = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command,
        "driver": DRIVER,
        "adaptations": [],
        "input_order": ["which", "cum", "ccum", "x", "mean", "sd"],
        "result_order": ["cum", "ccum", "x", "mean", "sd"],
        "cases": cases,
        "invalid_sd_cases": failures,
    }
    target = Path("tests/fixtures/cdflib_normal.json")
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(cases)} normal cases and {len(failures)} invalid-SD source results")


if __name__ == "__main__":
    main()
