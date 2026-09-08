"""Generate original CDFLIB90 beta references without copying routines into the package."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use cdf_beta_mod
implicit none
integer which,status
real(8) p,q,x,y,a,b
read(*,*) which,p,q,x,y,a,b
call cdf_beta(which,cum=p,ccum=q,x=x,cx=y,a=a,b=b,status=status)
write(*,'(I5,6ES26.17E3)') status,p,q,x,y,a,b
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    work = Path("research/raw/reference/cdflib-beta").resolve()
    work.mkdir(parents=True, exist_ok=True)
    names = [
        "biomath_constants_mod",
        "biomath_strings_mod",
        "biomath_sort_mod",
        "biomath_interface_mod",
        "biomath_mathlib_mod",
        "zero_finder",
        "cdf_aux_mod",
        "cdf_beta_mod",
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
        completed = subprocess.run(
            [str(executable)],
            input=" ".join(map(str, values)) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        fields = completed.stdout.split()
        if len(fields) != 7:
            raise RuntimeError(f"unexpected native output: {completed.stdout}")
        return int(fields[0]), [float(x) for x in fields[1:]]

    cases = []
    for a, b, x in itertools.product(
        (0.2, 1.0, 3.0, 25.0), (0.3, 1.0, 5.0, 40.0), (0.01, 0.2, 0.7, 0.99)
    ):
        inputs = [1, 0.0, 0.0, x, 1 - x, a, b]
        status, result = run(inputs)
        cases.append({"input": inputs, "status": status, "result": result})
        if status != 0:
            raise RuntimeError("unexpected native forward failure")
        p, q = result[:2]
        if min(p, q) < 1e-8:
            continue
        for which in (2, 3, 4):
            inputs = [which, p, q, x, 1 - x, a, b]
            status, result = run(inputs)
            cases.append({"input": inputs, "status": status, "result": result})
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    fixture = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command,
        "driver": DRIVER,
        "adaptations": [],
        "input_order": ["which", "cum", "ccum", "x", "cx", "a", "b"],
        "result_order": ["cum", "ccum", "x", "cx", "a", "b"],
        "cases": cases,
    }
    target = Path("tests/fixtures/cdflib_beta.json")
    target.write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        f"Recorded {len(cases)} native beta cases; "
        f"{sum(c['status'] != 0 for c in cases)} nonzero statuses"
    )


if __name__ == "__main__":
    main()
