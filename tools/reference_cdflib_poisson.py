"""Original CDFLIB90 poisson forward and inverse reference cases."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use cdf_poisson_mod
implicit none
integer which,status
real(8) p,q,s,lambda
read(*,*) which,p,q,s,lambda
call cdf_poisson(which,cum=p,ccum=q,s=s,lambda=lambda,status=status)
write(*,'(I5,4ES26.17E3)') status,p,q,s,lambda
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    work = Path("research/raw/reference/cdflib-poisson").resolve()
    work.mkdir(parents=True, exist_ok=True)
    names = [
        "biomath_constants_mod",
        "biomath_strings_mod",
        "biomath_sort_mod",
        "biomath_interface_mod",
        "biomath_mathlib_mod",
        "zero_finder",
        "cdf_aux_mod",
        "cdf_gamma_mod",
        "cdf_poisson_mod",
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
        if len(fields) != 5:
            raise RuntimeError(f"unexpected native output: {result.stdout}")
        return int(fields[0]), [float(v) for v in fields[1:]]

    cases = []
    for count, mean in itertools.product((0.0, 0.2, 1.0, 4.0, 10.0), (0.01, 0.2, 1.0, 5.0, 30.0)):
        inputs = [1, 0.0, 0.0, count, mean]
        status, values = run(inputs)
        cases.append({"input": inputs, "status": status, "result": values})
        if min(values[:2]) < 1e-8:
            continue
        for which in (2, 3):
            inputs = [which, values[0], values[1], count, mean]
            status, inverted = run(inputs)
            cases.append({"input": inputs, "status": status, "result": inverted})
    unattainable = []
    for p in (0.1, 0.2):
        inputs = [2, p, 1 - p, 0.0, 1.0]
        status, values = run(inputs)
        unattainable.append({"input": inputs, "status": status, "result": values})
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    output = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command,
        "driver": DRIVER,
        "adaptations": [],
        "input_order": ["which", "cum", "ccum", "s", "lambda"],
        "result_order": ["cum", "ccum", "s", "lambda"],
        "cases": cases,
        "unattainable_count_cases": unattainable,
    }
    target = Path("tests/fixtures/cdflib_poisson.json")
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(cases)} poisson cases")


if __name__ == "__main__":
    main()
