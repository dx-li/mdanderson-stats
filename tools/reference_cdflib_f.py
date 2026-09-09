"""Original CDFLIB90 F forward and inverse reference cases."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use cdf_f_mod
implicit none
integer which,status
real(8) p,q,f,dfn,dfd
read(*,*) which,p,q,f,dfn,dfd
call cdf_f(which,cum=p,ccum=q,f=f,dfn=dfn,dfd=dfd,status=status)
write(*,'(I5,5ES26.17E3)') status,p,q,f,dfn,dfd
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    work = Path("research/raw/reference/cdflib-f").resolve()
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
        "cdf_f_mod",
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
    for f, dfn, dfd in itertools.product(
        (0.1, 1.0, 3.0, 10.0), (0.2, 1.0, 5.0, 30.0), (0.3, 2.0, 10.0, 40.0)
    ):
        inputs = [1, 0.0, 0.0, f, dfn, dfd]
        status, values = run(inputs)
        cases.append({"input": inputs, "status": status, "result": values})
        if min(values[:2]) < 1e-8:
            continue
        inputs = [2, values[0], values[1], f, dfn, dfd]
        status, inverted = run(inputs)
        cases.append({"input": inputs, "status": status, "result": inverted})
    small_f = []
    for f in (1e-20, 1e-100):
        inputs = [1, 0.0, 0.0, f, 2.0, 2.0]
        status, values = run(inputs)
        small_f.append({"input": inputs, "status": status, "result": values})
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    output = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command,
        "driver": DRIVER,
        "adaptations": [],
        "input_order": ["which", "cum", "ccum", "f", "dfn", "dfd"],
        "result_order": ["cum", "ccum", "f", "dfn", "dfd"],
        "cases": cases,
        "tiny_coordinate_cases": small_f,
    }
    target = Path("tests/fixtures/cdflib_f.json")
    target.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(cases)} F cases")


if __name__ == "__main__":
    main()
