"""Record unchanged F95 error-function and exponential support routines."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode,ind
real(8) x,value
read(*,*) mode,ind,x
select case(mode)
case(1)
 value=erf(x)
case(2)
 value=erfc1(ind,x)
case(3)
 value=esum(ind,x)
case(4)
 value=exparg(ind)
end select
write(*,'(ES26.17E3)') value
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-error-exponential").resolve()
    work.mkdir(parents=True, exist_ok=True)
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    exe = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, paths),
        str(driver),
        "-o",
        str(exe),
    ]
    subprocess.run(command, cwd=work, check=True)
    cases = []

    def record(mode, ind, x):
        payload = f"{mode} {ind} {x}\n"
        row = dict(mode=mode, ind=ind, x=x)
        try:
            r = subprocess.run(
                [str(exe)], input=payload, text=True, capture_output=True, check=True, timeout=3
            )
        except subprocess.CalledProcessError as error:
            row.update(
                execution_outcome="process_error", stderr=error.stderr, exit_code=error.returncode
            )
        except subprocess.TimeoutExpired:
            row.update(execution_outcome="timeout", timeout_seconds=3)
        else:
            value = float(r.stdout)
            row.update(
                execution_outcome="completed", result=value if math.isfinite(value) else str(value)
            )
        cases.append(row)

    coordinates = [
        -1e308,
        -30.0,
        -27.0,
        -26.0,
        -5.8,
        -5.6,
        -4.0,
        -0.5,
        -1e-10,
        -5e-324,
        0.0,
        5e-324,
        1e-10,
        0.5,
        4.0,
        5.6,
        5.8,
        26.0,
        26.6,
        26.7,
        27.0,
        27.2,
        27.3,
        28.0,
        30.0,
        100.0,
        1e154,
        1e308,
    ]
    for x in coordinates:
        record(1, 0, x)
        for ind in [0, 1, -2]:
            record(2, ind, x)
    for mu, x in [
        (0, 0.0),
        (1, 0.25),
        (-1, -0.25),
        (700, -699.5),
        (-700, 699.5),
        (1000, -999.5),
        (1000, -1000.5),
        (-1000, 999.5),
        (-1000, 1000.5),
        (710, -1.0),
        (-745, 1.0),
        (0, -745.0),
        (0, -746.0),
        (0, 710.0),
        (2147483647, -2147483647.0),
        (-2147483648, 2147483648.0),
    ]:
        record(3, mu, x)
    for ind in [0, 1, -1, 2, 2147483647, -2147483648]:
        record(4, ind, 0.0)
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    data = dict(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        command=command,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        driver=DRIVER,
        adaptations=[],
        cases=cases,
    )
    Path("tests/fixtures/cdflib_error_exponential.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
