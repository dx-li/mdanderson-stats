"""Record unchanged F95 elementary mathematical support routines."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode,n
real(8) x,value
real(8),allocatable :: a(:)
read(*,*) mode,x,n
allocate(a(n))
if(n>0) read(*,*) a
select case(mode)
case(1)
 value=alnrel(x)
case(2)
 value=rexp(x)
case(3)
 value=rlog(x)
case(4)
 value=rlog1(x)
case(5)
 value=evaluate_polynomial(a,x)
end select
write(*,'(ES26.17E3)') value
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-elementary").resolve()
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

    def record(mode, x, a=()):
        payload = f"{mode} {x} {len(a)}\n" + " ".join(map(str, a)) + "\n"
        row = dict(mode=mode, x=x, a=list(a))
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

    for mode in (1, 2, 4):
        for x in [
            -0.99,
            -0.5,
            -0.125,
            -1e-8,
            -1e-100,
            0.0,
            1e-100,
            1e-8,
            0.125,
            0.5,
            1.0,
            10.0,
            100.0,
        ]:
            record(mode, x)
    for x in [
        5e-324,
        1e-308,
        0.1,
        0.61,
        0.82,
        1 - 2**-52,
        1.0,
        1 + 2**-52,
        1.18,
        1.57,
        10.0,
        1e100,
        1e308,
    ]:
        record(3, x)
    for x in [-5e-162, -3e-162, 3e-162, 5e-162, 1e-160]:
        record(4, x)
    for mode in (1, 3, 4):
        for x in [-2.0, -1.0, 0.0]:
            record(mode, x)
    for x in [-1000.0, 710.0, 1000.0]:
        record(2, x)
    for a in [[2.0], [1.0, 2.0, 3.0], [-1.0, 0.0, 1.0], [1.0, -4.0, 6.0, -4.0, 1.0]]:
        for x in [-2.0, 0.0, 0.5, 1.0, 2.0]:
            record(5, x, a)
    record(5, 1.0, [])
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
    Path("tests/fixtures/cdflib_elementary.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
