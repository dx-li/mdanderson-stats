"""Record unchanged F95 gamma and digamma support routines."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode
real(8) x,value
read(*,*) mode,x
select case(mode)
case(1)
 value=alngam(x)
case(2)
 value=gamln(x)
case(3)
 value=log_gamma(x)
case(4)
 value=gamln1(x)
case(5)
 value=gam1(x)
case(6)
 value=gamma(x)
case(7)
 value=psi(x)
end select
write(*,'(ES26.17E3)') value
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-gamma-support").resolve()
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

    def record(mode, x):
        payload = f"{mode} {x}\n"
        row = dict(mode=mode, x=x)
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

    positive = [
        5e-324,
        1e-309,
        1e-308,
        1e-100,
        1e-10,
        0.5,
        0.8,
        1 - 2**-53,
        1.0,
        1 + 2**-52,
        1.25,
        1.5,
        2.0,
        2.25,
        3.0,
        6.0,
        10.0,
        12.0,
        15.0,
        50.0,
        170.0,
        171.5,
        172.0,
        1000.0,
        1e10,
        1e100,
        1e305,
        1e306,
        1e308,
    ]
    for mode in [1, 2, 3, 6, 7]:
        for x in positive:
            record(mode, x)
    for mode in [4, 5]:
        for x in [
            -0.2,
            -1e-10,
            -1e-100,
            -5e-324,
            0.0,
            5e-324,
            1e-100,
            1e-10,
            0.5,
            0.6,
            1 - 2**-53,
            1.0,
            1 + 2**-52,
            1.25,
        ]:
            record(mode, x)
    for x in [-0.5, 1.5]:
        record(5, x)
    for mode in [1, 2, 3, 6, 7]:
        for x in [0.0, -1.0, -2.0, -0.5, -1.5, -15.5, -171.5, -172.5, -175.5, -177.5, -180.5]:
            record(mode, x)
    for x in [-2147483647.5, -1e-309, -5e-324]:
        for mode in [6, 7]:
            record(mode, x)
    record(1, -1e100)
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
    Path("tests/fixtures/cdflib_gamma_support.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
