"""Record the four remaining F95 incomplete-beta helpers."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode,ierr
real(8) a,b,x,y,lambda,eps,value,complement
read(*,*) mode,a,b,x,y,lambda,eps,value
ierr=0
complement=0
select case(mode)
case(1)
 value=basym(a,b,lambda,eps)
case(2)
 value=bfrac(a,b,x,y,lambda,eps)
case(3)
 call bgrat(a,b,x,y,value,eps,ierr)
case(4)
 call bratio(a,b,x,y,value,complement,ierr)
end select
write(*,'(2ES26.17E3,I5)') value,complement,ierr
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-beta-remaining").resolve()
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

    def record(mode, a, b, x=0.5, y=None, lam=None, eps=5e-15, initial=0.0):
        y = 1 - x if y is None else y
        lam = a * y - b * x if lam is None else lam
        row = dict(mode=mode, a=a, b=b, x=x, y=y, lam=lam, eps=eps, initial=initial)
        try:
            result = subprocess.run(
                [str(exe)],
                input=f"{mode} {a} {b} {x} {y} {lam} {eps} {initial}\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=3,
            )
        except subprocess.CalledProcessError as error:
            row.update(
                execution_outcome="process_error", stderr=error.stderr, exit_code=error.returncode
            )
        except subprocess.TimeoutExpired:
            row.update(execution_outcome="timeout", timeout_seconds=3)
        else:
            value, complement, status = result.stdout.split()
            values = [float(value), float(complement)]
            row.update(
                execution_outcome="completed",
                result=[v if math.isfinite(v) else str(v) for v in values],
                status=int(status),
            )
        cases.append(row)

    for a, b in [(15.0, 15.0), (15.0, 50.0), (50.0, 15.0), (100.0, 100.0)]:
        for lam in [0.0, a * 0.1, a * 0.5, a]:
            record(1, a, b, lam=lam)
    for lam in [0.0, 1e140, 1e146, 1e154]:
        record(1, 1e308, 1e308, lam=lam)
    for a, b in [(2.0, 3.0), (15.0, 15.0), (20.0, 50.0), (100.0, 100.0)]:
        for x in [0.0, 0.1, 0.5, 0.9, 1.0]:
            record(2, a, b, x)
    record(2, 1e308, 1e308, lam=0.0)
    for a, b in [(15.0, 0.5), (100.0, 0.5), (15.0, 1.0), (15.0, 1e-20), (1e308, 0.5)]:
        for x in [0.0, 0.1, 0.5, 1 - 1e-12, 1.0]:
            record(3, a, b, x)
    for initial in [0.25, -1.0, 1.0]:
        record(3, 15.0, 0.5, 0.5, initial=initial)
    for a, b in [
        (0.0, 1.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
        (2.0, 3.0),
        (15.0, 50.0),
        (1e-309, 5e-324),
        (1e308, 1e308),
    ]:
        for x in [0.0, 0.1, 0.5, 0.9, 1.0]:
            record(4, a, b, x)
    for a, b, x, y in [
        (-1.0, 1.0, 0.5, 0.5),
        (0.0, 0.0, 0.5, 0.5),
        (1.0, 1.0, -0.1, 1.1),
        (1.0, 1.0, 0.5, -0.1),
        (1.0, 1.0, 0.5, 0.4),
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 1.0, 0.0),
    ]:
        record(4, a, b, x, y, lam=0.0)
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
    Path("tests/fixtures/cdflib_beta_remaining.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
