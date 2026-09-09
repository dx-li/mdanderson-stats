"""Record unchanged F95 incomplete-gamma support, including failure outcomes."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode,ind
real(8) a,x,eps,factor,r,p,q
read(*,*) mode,a,x,ind,eps,factor
p=-999
q=-999
r=rcomp(a,x)*factor
select case(mode)
case(1)
 p=r
case(2)
 call gratio(a,x,p,q,ind)
case(3)
 call grat1(a,x,r,p,q,eps)
end select
write(*,'(3(ES26.17E3,1X))') r,p,q
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-incomplete-gamma").resolve()
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

    def record(mode, a, x, ind=0, eps=5e-15, factor=1.0):
        row = dict(mode=mode, a=a, x=x, ind=ind, eps=eps, factor=factor)
        try:
            result = subprocess.run(
                [str(exe)],
                input=f"{mode} {a} {x} {ind} {eps} {factor}\n",
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
            values = [float(v) for v in result.stdout.split()]
            row.update(execution_outcome="completed")
            row.update(
                (name, value if math.isfinite(value) else str(value))
                for name, value in zip(("r", "p", "q"), values, strict=True)
            )
        cases.append(row)

    for a in [5e-324, 1e-100, 0.1, 0.5, 1.0, 2.0, 19.9, 20.0, 100.0]:
        for x in [5e-324, 0.1, 1.1, 10.0, 100.0, 744.0]:
            record(1, a, x)
    for a, x in [
        (0.0, 1.0),
        (1.0, 0.0),
        (-0.5, 1.0),
        (1e100, 1e100),
        (1e308, 1e308),
        (1e308, 1.0),
        (20.0, 1e-309),
    ]:
        record(1, a, x)
    for a in [0.1, 0.5, 1.0, 2.0, 10.0, 20.0, 100.0]:
        for x in [0.1, 1.1, 10.0, 100.0, 744.0]:
            record(2, a, x)
    for ind in [-1, 1, 2, 100]:
        for a, x in [(0.1, 1.1), (20.0, 20.0), (100.0, 90.0)]:
            record(2, a, x, ind=ind)
    record(2, 20.0, 20.0)
    for a, x in [
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, 0.0),
        (-1.0, 1.0),
        (1.0, -1.0),
        (5e-324, 5e-324),
        (1e-100, 1e-300),
        (1e100, 1e100),
        (1e308, 1e308),
        (1e308, 1.0),
    ]:
        record(2, a, x)
    for a in [5e-324, 1e-100, 0.1, 0.5, 1.0]:
        for x in [5e-324, 0.1, 1.1, 10.0, 100.0, 744.0]:
            record(3, a, x)
    for a, x in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0)]:
        record(3, a, x)
    for eps in [1e-6, 1e-3, 0.0, -1e-6]:
        record(3, 0.1, 1.1, eps=eps)
    for factor in [0.0, 0.5, 2.0]:
        for x in [0.1, 1.1]:
            record(3, 0.1, x, factor=factor)
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
    Path("tests/fixtures/cdflib_incomplete_gamma.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
