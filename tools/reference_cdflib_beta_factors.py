"""Record unchanged F95 beta factors and finite shape-shift differences."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode,mu,n
real(8) a,b,x,y,eps,value
read(*,*) mode,a,b,x,y,mu,n,eps
select case(mode)
case(1)
 value=brcomp(a,b,x,y)
case(2)
 value=brcmp1(mu,a,b,x,y)
case(3)
 value=bup(a,b,x,y,n,eps)
end select
write(*,'(ES26.17E3)') value
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-beta-factors").resolve()
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

    def record(mode, a, b, x, y, mu=0, n=1, eps=5e-15):
        row = dict(mode=mode, a=a, b=b, x=x, y=y, mu=mu, n=n, eps=eps)
        try:
            result = subprocess.run(
                [str(exe)],
                input=f"{mode} {a} {b} {x} {y} {mu} {n} {eps}\n",
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
            value = float(result.stdout)
            row.update(
                execution_outcome="completed", result=value if math.isfinite(value) else str(value)
            )
        cases.append(row)

    pairs = [
        (5e-324, 1.0),
        (1e-100, 1.0),
        (0.1, 0.9),
        (0.5, 0.5),
        (0.9, 0.1),
        (1.0, 1e-100),
        (1.0, 5e-324),
        (0.0, 1.0),
        (1.0, 0.0),
    ]
    for a, b in [
        (0.1, 0.2),
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 3.0),
        (8.0, 8.0),
        (20.0, 100.0),
        (1e-100, 1e-100),
    ]:
        for x, y in pairs:
            record(1, a, b, x, y)
    for a, b, x, y in [
        (1e308, 1e308, 0.5, 0.5),
        (1e100, 1e100, 0.5, 0.5),
        (1e308, 8.0, 1.0, 8e-308),
        (8.0, 1e308, 8e-308, 1.0),
        (5e-324, 5e-324, 0.5, 0.5),
        (1e-309, 1.0, 0.5, 0.5),
        (0.0, 1.0, 0.5, 0.5),
        (-0.5, 1.0, 0.5, 0.5),
        (2.0, 3.0, 0.1, 0.8),
        (8.0, 8.0, 0.1, 0.8),
    ]:
        record(1, a, b, x, y)
    for mu in [-1000, -744, 0, 700, 1000]:
        for a, b, x, y in [
            (1.0, 1.0, 0.5, 0.5),
            (1.0, 1.0, 1e-300, 1.0),
            (8.0, 8.0, 0.5, 0.5),
            (8.0, 8.0, 1e-100, 1.0),
            (1e-100, 1e-100, 0.5, 0.5),
            (1e-309, 1e-309, 0.5, 0.5),
            (1e100, 1e100, 0.5, 0.5),
            (1e300, 1e300, 0.5, 0.5),
        ]:
            record(2, a, b, x, y, mu=mu)
    for x, y in [(0.0, 1.0), (1.0, 0.0)]:
        record(2, 0.5, 0.5, x, y)
    for n in [1, 2, 10, 100]:
        for a, b, x, y in [
            (0.5, 0.5, 0.1, 0.9),
            (1.0, 1.0, 0.5, 0.5),
            (2.0, 3.0, 0.9, 0.1),
            (8.0, 8.0, 0.5, 0.5),
            (1e-100, 1e-100, 0.5, 0.5),
            (1e308, 1e308, 0.5, 0.5),
            (1.0, 1.0, 1.0, 5e-324),
        ]:
            record(3, a, b, x, y, n=n)
    for n in [0, -1, 2**31 - 1]:
        record(3, 1.0, 1.0, 1 - 1e-12, 1e-12, n=n)
    for eps in [0.0, -1e-6, 1e-3]:
        record(3, 2.0, 3.0, 0.9, 0.1, n=100, eps=eps)
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
    Path("tests/fixtures/cdflib_beta_factors.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
