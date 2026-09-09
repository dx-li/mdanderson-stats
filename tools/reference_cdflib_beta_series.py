"""Record unchanged F95 incomplete-beta series helpers."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode
real(8) a,b,x,eps,value
read(*,*) mode,a,b,x,eps
select case(mode)
case(1)
 value=apser(a,b,x,eps)
case(2)
 value=fpser(a,b,x,eps)
case(3)
 value=bpser(a,b,x,eps)
end select
write(*,'(ES26.17E3)') value
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-beta-series").resolve()
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

    def record(mode, a, b, x, eps=5e-15):
        row = dict(mode=mode, a=a, b=b, x=x, eps=eps)
        try:
            result = subprocess.run(
                [str(exe)],
                input=f"{mode} {a} {b} {x} {eps}\n",
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

    for a, b in [
        (1e-20, 0.5),
        (1e-20, 1.0),
        (1e-20, 2.0),
        (1e-20, 8.0),
        (1e-100, 1e-50),
        (5e-324, 1.0),
        (5e-324, 1e-309),
        (1e-20, 1e20),
        (1e-309, 1e308),
    ]:
        for x in [0.0, 5e-324, min(0.1, 0.5 / b), min(0.5, 1 / b)]:
            record(1, a, b, x)
    for a, b in [
        (1e-20, 1e-40),
        (0.1, 1e-20),
        (1.0, 1e-20),
        (2.0, 1e-20),
        (100.0, 1e-20),
        (1e-309, 5e-324),
        (1e308, 1e-20),
    ]:
        for x in [0.0, 5e-324, 0.1, 0.5]:
            record(2, a, b, x)
    for a, b in [
        (0.1, 0.2),
        (0.5, 0.5),
        (1.0, 1.0),
        (2.0, 3.0),
        (8.0, 8.0),
        (20.0, 100.0),
        (1e-100, 1e-100),
        (5e-324, 5e-324),
        (1.0, 1e308),
    ]:
        for x in [0.0, 5e-324, min(0.1, 0.3 / b), min(0.5, 0.7 / b)]:
            record(3, a, b, x)
    for x in [0.5, 1 - 1e-12, 1.0]:
        record(3, 1.0, 0.5, x)
        record(3, 1.0, 1.0, x)
    for mode, a, b in [(1, 1e-20, 1.0), (2, 1.0, 1e-20), (3, 2.0, 1.0)]:
        for eps in [0.0, -1e-6, 1e-3]:
            record(mode, a, b, 0.5, eps)
    record(2, 1.0, 1e-15, 1e-308)
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
    Path("tests/fixtures/cdflib_beta_series.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
