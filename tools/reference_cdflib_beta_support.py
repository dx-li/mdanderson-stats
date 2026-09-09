"""Record unchanged F95 beta and gamma-ratio support routines."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_mathlib_mod
implicit none
integer mode
real(8) a,b,value
read(*,*) mode,a,b
select case(mode)
case(1)
 value=algdiv(a,b)
case(2)
 value=bcorr(a,b)
case(3)
 value=betaln(a,b)
case(4)
 value=log_beta(a,b)
case(5)
 value=gsumln(a,b)
case(6)
 value=log_bicoef(a,b)
end select
write(*,'(ES26.17E3)') value
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_mathlib_mod")]
    work = Path("research/raw/reference/cdflib-beta-support").resolve()
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

    def record(mode, a, b):
        payload = f"{mode} {a} {b}\n"
        row = dict(mode=mode, a=a, b=b)
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

    for a in [0.0, 5e-324, 1e-309, 1e-100, 1e-10, 0.5, 1.0, 8.0, 10.0, 1e100, 1e308]:
        for b in [8.0, 10.0, 1e100, 1e308]:
            record(1, a, b)
    for a, b in [(-1.0, 10.0), (-8.0, 10.0), (-9.5, 10.0), (-10.0, 10.0)]:
        record(1, a, b)
    for a in [8.0, 10.0, 100.0, 1e100, 1e308]:
        for b in [8.0, 10.0, 100.0, 1e100, 1e308]:
            record(2, a, b)
    pairs = [
        (5e-324, 5e-324),
        (1e-309, 1e-309),
        (1e-100, 0.5),
        (0.5, 0.5),
        (0.1, 0.9),
        (1.0, 1.0),
        (1.0, 1 + 2**-52),
        (1.0, 2.0),
        (2.0, 2.0),
        (1.5, 2.5),
        (2.25, 7.9),
        (7.9, 8.0),
        (8.0, 8.0),
        (8.0, 1000.0),
        (8.0, 1001.0),
        (0.5, 1e308),
        (1.0, 1e308),
        (8.0, 1e308),
        (1e100, 1e100),
        (1e308, 1e308),
        (0.0, 1.0),
        (-1.0, 1.0),
    ]
    for mode in [3, 4]:
        for a, b in pairs:
            record(mode, a, b)
            if a != b:
                record(mode, b, a)
    for a in [1.0, 1 + 2**-52, 1.125, 1.25, 1.5, 1.75, 2.0]:
        for b in [1.0, 1 + 2**-52, 1.25, 2.0]:
            record(5, a, b)
    for k, n in [
        (0.0, 0.0),
        (0.0, 10.0),
        (1.0, 10.0),
        (5.0, 10.0),
        (10.0, 10.0),
        (0.5, 1.0),
        (1.5, 2.5),
        (-0.5, 0.5),
        (1.5, 1.0),
        (-0.25, -0.5),
        (1e-100, 1.0),
        (5e-324, 1.0),
        (1.0, 1e100),
        (0.0, 1e308),
        (1.0, 1e308),
        (5e307, 1e308),
        (-1.0, 10.0),
        (11.0, 10.0),
        (0.0, -1.0),
    ]:
        record(6, k, n)
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
    Path("tests/fixtures/cdflib_beta_support.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print("Recorded", len(cases), "native cases")


if __name__ == "__main__":
    main()
