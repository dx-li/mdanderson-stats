"""Build archived originals and capture independent numerical test fixtures.

Requires gfortran and the archives extracted beneath research/raw/<NAME>/.
No upstream numerical source is copied into the package. The CUMNOR harness
removes only its interactive main program; INVMF receives a Fortran callback
bridge in place of the unavailable S interpreter bridge.
"""

import hashlib
import json
import re
import subprocess
from pathlib import Path

RAW = Path("research/raw")
BUILD = RAW / "reference"


def compile_program(name: str, sources: list[Path]) -> Path:
    directory = BUILD / name
    directory.mkdir(parents=True, exist_ok=True)
    executable = directory / name
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", *map(str, sources), "-o", str(executable)],
        capture_output=True,
        text=True,
        check=True,
    )
    return executable.resolve()


def run(executable: Path, data: str) -> str:
    process = subprocess.run(
        [str(executable)],
        input=data,
        cwd=executable.parent,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return process.stdout


def normal_cases() -> list[dict]:
    directory = BUILD / "normal"
    directory.mkdir(parents=True, exist_ok=True)
    source = (RAW / "CUMNOR/source/cumnor-1.0/CUMNOR.f").read_text()
    routines = directory / "routines.f"
    routines.write_text(source[source.index("      SUBROUTINE crlist") :])
    driver = directory / "driver.f90"
    driver.write_text("""program normal_reference
  implicit none
  double precision :: x, p, q, lp, lq
  double precision, external :: dlanor
  do
    read(*,*,end=100) x
    if (abs(x) < 30d0) then
      call cumnor(x,p,q)
      lp=log(p)
      lq=log(q)
    else if (x < 0d0) then
      lp=dlanor(x)
      lq=0d0
    else
      lp=0d0
      lq=dlanor(x)
    end if
    write(*,'(3es26.17)') x,lp,lq
  end do
100 continue
end program
""")
    executable = compile_program("normal", [routines, driver])
    x = [
        -67861400,
        -1e7,
        -1e4,
        -100,
        -38,
        -30,
        -29.9,
        -8,
        -3,
        -1,
        0,
        1,
        3,
        8,
        29.9,
        30,
        38,
        100,
        1e4,
        1e7,
        67861400,
    ]
    output = run(executable, "\n".join(map(str, x)) + "\n")
    cases = []
    for line in output.splitlines():
        argument, log_cdf, log_sf = map(float, line.split())
        cases.append({"x": argument, "log_cdf": log_cdf, "log_sf": log_sf})
    if len(cases) != len(x):
        raise RuntimeError("CUMNOR did not produce the requested reference cases")
    return cases


def gof_cases() -> list[dict]:
    executable = compile_program("gofchi", [RAW / "GOFCHI/source/gofchi/gofchi.f"])
    cases = []
    inputs = [
        ([10, 20, 30, 40], [1, 1, 1, 1], 3),
        ([6, 9, 5], [2, 3, 5], 2),
        ([8, 12], [1, 1], 1),
        ([4.5, 10.5, 15], [1, 2, 3], 2),
        ([25, 25, 25, 25], [1, 1, 1, 1], 3),
        ([4, 8, 12, 6], [1, 1, 1, 1], 2),
        ([0, 12, 8], [0, 3, 2], 1),
        ([5, 10, 15], [0, 1, 1], 2),
    ]
    for observed, weights, df in inputs:
        output = run(
            executable,
            f"{len(observed)} {df}\n"
            + " ".join(map(str, observed))
            + "\n"
            + " ".join(map(str, weights))
            + "\nn\nn\n",
        )
        match = re.search(r"Chi-square=\s*([\d.]+).*P-value=\s*([\d.]+)", output)
        if match is None:
            raise RuntimeError(f"GOFCHI interval missing from output: {output}")
        cases.append(
            {
                "observed": observed,
                "weights": weights,
                "degrees_of_freedom": df,
                "statistic": float(match[1]),
                "pvalue": float(match[2]),
            }
        )
    return cases


def inverse_cases() -> list[dict]:
    directory = BUILD / "invmf"
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text("""program inverse_reference
  implicit none
  integer :: kind, status
  common /choice/ kind
  logical :: qleft,qhi
  double precision :: x,y,lo,hi
  double precision, external :: func
  do
    read(*,*,end=100) kind,x,y,lo,hi
    call dinvr(func,x,y,status,qleft,qhi,lo,hi,1d-5,1d-3,2d0,1d-8,1d-6)
    write(*,'(i3,es26.17)') status,x
  end do
100 continue
end program

double precision function func(x)
  implicit none
  double precision, intent(in) :: x
  integer :: kind
  common /choice/ kind
  select case(kind)
    case(1)
      func=sqrt(x)
    case(2)
      func=exp(x)
    case(3)
      func=7d0-3d0*x
    case(4)
      func=x**3
  end select
end function

subroutine devlsf(f,x,answer)
  implicit none
  double precision, external :: f
  double precision, intent(in) :: x
  double precision, intent(out) :: answer
  answer=f(x)
end subroutine
""")
    executable = compile_program(
        "invmf",
        [
            RAW / "INVMF/S/invmf/dinvr.f",
            RAW / "INVMF/S/invmf/dzror.f",
            driver,
        ],
    )
    data = [
        (1, 4, 4, 0, 100),
        (1, 90, 3, 0, 100),
        (2, 0, 10, -100, 100),
        (2, 8, 0.01, -100, 100),
        (3, 0, 13, -100, 100),
        (3, 90, 300, -100, 100),
        (4, 0, -8, -100, 100),
        (4, -20, 27, -100, 100),
    ]
    cases = []
    for kind, initial, target, lower, upper in data:
        output = run(executable, f"{kind} {initial} {target} {lower} {upper}\n")
        status, root = output.split()
        if int(status) != 0:
            raise RuntimeError(f"INVMF failed with status {status}")
        cases.append(
            {
                "kind": kind,
                "initial": initial,
                "target": target,
                "lower": lower,
                "upper": upper,
                "root": float(root),
            }
        )
    return cases


def main() -> None:
    sources = {}
    for name in ("CUMNOR", "GOFCHI", "INVMF"):
        archive = RAW / name / f"{name}_V1.tar.gz"
        sources[name] = {
            "url": "https://biostatistics.mdanderson.org/SoftwareDownload/"
            f"SoftwareFiles/{name}/{name}_V1.tar.gz",
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    result = {
        "sources": sources,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "build_flags": ["-O2", "-std=legacy"],
        "normal": normal_cases(),
        "gof": gof_cases(),
        "inverse": inverse_cases(),
    }
    Path("tests/fixtures/numerics.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Recorded CUMNOR, GOFCHI, and INVMF reference cases")


if __name__ == "__main__":
    main()
