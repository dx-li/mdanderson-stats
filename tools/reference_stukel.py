"""Compile archived STUKEL FGH and record its transformed linear predictor."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/STUKEL/S/stukel/all.f")
    directory = Path("research/raw/reference/stukel")
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,i
logical bad
double precision a(2),b(1),f,g(1),hess(1)
double precision,allocatable :: x(:,:),y(:),m(:),h(:)
read(*,*) n,a
allocate(x(n,1),y(n),m(n),h(n))
read(*,*) x(:,1)
b=1d0
y=0d0
m=1d0
call fgh(1,0,n,1,1,x,y,m,b,a,1,bad,h,f,g,hess)
if (bad) stop 1
do i=1,n
write(*,'(es27.17e3)') h(i)
end do
end program
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(source), str(driver), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    eta = np.r_[-5, -1, -0.1, -0.009, -1e-8, 0, 1e-8, 0.009, 0.1, 1, 5]
    cases = []
    for a1 in (-2, -0.01, 0, 0.01, 2):
        for a2 in (-1, 0, 1):
            data = f"{len(eta)} {a1} {a2}\n" + " ".join(map(str, eta)) + "\n"
            result = subprocess.run(
                [str(executable)], input=data, text=True, capture_output=True, check=True
            )
            cases.append(
                {
                    "eta": eta.tolist(),
                    "alpha1": a1,
                    "alpha2": a2,
                    "log_odds": [float(v) for v in result.stdout.split()],
                }
            )
    output = {
        "archive_sha256": hashlib.sha256(
            Path("research/raw/STUKEL/STUKEL_V1.tar.gz").read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "notes": "Original FGH, icode=1 and icase=0 with fixed alpha; "
        "one covariate eta and coefficient one.",
        "cases": cases,
    }
    Path("tests/fixtures/stukel.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} original STUKEL link cases")


if __name__ == "__main__":
    main()
