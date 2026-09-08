"""Native HAZDEN oracle from the MUHAZ archive, without redistributing its code."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/MUHAZ/source/S/muhaz.f")
    directory = Path("research/raw/reference/muhaz-fixed").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for declaration in [
        "DOUBLE PRECISION FUNCTION hazden",
        "SUBROUTINE ibnds",
        "DOUBLE PRECISION FUNCTION kernel",
    ]:
        match = re.search(
            r"^      " + declaration + r"\(.*?^      END\s*$", source.read_text(), re.M | re.S
        )
        if match is None:
            raise RuntimeError(f"Missing {declaration}")
        blocks.append(match.group())
    numerical = directory / "numerical.f"
    numerical.write_text("\n".join(blocks) + "\n")
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,m,ks,flag,i
integer,allocatable::delta(:)
real(8),allocatable::x(:),z(:)
real(8) b,left,right,hazden
external hazden
read(*,*)n,m,ks,flag,b,left,right
allocate(x(n),delta(n),z(m))
do i=1,n
 read(*,*)x(i),delta(i)
enddo
read(*,*)z
do i=1,m
 write(*,'(ES27.17E3)')hazden(n,ks,x,delta,z(i),b,left,right,flag)
enddo
end program
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(numerical), str(driver), "-o", str(executable)],
        check=True,
        capture_output=True,
        cwd=directory,
    )
    inputs = [
        ([0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3], [1, 0, 1, 1, 0, 1, 0], 0.75),
        ([0, 1, 1, 2, 2, 3], [1, 0, 1, 1, 0, 1], 1),
        ([0.1, 0.5, 1, 1.3, 2, 2.7, 3], [0, 1, 1, 0, 1, 1, 0], 4),
    ]
    cases = []
    grid = np.linspace(0, 3, 25)
    for times, status, b in inputs:
        for shape in range(4):
            for boundary in range(3):
                text = f"{len(times)} {len(grid)} {shape} {boundary} {b} 0 3\n"
                text += "\n".join(f"{t} {d}" for t, d in zip(times, status, strict=True)) + "\n"
                text += " ".join(map(str, grid)) + "\n"
                out = subprocess.run(
                    [str(executable)],
                    input=text,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                values = list(map(float, out.stdout.split()))
                cases.append(
                    dict(
                        times=times,
                        delta=status,
                        bandwidth=b,
                        kernel=shape,
                        boundary=boundary,
                        grid=grid.tolist(),
                        hazard=values,
                    )
                )
    data = dict(
        source="Unchanged MUHAZ HAZDEN, IBNDS and KERNEL; independent driver",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/muhaz-fixed.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} native fixed-bandwidth curves")


if __name__ == "__main__":
    main()
