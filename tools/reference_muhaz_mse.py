"""Native MUHAZ MSEMSE diagnostics from unchanged Fortran routines."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/MUHAZ/source/S/muhaz.f").resolve()
    directory = Path("research/raw/reference/muhaz-mse").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,m,nb,ks,flag,i,j
integer,allocatable::delta(:)
real(8),allocatable::x(:),z(:),bw(:)
real(8) pilot,left,right,hazden,h,mse,bias,var
external hazden
read(*,*)n,m,nb,ks,flag,pilot,left,right
allocate(x(n),delta(n),z(m),bw(nb))
do i=1,n
 read(*,*)x(i),delta(i)
enddo
read(*,*)z
read(*,*)bw
do j=1,nb
 do i=1,m
  h=hazden(n,ks,x,delta,z(i),pilot,left,right,flag)
  call msemse(n,ks,z(i),left,right,x,delta,bw(j),mse,bias,var,pilot,h,flag)
  write(*,'(*(ES27.17E3,1X))')bias,var,mse
 enddo
enddo
end program
""")
    executable = directory / "reference"
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-ffp-contract=off",
            "-std=legacy",
            str(source),
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        cwd=directory,
    )
    cases = []
    for times, status in [
        ([0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3], [1, 0, 1, 1, 0, 1, 0]),
        ([0, 1, 1, 2, 2, 3], [1, 0, 1, 1, 0, 1]),
    ]:
        for kernel in range(4):
            for boundary in range(3):
                z = np.linspace(0, 3, 9)
                bw = [0.35, 0.8, 2.0]
                text = f"{len(times)} {len(z)} {len(bw)} {kernel} {boundary} .65 0 3\n"
                text += "\n".join(f"{t} {d}" for t, d in zip(times, status, strict=True)) + "\n"
                text += " ".join(map(str, z)) + "\n" + " ".join(map(str, bw)) + "\n"
                out = subprocess.run(
                    [str(executable)],
                    input=text,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                rows = np.array(
                    [list(map(float, line.split())) for line in out.stdout.splitlines()]
                )
                cases.append(
                    dict(
                        times=times,
                        delta=status,
                        grid=z.tolist(),
                        bandwidths=bw,
                        pilot_bandwidth=0.65,
                        kernel=kernel,
                        boundary=boundary,
                        diagnostics=rows.reshape(len(bw), len(z), 3).tolist(),
                    )
                )
    fixture = dict(
        source="Unchanged MUHAZ Fortran MSEMSE and dependencies, independent driver",
        compiler_flags=["-O2", "-ffp-contract=off", "-std=legacy"],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/muhaz-mse.json").write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} native MUHAZ MSE grids")


if __name__ == "__main__":
    main()
