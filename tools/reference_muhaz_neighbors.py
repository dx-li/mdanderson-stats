"""Native KNNCEN/OLAFBW nearest-neighbor bandwidth reference."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/MUHAZ/source/S/muhaz.f").resolve()
    directory = Path("research/raw/reference/muhaz-neighbors").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,m,k,method,i
integer,allocatable::d(:)
real(8),allocatable::x(:),z(:),bw(:)
read(*,*)n,m,k,method
allocate(x(n),d(n),z(m),bw(m))
do i=1,n
 read(*,*)x(i),d(i)
enddo
read(*,*)z
if(method==1)then
 call knncen(x,d,n,z,m,k,bw)
else
 call olafbw(x,d,n,z,m,k,bw)
endif
write(*,'(*(ES27.17E3,1X))')bw
end program
""")
    flags = ["-O2", "-ffp-contract=off", "-std=legacy"]
    exe = directory / "reference"
    subprocess.run(
        ["gfortran", *flags, str(source), str(driver), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    cases = []
    inputs = [
        ([0.2, 0.4, 0.8, 1.1, 1.7, 2.1, 2.4, 2.8, 3], [1] * 9),
        ([0.2, 0.4, 0.8, 1.1, 1.1, 1.7, 2.1, 2.4, 2.8, 3], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]),
        ([0, 0.5, 1, 1, 2, 2, 3, 3], [1, 0, 1, 1, 0, 1, 1, 0]),
    ]
    grid = np.linspace(0, 4, 17)
    for times, delta in inputs:
        for method in [1, 2]:
            for k in [1, 2, 3, 5]:
                data = f"{len(times)} {grid.size} {k} {method}\n"
                data += "\n".join(f"{t} {d}" for t, d in zip(times, delta, strict=True)) + "\n"
                data += " ".join(map(str, grid)) + "\n"
                out = subprocess.run(
                    [str(exe)], input=data, capture_output=True, text=True, check=True, timeout=10
                )
                cases.append(
                    dict(
                        times=times,
                        delta=delta,
                        method=method,
                        neighbors=k,
                        grid=grid.tolist(),
                        bandwidth=list(map(float, out.stdout.split())),
                    )
                )
    fixture = dict(
        source="Unchanged MUHAZ KNNCEN/OLAFBW and dependencies",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler_flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/muhaz-neighbors.json").write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} native nearest-neighbor grids")


if __name__ == "__main__":
    main()
