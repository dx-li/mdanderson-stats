"""Native KNNHAD neighbor selector and fitted hazard reference."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/MUHAZ/source/S/muhaz.f").resolve()
    directory = Path("research/raw/reference/muhaz-knn").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,m,ng,nb,ks,flag,i,method,kmin,kmax
integer,allocatable::d(:)
real(8),allocatable::x(:),z(:),zz(:),h(:),bopt(:),bs(:),scores(:)
real(8) pilot,left,right,smooth
read(*,*)n,m,ng,nb,ks,flag,pilot,left,right,smooth,method,kmin,kmax
allocate(x(n),d(n),z(ng),zz(m),h(m),bopt(ng),bs(m),scores(nb))
do i=1,n
 read(*,*)x(i),d(i)
enddo
read(*,*)z
read(*,*)zz
call knnhad(n,x,d,ks,method,ng,z,m,zz,pilot,left,right,smooth,flag, &
            h,kmin,kmax,bopt,bs,scores)
write(*,*)kmin
write(*,'(*(ES27.17E3,1X))')bopt
write(*,'(*(ES27.17E3,1X))')bs
if(nb>1)then
 write(*,'(*(ES27.17E3,1X))')scores
endif
write(*,'(*(ES27.17E3,1X))')h
end program
""")
    exe = directory / "reference"
    flags = ["-O2", "-ffp-contract=off", "-std=legacy"]
    subprocess.run(
        ["gfortran", *flags, str(source), str(driver), "-o", str(exe)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    inputs = [
        ([0.2, 0.4, 0.8, 1.1, 1.7, 2.1, 2.4, 2.8, 3], [1] * 9),
        ([0.2, 0.4, 0.8, 1.1, 1.1, 1.7, 2.1, 2.4, 2.8, 3], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]),
    ]
    cases = [
        dict(
            times=t,
            delta=d,
            kernel=ks,
            boundary=flag,
            method=method,
            neighbors=counts,
            smoothing=1.3,
        )
        for t, d in inputs
        for ks in range(4)
        for flag in range(3)
        for method in [1, 2]
        for counts in [[2, 3, 4], [3]]
    ]
    for c in cases:
        z = np.linspace(0, 3, 9)
        zz = np.linspace(0, 3, 25)
        bw = c["neighbors"]
        data = (
            f"{len(c['times'])} 25 9 {len(bw)} {c['kernel']} {c['boundary']} "
            f".65 0 3 {c['smoothing']} {c['method']} {bw[0]} {bw[-1]}\n"
        )
        data += "\n".join(f"{t} {d}" for t, d in zip(c["times"], c["delta"], strict=True)) + "\n"
        data += " ".join(map(str, z)) + "\n" + " ".join(map(str, zz)) + "\n"
        out = subprocess.run(
            [str(exe)], input=data, text=True, capture_output=True, check=True, timeout=10
        )
        rows = [list(map(float, row.split())) for row in out.stdout.splitlines()]
        c.update(
            selected_neighbors=int(rows[0][0]),
            local_bandwidth=rows[1],
            bandwidth=rows[2],
            scores=rows[3] if len(bw) > 1 else None,
            hazard=rows[-1],
        )
    data = dict(
        source="Unchanged MUHAZ KNNHAD and dependencies",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler_flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/muhaz-knn.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} native nearest-neighbor fits")


if __name__ == "__main__":
    main()
