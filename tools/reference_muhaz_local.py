"""Native NEW_HAD local selector and fitted hazard reference."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/MUHAZ/source/S/muhaz.f").resolve()
    directory = Path("research/raw/reference/muhaz-local").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,m,ng,nb,ks,flag,i
integer,allocatable::d(:)
real(8),allocatable::x(:),z(:),zz(:),bw(:),h(:),bopt(:),bs(:),mse(:),bias(:),var(:),scores(:)
real(8) pilot,left,right,imse,chosen,smooth
read(*,*)n,m,ng,nb,ks,flag,pilot,left,right,smooth
allocate(x(n),d(n),z(ng),zz(m),bw(nb),h(m),bopt(ng),bs(m),mse(ng),bias(ng),var(ng),scores(nb))
do i=1,n
 read(*,*)x(i),d(i)
enddo
read(*,*)z
read(*,*)zz
read(*,*)bw
call newhad(n,x,d,ks,1,z,ng,zz,m,pilot,bw,nb,left,right,smooth,flag, &
            h,bopt,bs,mse,bias,var,imse,chosen,scores)
! Local output arrays are undefined in the single-bandwidth bypass.
if(nb>1)then
 write(*,'(ES27.17E3)')imse
 write(*,'(*(ES27.17E3,1X))')bopt
 write(*,'(*(ES27.17E3,1X))')bs
 write(*,'(*(ES27.17E3,1X))')mse
 ! Do not print uninitialized bias/variance when no positive MSE qualified.
 do i=1,ng
  if(mse(i)==1d30)then
   bias(i)=0
   var(i)=0
  endif
 enddo
 write(*,'(*(ES27.17E3,1X))')bias
 write(*,'(*(ES27.17E3,1X))')var
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
        ([0.2, 0.7, 1.2, 1.8, 2.1, 2.6, 3], [1, 0, 1, 1, 0, 1, 0]),
        ([0, 1, 1, 2, 2, 3], [1, 0, 1, 1, 0, 1]),
        ([0.2, 0.8, 1.5, 2, 3], [0] * 5),
    ]
    cases = []
    for times, d in inputs:
        for ks in range(4):
            for flag in range(3):
                cases.append(
                    dict(times=times, delta=d, kernel=ks, boundary=flag, bandwidths=[0.35, 0.8, 2])
                )
        cases.append(dict(times=times, delta=d, kernel=1, boundary=2, bandwidths=[0.8]))
    cases = [dict(c, smoothing=smooth) for c in cases for smooth in (0.65, 2.0, 4.0)]
    for c in cases:
        z = np.linspace(0, 3, 9)
        zz = np.linspace(0, 3, 25)
        bw = c["bandwidths"]
        data = (
            f"{len(c['times'])} 25 9 {len(bw)} {c['kernel']} {c['boundary']} "
            f".65 0 3 {c['smoothing']}\n"
        )
        data += "\n".join(f"{t} {d}" for t, d in zip(c["times"], c["delta"], strict=True)) + "\n"
        data += (
            " ".join(map(str, z))
            + "\n"
            + " ".join(map(str, zz))
            + "\n"
            + " ".join(map(str, bw))
            + "\n"
        )
        out = subprocess.run(
            [str(exe)], input=data, text=True, capture_output=True, check=True, timeout=10
        )
        rows = [list(map(float, row.split())) for row in out.stdout.splitlines()]
        c.update(
            score=rows[0][0] if len(bw) > 1 else None,
            local_bandwidth=rows[1] if len(bw) > 1 else None,
            bandwidth=rows[2] if len(bw) > 1 else [bw[0]] * 25,
            minimum_mse=rows[3] if len(bw) > 1 else None,
            bias=rows[4] if len(bw) > 1 else None,
            variance=rows[5] if len(bw) > 1 else None,
            hazard=rows[-1],
        )
    data = dict(
        source="Unchanged MUHAZ NEW_HAD local branch and dependencies",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler_flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/muhaz-local.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} native local fits")


if __name__ == "__main__":
    main()
