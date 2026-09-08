"""Record original STUKEL/David Gay regression fits on the archived examples."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def build_native(*, scan=False):
    source = Path("research/raw/STUKEL/S/stukel")
    directory = Path("research/raw/reference/stukel")
    directory.mkdir(parents=True, exist_ok=True)
    library = (source / "dgay.f").read_text()
    for declaration in (
        "      INTEGER FUNCTION I1MACH(I)",
        "      DOUBLE PRECISION FUNCTION D1MACH(I)",
    ):
        start = library.index(declaration)
        end = re.search(r"^      END\s*$", library[start:], flags=re.MULTILINE)
        if end is None:
            raise ValueError("Machine constant unit not found")
        library = library[:start] + library[start + end.end() :]
    native = directory / "dgay-portable.f"
    native.write_text(library)
    machine = directory / "machine.f90"
    machine.write_text("""integer function i1mach(i)
integer i
integer,parameter :: v(16)=[5,6,7,0,32,4,2,31,2147483647,2,24,-125,128,53,-1021,1024]
i1mach=v(i)
end function
double precision function d1mach(i)
integer i
double precision v(5)
v=[tiny(1d0),huge(1d0),epsilon(1d0)/2,epsilon(1d0),log10(2d0)]
d1mach=v(i)
end function
""")
    driver = directory / "fit.f90"
    driver.write_text("""program reference
implicit none
integer n,p,k,family,lh,iv(500),i,steps
logical bad
double precision v(300),fixed(2),fx
double precision,allocatable :: x(:,:),y(:),m(:),h(:),b(:),g(:),hes(:),bounds(:,:),d(:)
read(*,*) n,p,family
k=p
if(family>0) k=k+1
if(family==5) k=k+1
lh=k*(k+1)/2
allocate(x(n,p),y(n),m(n),h(n),b(k),g(k),hes(lh),bounds(2,k),d(k))
read(*,*) x
read(*,*) y
read(*,*) m
b=0d0
g=0d0
hes=0d0
fixed=0d0
bounds(1,:)=-1d20
bounds(2,:)=1d20
if(k>p) then
 bounds(1,p+1:)=-10d0
 bounds(2,p+1:)=10d0
endif
d=1d0/bounds(2,:)
call divset(2,iv,500,300,v)
v(32)=1d-6
v(33)=1d-6
iv(17)=500
do steps=1,10000
call fgh(iv(1),family,n,p,k,x,y,m,b,fixed,lh,bad,h,fx,g,hes)
if(bad) iv(2)=1
call drmnhb(bounds,d,fx,g,hes,iv,lh,500,300,k,v,b)
if(iv(1)>2) exit
enddo
call fgh(2,family,n,p,k,x,y,m,b,fixed,lh,bad,h,fx,g,hes)
write(*,*) iv(1)
write(*,'(es27.17e3)') fx
write(*,'(*(es27.17e3,1x))') b
end program
""")
    if scan:
        text = driver.read_text()
        text = text.replace("fixed=0d0", "read(*,*) fixed")
        text = text.replace("1d20", "1000d0")
        text = text.replace("v(32)=1d-6\nv(33)=1d-6\n", "")
        driver = directory / "scan.f90"
        driver.write_text(text)
    executable = directory / ("scan" if scan else "fit")
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-std=legacy",
            "-fallow-argument-mismatch",
            str(source / "all.f"),
            str(native),
            str(machine),
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    return executable


def main():
    source = Path("research/raw/STUKEL/S/stukel")
    executable = build_native()
    cases = []
    for dataset in ("beetles", "warsaw"):
        values = np.loadtxt(source / f"{dataset}.dat")
        x = np.column_stack((np.ones(len(values)), values[:, 0]))
        for family in range(6):
            data = f"{len(x)} 2 {family}\n"
            for array in (x.ravel(order="F"), values[:, 1], values[:, 2]):
                data += " ".join(map(str, array)) + "\n"
            result = subprocess.run(
                [str(executable)], input=data, text=True, capture_output=True, check=True
            )
            rows = result.stdout.splitlines()
            cases.append(
                {
                    "dataset": dataset,
                    "x": values[:, :1].tolist(),
                    "successes": values[:, 1].tolist(),
                    "trials": values[:, 2].tolist(),
                    "family": family,
                    "status": int(rows[0]),
                    "objective": float(rows[1]),
                    "coefficients": [float(v) for v in rows[2].split()],
                }
            )
    record = {
        "source_sha256": hashlib.sha256((source / "all.f").read_bytes()).hexdigest(),
        "optimizer_sha256": hashlib.sha256((source / "dgay.f").read_bytes()).hexdigest(),
        "notes": "Original FGH/DRMNHB and minimize.S zero starts/bounds/tolerances; "
        "machine constants replaced for IEEE binary64 and 32-bit default integers. "
        "Statuses outside 3..6 are failed native fits; family 4 retains its source derivative bug.",
        "cases": cases,
    }
    Path("tests/fixtures/stukel_fit.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print([(c["dataset"], c["family"], c["status"], c["objective"]) for c in cases])


if __name__ == "__main__":
    main()
