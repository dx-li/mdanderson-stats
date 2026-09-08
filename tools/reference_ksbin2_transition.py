"""Record SSUPD's unchanged transition loop with independent scaled coefficients."""

import hashlib
import json
import subprocess
from math import comb
from pathlib import Path

from mdanderson_stats import ksbin2_ordering


def main():
    source = Path("research/raw/KSBIN2/source/ksbin290_2.1/source/ksbin2_aux_mod.f90")
    original = source.read_text()
    start = original.index("      BCOF(:NMAX,:NMAX) = zero", original.index("SUBROUTINE SSUPD("))
    end = original.index("!     Set up the sort criterion", start)
    kernel = original[start:end]
    directory = Path("research/raw/reference/ksbin2-transition").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text(
        """program reference
implicit none
integer,parameter::dpkind=kind(1d0),nmax=100
real(dpkind),parameter::zero=0d0
integer newobs(2),nlow,nhi,nptgrp(2),npts,s1(10201),s2(10201)
integer ipt,s1new,s1old,s2new,s2old,i,j
real(dpkind) bold,bcof(0:nmax,0:nmax),bincof(10201)
read(*,*) nptgrp,newobs,npts
nlow=1
nhi=npts
do i=1,npts
 read(*,*) s1(i),s2(i),bincof(i)
enddo
"""
        + kernel
        + """
do i=1,npts
 write(*,'(I4,1X,I4,1X,ES27.17E3)') s1(i),s2(i),bincof(i)
enddo
contains
function scbic(k,n) result(value)
integer k,n,t
real(dpkind) value
value=2d0**(-n)
do t=1,k
 value=value*real(n-t+1,dpkind)/real(t,dpkind)
enddo
end function
function scbic1(k,n) result(value)
integer k,n
real(dpkind) value
value=scbic(k,n)
end function
end program
"""
    )
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-fcheck=bounds", str(driver), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    cases = []
    for sizes, increment in [((2, 2), (1, 1)), ((2, 3), (2, 1)), ((4, 3), (1, 2))]:
        for alternative in ["less", "greater", "two-sided"]:
            ordering = ksbin2_ordering(*sizes, criteria=(1,), alternative=alternative)
            # Continue after rejecting group 0 and before quitting from group 2.
            points = ordering.events[ordering.group_end[0] + 1 : ordering.group_end[1] + 1]
            data = " ".join(map(str, [*sizes, *increment, len(points)])) + "\n"
            for i, j in points:
                c = comb(sizes[0], int(i)) * comb(sizes[1], int(j)) * 2.0 ** (-sum(sizes))
                data += f"{i} {j} {c:.17g}\n"
            output = subprocess.run(
                [str(executable)], input=data, text=True, capture_output=True, check=True, timeout=5
            )
            rows = [list(map(float, line.split())) for line in output.stdout.splitlines()]
            cases.append(
                dict(initial=sizes, increment=increment, alternative=alternative, rows=rows)
            )
    fixture = dict(
        source=(
            "Unchanged SSUPD coefficient transition/repacking loop; "
            "independent exact-combinatorial input and scaled binomial helpers; "
            "no native sorting invoked"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(kernel.encode()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/ksbin2_transition.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native transitions")


if __name__ == "__main__":
    main()
