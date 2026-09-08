"""Compile unchanged KSBIN2 score routines and record complete small count grids."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/KSBIN2/source/ksbin290_2.1/source/ksbin2_aux_mod.f90")
    original = source.read_text()
    directory = Path("research/raw/reference/ksbin2").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name, kind in [("SSSSRT", "SUBROUTINE"), ("XLR", "FUNCTION"), ("QEQDBL", "FUNCTION")]:
        prefix = r"(?:LOGICAL )?" if name == "QEQDBL" else ""
        match = re.search(
            r"^\s*" + prefix + kind + " " + name + r"\b.*?END " + kind + " " + name,
            original,
            re.M | re.S | re.I,
        )
        if match is None:
            raise RuntimeError(name)
        blocks.append(match.group())
    numerical = directory / "numerical.f90"
    numerical.write_text(
        """module biomath_constants_mod
implicit none
integer,parameter::dpkind=kind(1d0)
real(dpkind),parameter::zero=0d0,one=1d0,thousandth=.001d0
end module
module numerical
use biomath_constants_mod
implicit none
integer npts,nptgrp(2),side12,ncrit,lcrit(4),s1(10201),s2(10201)
real(dpkind) pa(2),scrit(10201)
contains
"""
        + "\n".join(blocks)
        + "\nend module\n"
    )
    driver = directory / "driver.f90"
    driver.write_text("""program reference
use numerical
implicit none
integer i,j,k
read(*,*) nptgrp,side12,pa,ncrit
read(*,*) lcrit(:ncrit)
k=0
do i=0,nptgrp(1)
 do j=0,nptgrp(2)
  k=k+1
  s1(k)=i
  s2(k)=j
 enddo
enddo
npts=k
call ssssrt
write(*,'(*(ES27.17E3,1X))') scrit(:npts)
end program
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(numerical), str(driver), "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for sizes in [(3, 4), (10, 10), (20, 15)]:
        for alternative, side, pair in [
            ("less", 1, [0.2, 0.6]),
            ("greater", 1, [0.6, 0.2]),
            ("two-sided", 2, [0.2, 0.6]),
        ]:
            for criteria in [[1], [2], [3], [4], [1, 2], [4, 3, 2, 1]]:
                data = " ".join(map(str, [*sizes, side, *pair, len(criteria)])) + "\n"
                data += " ".join(map(str, criteria)) + "\n"
                result = subprocess.run(
                    [str(executable)],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                cases.append(
                    dict(
                        trials=sizes,
                        alternative=alternative,
                        criteria=criteria,
                        score=list(map(float, result.stdout.split())),
                    )
                )
    fixture = dict(
        source=(
            "Unchanged SSSSRT (including CHISQ), XLR (including XMLBIN), "
            "QEQDBL; independent grid driver"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=["-O2", "-std=legacy"],
        cases=cases,
    )
    Path("tests/fixtures/ksbin2.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native grids")


if __name__ == "__main__":
    main()
