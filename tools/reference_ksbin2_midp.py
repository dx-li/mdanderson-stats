"""Native BRKARR display adjustment with nonzero earlier rejection probability."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from mdanderson_stats import KStageTwoSampleBinomial, ksbin2_boundary_table


def main():
    source = Path("research/raw/KSBIN2/source/ksbin290_2.1/source/ksbin2_aux_mod.f90")
    original = source.read_text()
    match = re.search(
        r"^\s*SUBROUTINE BRKARR\b.*?END SUBROUTINE BRKARR", original, re.M | re.S | re.I
    )
    if match is None:
        raise RuntimeError("BRKARR not found")
    directory = Path("research/raw/reference/ksbin2-midp").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    driver = directory / "driver.f90"
    driver.write_text(
        """module numerical
implicit none
integer,parameter::dpkind=kind(1d0)
real(dpkind),parameter::zero=0d0,half=.5d0
contains
"""
        + match.group()
        + """
end module
program reference
use numerical
implicit none
integer n,i,ends(10201)
real(dpkind) values(10201)
read(*,*) n
read(*,*) values(:n)
ends(:n)=[(i,i=1,n)]
call brkarr(ends,n,values,.true.,.true.)
write(*,'(*(ES27.17E3,1X))') values(:n)
end program
"""
    )
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", str(driver), "-o", str(executable)], check=True, capture_output=True
    )
    cases = []
    for alternative in ["less", "greater", "two-sided"]:
        design = KStageTwoSampleBinomial(
            [[2, 2], [3, 3], [4, 4]], [0, 0, 1], [2, 2], criteria=(1,), alternative=alternative
        )
        for stage in [1, 2, 3]:
            table = ksbin2_boundary_table(design, stage, 0.6, 0.2)
            data = (
                str(len(table.significance))
                + "\n"
                + " ".join(format(v, ".17g") for v in table.significance)
                + "\n"
            )
            output = subprocess.run(
                [str(executable)], input=data, text=True, capture_output=True, check=True, timeout=5
            )
            cases.append(
                dict(
                    alternative=alternative,
                    stage=stage,
                    input=table.significance.tolist(),
                    midp=list(map(float, output.stdout.split())),
                )
            )
    fixture = dict(
        source="Unchanged BRKARR; Python supplies cumulative grid maxima for complete groups",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=["-O2"],
        cases=cases,
    )
    Path("tests/fixtures/ksbin2_midp.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native mid-p adjustments")


if __name__ == "__main__":
    main()
