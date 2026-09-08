"""Native KSBIN2 probability routines on explicitly supplied ordered count grids."""

import hashlib
import json
import re
import subprocess
from math import comb
from pathlib import Path

from mdanderson_stats.ksbin2 import ksbin2_ordering


def main():
    source = Path("research/raw/KSBIN2/source/ksbin290_2.1/source/ksbin2_aux_mod.f90")
    original = source.read_text()
    directory = Path("research/raw/reference/ksbin2-probability").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name, kind in [
        ("SSPOW", "SUBROUTINE"),
        ("BRKARR", "SUBROUTINE"),
        ("SSSIG", "SUBROUTINE"),
        ("PQTAB", "FUNCTION"),
        ("PQTAB1", "FUNCTION"),
        ("QEQDBL", "FUNCTION"),
    ]:
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
        """module numerical
implicit none
integer,parameter::dpkind=kind(1d0),mxprob=51
real(dpkind),parameter::zero=0d0,one=1d0,two=2d0,half=.5d0
integer npts,nptgrp(2),side12,s1(10201),s2(10201)
real(dpkind) pa(2),bincof(10201),sig(10201),pow(10201)
contains
"""
        + "\n".join(blocks)
        + "\nend module\n"
    )
    driver = directory / "driver.f90"
    driver.write_text("""program reference
use numerical
implicit none
integer i,ng,ends(10201)
real(dpkind) old(51)
read(*,*) nptgrp,side12,pa,npts,ng
read(*,*) ends(:ng)
do i=1,npts
 read(*,*) s1(i),s2(i),bincof(i)
enddo
old=zero
call sssig(old,one)
call sspow(zero)
write(*,'(*(ES27.17E3,1X))') sig(ends(:ng)),pow(ends(:ng))
call brkarr(ends,ng,sig,.true.,.true.)
write(*,'(*(ES27.17E3,1X))') sig(ends(:ng))
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
    for n1, n2 in [(3, 4), (10, 10), (20, 15)]:
        for alternative in ["less", "greater", "two-sided"]:
            for criteria in [(1,), (1, 2)]:
                p1, p2 = (0.6, 0.2) if alternative == "greater" else (0.2, 0.6)
                ordering = ksbin2_ordering(n1, n2, criteria=criteria, alternative=alternative)
                side = 2 if alternative == "two-sided" else 1
                data = (
                    f"{n1} {n2} {side} {p1} {p2} {len(ordering.score)} {len(ordering.group_end)}\n"
                )
                data += " ".join(map(str, ordering.group_end + 1)) + "\n"
                for i, j in ordering.events:
                    coefficient = comb(n1, int(i)) * comb(n2, int(j)) * 2.0 ** (-n1 - n2)
                    data += f"{i} {j} {coefficient:.17g}\n"
                output = subprocess.run(
                    [str(executable)],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                values = list(map(float, output.stdout.split()))
                ng = len(ordering.group_end)
                cases.append(
                    dict(
                        trials=[n1, n2],
                        alternative=alternative,
                        criteria=criteria,
                        probabilities=[p1, p2],
                        significance=values[:ng],
                        power=values[ng : 2 * ng],
                        midp_significance=values[2 * ng :],
                    )
                )
    fixture = dict(
        source=(
            "Unchanged SSSIG, SSPOW, BRKARR, PQTAB, PQTAB1, QEQDBL; Python supplies ordered "
            "count pairs, exact-combinatorial scaled coefficients and tied group ends"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=["-O2", "-std=legacy"],
        cases=cases,
    )
    Path("tests/fixtures/ksbin2_probability.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} probability tables")


if __name__ == "__main__":
    main()
