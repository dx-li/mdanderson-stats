"""Record the original first-stage table, including its lower-cutoff defect."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/KSBIN1/source/ksbin190_1.0/source")
    auxiliary = (root / "ksbin1_aux_mod.f90").read_text()
    original = (root / "ksbin1_main.f90").read_text()
    directory = Path("research/raw/reference/ksbin1-table").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name in ["BINCUM", "BINDEN"]:
        match = re.search(
            rf"^\s*FUNCTION {name}\s*\(.*?END FUNCTION {name}", auxiliary, re.M | re.S | re.I
        )
        if match is None:
            raise RuntimeError(name)
        blocks.append(match.group())
    numerical = directory / "numerical.f90"
    numerical.write_text(
        "module numerical\nimplicit none\ninteger,parameter::dpkind=kind(1d0)\n"
        "real(dpkind),parameter::zero=0d0,one=1d0\ncontains\n"
        + "\n".join(blocks)
        + "\nend module\n"
    )
    start = original.index("         do i = 0, ihi", original.index("dum = setbc"))
    end = original.index("         WRITE (hdrfmt", start)
    loop = original[start:end]
    driver = directory / "driver.f90"
    driver.write_text(
        """program reference
use numerical
implicit none
integer,parameter::ioff=-1,ihihld=-2,nx=2,ha=2
integer ip(2,2),i,ihi,ievt,ievtof,ncrit1,n,total,direction
real(dpkind) table(-4:200,5),p,powcon,dum
logical qhalo
read(*,*) n,total,ncrit1,direction,p
qhalo=direction==1
ip=1
ievtof=0
ihi=n
dum=setbd(n,p)
do i=0,n
 table(i,1)=binden(i)
enddo
dum=setbc(total-n,p)
"""
        + loop
        + """
write(*,'(*(ES27.17E3,1X))') table(0:n,1),table(0:n,5)
contains
function pnormr(n)
integer,intent(in)::n
real(dpkind) pnormr
pnormr=one-bincum(n-1)
end function
end program
"""
    )
    executable = directory / "reference"
    subprocess.run(
        [
            "gfortran",
            "-O2",
            "-std=legacy",
            "-fcheck=bounds",
            str(numerical),
            str(driver),
            "-o",
            str(executable),
        ],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for direction in [1, 2]:
        for p in [0.06, 0.2, 0.7]:
            for cutoff in [0, 4]:
                output = subprocess.run(
                    [str(executable)],
                    input=f"14 42 {cutoff} {direction} {p}\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                values = list(map(float, output.stdout.split()))
                cases.append(
                    dict(
                        trials=14,
                        total=42,
                        cutoff=cutoff,
                        probability=p,
                        alternative="less" if direction == 1 else "greater",
                        arrival=values[:15],
                        power_loss=values[15:],
                    )
                )
    fixture = dict(
        source=(
            "Unchanged BINCUM/BINDEN and main-program contribution/cumulation loop; "
            "independent first-stage driver"
        ),
        auxiliary_sha256=hashlib.sha256(auxiliary.encode()).hexdigest(),
        main_sha256=hashlib.sha256(original.encode()).hexdigest(),
        numerical_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=["-O2", "-std=legacy", "-fcheck=bounds"],
        cases=cases,
    )
    Path("tests/fixtures/ksbin1_table.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native tables")


if __name__ == "__main__":
    main()
