"""Native SINGLE response/derivative functions for fixed-design information."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/SINGLE/source/single/single.f")
    original = source.read_text()
    directory = Path("research/raw/reference/single").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name in ["cprob", "cdpdb", "mix", "gexp"]:
        match = re.search(
            r"^      (?:DOUBLE PRECISION FUNCTION|INTEGER FUNCTION|SUBROUTINE) "
            + name
            + r"\(.*?^      END\s*$",
            original,
            re.M | re.S,
        )
        if match is None:
            raise RuntimeError(name)
        blocks.append(match.group())
    numerical = directory / "numerical.f"
    numerical.write_text("\n".join(blocks) + "\n")
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer iu,ngrp,nmodel,npar,ncrit,i,j,k
real(8) rquan,b(2),x(4),subjects(4),p,dp(2),h(2,2),cprob
common /model/iu,ngrp,nmodel,npar
common /criter/rquan,ncrit
read(*,*) nmodel,iu,b
ngrp=1
npar=2
ncrit=2
rquan=.05d0
x=[-1d0,0d0,1d0,2d0]
subjects=[10d0,20d0,30d0,40d0]
h=0d0
do i=1,4
 p=cprob(1,b,x(i))
 call cdpdb(1,b,x(i),p,dp)
 do j=1,2
  do k=1,2
   h(j,k)=h(j,k)+subjects(i)*dp(j)*dp(k)/(p*(1-p))
  enddo
 enddo
 write(*,'(*(ES27.17E3,1X))') p,dp
enddo
write(*,'(*(ES27.17E3,1X))') h(1,:),h(2,:)
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
    for model in [1, 2]:
        for form in [1, 2]:
            for parameters in [[0, 1], [0.2, 1.2], [-0.5, 0.75]]:
                output = subprocess.run(
                    [str(executable)],
                    input=f"{model} {form} {parameters[0]} {parameters[1]}\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                rows = [list(map(float, line.split())) for line in output.stdout.splitlines()]
                cases.append(
                    dict(
                        model="loglog" if model == 1 else "logistic",
                        form="linear" if form == 1 else "centered",
                        parameters=parameters,
                        probabilities=[r[0] for r in rows[:4]],
                        derivatives=[r[1:] for r in rows[:4]],
                        information=[rows[4][:2], rows[4][2:]],
                    )
                )
    fixture = dict(
        source="Unchanged CPROB, CDPDB, MIX, GEXP; independent information-matrix assembly",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/single.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native cases")


if __name__ == "__main__":
    main()
