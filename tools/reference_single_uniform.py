"""Native SINGLE response/derivative functions for fixed-design information."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/SINGLE/source/single/single.f")
    original = source.read_text()
    directory = Path("research/raw/reference/single_uniform").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name in ["cprob", "cdpdb", "mix", "gexp", "recgs", "recgsx", "qnxtix"]:
        match = re.search(
            r"^      (?:DOUBLE PRECISION FUNCTION|INTEGER FUNCTION|LOGICAL FUNCTION|SUBROUTINE) "
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
integer iu,ngrp,nmodel,npar,ncrit,criterion
real(8) rquan,low(3),high(3),recgs,loss
external loss
common /model/iu,ngrp,nmodel,npar
common /criter/rquan,ncrit
common /choice/criterion
read(*,*) nmodel,iu,ngrp,ncrit,criterion
npar=ngrp+1
read(*,*) low(:npar),high(:npar)
rquan=.05d0
write(*,'(ES27.17E3)') recgs(loss,npar,low,high,6)/product(high(:npar)-low(:npar))
end program
real(8) function loss(b)
implicit none
integer iu,ngrp,nmodel,npar,ncrit,criterion,i,j,k,g,indices(2),mix,compared
real(8) rquan,b(*),x(4),subjects(4),p,dp(2),h(3,3),cprob,v(3,3),c(3),pivot,mult,link,dose
common /model/iu,ngrp,nmodel,npar
common /criter/rquan,ncrit
common /choice/criterion
x=[-1d0,0d0,1d0,2d0]
subjects=[10d0,20d0,30d0,40d0]
h=0d0
v=0d0
c=0d0
do g=1,ngrp
 indices=[mix(g,1,ncrit),mix(g,2,ncrit)]
 do i=1,4
  p=cprob(g,b,x(i))
  call cdpdb(g,b,x(i),p,dp)
  do j=1,2
   do k=1,2
    h(indices(j),indices(k))=h(indices(j),indices(k))+subjects(i)*dp(j)*dp(k)/(p*(1-p))
   enddo
  enddo
 enddo
enddo
do i=1,npar
 v(i,i)=1d0
enddo
do i=1,npar
 pivot=h(i,i)
 h(i,:)=h(i,:)/pivot
 v(i,:)=v(i,:)/pivot
 do j=1,npar
  if(j==i) cycle
  mult=h(j,i)
  h(j,:)=h(j,:)-mult*h(i,:)
  v(j,:)=v(j,:)-mult*v(i,:)
 enddo
enddo
if(ngrp==2) then
 compared=2
 if((iu==1.and.ncrit==1).or.(iu==2.and.ncrit==2)) compared=1
 c(compared)=1d0
 c(3)=-1d0
else if(ncrit==1) then
 if(iu==1) c(2)=1d0
 if(iu==2) c(1)=1d0
else
 if(nmodel==1) then
  link=-log(-log(rquan))
 else
  link=log(rquan/(1-rquan))
 endif
 if(iu==1) then
  dose=(link-b(1))/b(2)
  c(1)=-1d0/b(2)
  c(2)=-dose/b(2)
 else
  c(1)=-link/b(1)**2
  c(2)=1d0
 endif
endif
loss=dot_product(c,matmul(v,c))
if(criterion==1) loss=sqrt(loss)
end function
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
            for groups in [1, 2]:
                for criterion in [1, 2]:
                    for target in [1, 2]:
                        low = [0.2, 0.8, 0.4][: groups + 1]
                        high = [0.4, 1.2, 0.6][: groups + 1]
                        output = subprocess.run(
                            [str(executable)],
                            input=f"{model} {form} {groups} {target} {criterion}\n"
                            + " ".join(map(str, low + high))
                            + "\n",
                            text=True,
                            capture_output=True,
                            check=True,
                            timeout=5,
                        )
                        suffix = "sd" if criterion == 1 else "variance"
                        cases.append(
                            dict(
                                model="loglog" if model == 1 else "logistic",
                                form="linear" if form == 1 else "centered",
                                comparison=None
                                if groups == 1
                                else ("location" if target == 1 else "slope"),
                                criterion=(
                                    ("slope_" if target == 1 else "quantile_")
                                    if groups == 1
                                    else ""
                                )
                                + suffix,
                                lower=low,
                                upper=high,
                                value=float(output.stdout),
                            )
                        )
    fixture = dict(
        source=(
            "Unchanged RECGS, RECGSX, QNXTIX (including entries), CPROB, CDPDB, MIX, GEXP; "
            "independent criterion callback"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/single_uniform.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native cases")


if __name__ == "__main__":
    main()
