"""Native SINGLE response/derivative functions for fixed-design information."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/SINGLE/source/single/single.f")
    original = source.read_text()
    directory = Path("research/raw/reference/single_normal").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name in ["cprob", "cdpdb", "mix", "gexp", "rechrm", "qinix"]:
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
integer iu,ngrp,nmodel,npar,ncrit,criterion,logs(3)
real(8) rquan,mu(3),variance(3),rechrm,strech,loss,dummy
external loss
common /model/iu,ngrp,nmodel,npar
common /criter/rquan,ncrit
common /choice/criterion
common /priorinput/mu,variance,logs
read(*,*) nmodel,iu,ngrp,ncrit,criterion,logs
npar=ngrp+1
read(*,*) mu(:npar),variance(:npar)
rquan=.05d0
dummy=strech(6)
write(*,'(ES27.17E3)') rechrm(loss,npar)
end program
real(8) function loss(y)
implicit none
integer iu,ngrp,nmodel,npar,i,logs(3)
real(8) y(*),mu(3),variance(3),b(3),precision
external precision
common /model/iu,ngrp,nmodel,npar
common /priorinput/mu,variance,logs
do i=1,npar
 b(i)=mu(i)+sqrt(variance(i))*y(i)
 if(logs(i)==1) b(i)=exp(b(i))
enddo
loss=1d0/precision(b)
end function
real(8) function precision(b)
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
precision=dot_product(c,matmul(v,c))
if(criterion==1) precision=sqrt(precision)
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
                        for logged in [0, 1]:
                            mean = ([-1.0, 0.0, -0.7] if logged else [0.3, 1.0, 0.5])[: groups + 1]
                            variance = [0.001, 0.002, 0.003][: groups + 1]
                            output = subprocess.run(
                                [str(executable)],
                                input=f"{model} {form} {groups} {target} {criterion} "
                                + f"{logged} {logged} {logged}\n"
                                + " ".join(map(str, mean + variance))
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
                                    mean=mean,
                                    variance=variance,
                                    lognormal=[bool(logged)] * (groups + 1),
                                    value=float(output.stdout),
                                )
                            )
    fixture = dict(
        source=(
            "Unchanged RECHRM, QINIX (including entries), CPROB, CDPDB, MIX, GEXP; "
            "independent diagonal-prior transformation and criterion callback"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/single_normal.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native cases")


if __name__ == "__main__":
    main()
