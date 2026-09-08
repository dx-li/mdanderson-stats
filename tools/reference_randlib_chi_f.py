"""Native C/Fortran chi-square/F values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-chi-f").resolve()
    work.mkdir(parents=True, exist_ok=True)
    cases, provenance = [], {}
    for language in ["f77", "f95", "c"]:
        directory = work / language
        directory.mkdir(exist_ok=True)
        exe = directory / "reference"
        if language == "c":
            folder = root / "source/randlib.c/src"
            selected = [folder / name for name in ["com.c", "randlib.c", "linpack.c"]]
            driver = directory / "driver.c"
            driver.write_text("""#include "randlib.h"
#include <stdio.h>
int main(void){
long a,b,g,anti,x,y;int i,kind;float dfn,dfd,nc,z;
if(scanf("%ld %ld %ld %ld %d %f %f %f",&a,&b,&g,&anti,&kind,&dfn,&dfd,&nc)!=8)return 1;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){
if(kind==1)z=genchi(dfn);
else if(kind==2)z=gennch(dfn,nc);
else if(kind==3)z=genf(dfn,dfd);
else z=gennf(dfn,dfd,nc);
getsd(&x,&y);printf("VALUE %.15e %ld %ld\\n",(double)z,x,y);}
return 0;}
""")
            flags = ["-O0", "-std=gnu89", "-ffp-contract=off"]
            compiler = "cc"
            command = [
                compiler,
                *flags,
                "-I",
                str(folder),
                *map(str, selected),
                str(driver),
                "-lm",
                "-o",
                str(exe),
            ]
        else:
            driver = directory / "driver.f90"
            prefix = "program reference\n"
            if language == "f77":
                folder = root / "source/randlib.f/src"
                selected = [
                    folder / f"{name}.f"
                    for name in [
                        "getcgn",
                        "getsd",
                        "ignlgi",
                        "initgn",
                        "inrgcm",
                        "mltmod",
                        "qrgnin",
                        "setall",
                        "setant",
                        "ranf",
                        "snorm",
                        "sexpo",
                        "sgamma",
                        "genchi",
                        "gennch",
                        "genf",
                        "gennf",
                    ]
                ]
                declarations = "real genchi,gennch,genf,gennf\nexternal genchi,gennch,genf,gennf\n"
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_standard_normal_mod",
                        "random_standard_exponential_mod",
                        "random_standard_gamma_mod",
                        "random_chisq_mod",
                        "random_nc_chisq_mod",
                        "random_f_mod",
                        "random_nc_f_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_chisq_mod, only: genchi=>random_chisq
use random_nc_chisq_mod, only: gennch=>random_nc_chisq
use random_f_mod, only: genf=>random_f
use random_nc_f_mod, only: gennf=>random_nc_f
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i,kind
real dfn,dfd,nc,z
read(*,*)a,b,g,anti,kind,dfn,dfd,nc
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 select case(kind)
 case(1)
 z=genchi(dfn)
 case(2)
 z=gennch(dfn,nc)
 case(3)
 z=genf(dfn,dfd)
 case(4)
 z=gennf(dfn,dfd,nc)
 end select
 call getsd(x,y)
 write(*,'(A,ES24.15,2I16)')'VALUE ',z,x,y
enddo
end program
"""
            )
            flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
            compiler = "gfortran"
            command = [compiler, *flags, *map(str, selected), str(driver), "-o", str(exe)]
        subprocess.run(command, cwd=directory, check=True, capture_output=True)
        provenance[language] = {
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "flags": flags,
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in selected
            },
        }
        specifications = []
        for df in [0.0001, 0.5, 1, 5, 100]:
            specifications.append((1, df, 1, 0))
        for df in [1, 1.0000005, 1.0000009536743164, 1.0000011, 5, 100]:
            for nc in [0, 2.3, 100]:
                specifications.append((2, df, 1, nc))
        for dfn, dfd in [(0.5, 0.5), (1, 1), (5, 12), (100, 100), (5, 0.0001)]:
            specifications.append((3, dfn, dfd, 0))
        for dfn, dfd in [
            (1, 12),
            (1.0000005, 12),
            (1.0000009536743164, 12),
            (1.0000011, 12),
            (5, 12),
            (5, 0.0001),
        ]:
            for nc in [0, 2.3, 100]:
                specifications.append((4, dfn, dfd, nc))
        for stream in [1, 32]:
            for antithetic in [False, True]:
                for kind, dfn, dfd, nc in specifications:
                    seed = (1234567890, 123456789)
                    output = subprocess.run(
                        [str(exe)],
                        input=(
                            f"{seed[0]} {seed[1]} {stream} {int(antithetic)} "
                            f"{kind} {dfn} {dfd} {nc}\n"
                        ),
                        text=True,
                        capture_output=True,
                        check=True,
                        timeout=10,
                    )
                    rows = [
                        line.split()[1:]
                        for line in output.stdout.splitlines()
                        if line.startswith("VALUE ")
                    ]
                    if len(rows) != 20:
                        raise ValueError("native driver did not return 20 values")
                    cases.append(
                        dict(
                            language=language,
                            seed=seed,
                            stream=stream,
                            antithetic=antithetic,
                            kind=kind,
                            dfn=dfn,
                            dfd=dfd,
                            nc=nc,
                            truncated="returning" in output.stdout + output.stderr,
                            values=[float(row[0]) for row in rows],
                            states=[[int(row[1]), int(row[2])] for row in rows],
                        )
                    )
    Path("tests/fixtures/randlib_chi_f.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native chi-square/F cases")


if __name__ == "__main__":
    main()
