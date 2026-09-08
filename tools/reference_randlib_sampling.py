"""Native C, Fortran 77 and Fortran 95 uniform/permutation comparisons."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-sampling").resolve()
    work.mkdir(parents=True, exist_ok=True)
    body = """implicit none
integer a,b,g,anti,lo,hi,i,x,y,z,v(7)
real lower,upper,u
read(*,*)a,b,g,anti,lo,hi,lower,upper
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,5
 z=ignuin(lo,hi)
 call getsd(x,y)
 write(*,'(A,3I16)')'I ',z,x,y
enddo
u=genunf(lower,upper)
call getsd(x,y)
write(*,'(A,ES24.15,2I16)')'U ',u,x,y
v=(/-3,-2,-1,0,1,2,3/)
call genprm(v,7)
call getsd(x,y)
write(*,'(A,9I16)')'P ',v,x,y
end program
"""
    c_body = """#include "randlib.h"
#include <stdio.h>
int main(void) {
long a,b,g,anti,lo,hi,x,y,z,v[7]={-3,-2,-1,0,1,2,3};
int i; float lower,upper,u;
if(scanf("%ld %ld %ld %ld %ld %ld %f %f",&a,&b,&g,&anti,&lo,&hi,&lower,&upper)!=8)return 1;
setall(a,b); gscgn(1,&g); setant(anti);
for(i=0;i<5;i++){z=ignuin(lo,hi);getsd(&x,&y);printf("I %ld %ld %ld\\n",z,x,y);}
u=genunf(lower,upper);getsd(&x,&y);printf("U %.15e %ld %ld\\n",(double)u,x,y);
genprm(v,7);getsd(&x,&y);printf("P ");
for(i=0;i<7;i++)printf("%ld ",v[i]);printf("%ld %ld\\n",x,y);
return 0;}
"""
    cases, provenance = [], {}
    for language in ["f77", "f95", "c"]:
        directory = work / language
        directory.mkdir(exist_ok=True)
        exe = directory / "reference"
        if language == "c":
            folder = root / "source/randlib.c/src"
            selected = [folder / name for name in ["com.c", "randlib.c", "linpack.c"]]
            driver = directory / "driver.c"
            driver.write_text(c_body)
            flags = ["-O0", "-std=gnu89", "-ffp-contract=off"]
            command = [
                "cc",
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
            flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
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
                        "ignuin",
                        "genprm",
                        "genunf",
                        "ranf",
                    ]
                ]
                text = body.replace(
                    "integer a,b,g,anti,lo,hi,i,x,y,z,v(7)",
                    "integer a,b,g,anti,lo,hi,i,x,y,z,v(7),ignuin\n"
                    "real genunf\nexternal ignuin,genunf",
                )
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_uniform_integer_mod",
                        "random_uniform_mod",
                        "random_permutation_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, &
setant=>set_antithetic, getsd=>get_seeds
use random_uniform_integer_mod, only: ignuin=>random_uniform_integer
use random_uniform_mod, only: genunf=>random_uniform
use random_permutation_mod, only: genprm=>random_permutation
"""
                text = body
            driver.write_text(prefix + text)
            command = ["gfortran", *flags, *map(str, selected), str(driver), "-o", str(exe)]
        subprocess.run(command, cwd=directory, check=True, capture_output=True)
        provenance[language] = {
            "compiler": subprocess.check_output(
                ["cc" if language == "c" else "gfortran", "--version"], text=True
            ).splitlines()[0],
            "flags": flags,
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in selected
            },
        }
        for seed in [
            (1234567890, 123456789),
            (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399)),
        ]:
            for stream in [1, 2, 32]:
                for antithetic in [False, True]:
                    for low, high in [(-3, 7), (1, 1), (0, 1), (0, 1073741782)]:
                        lower, upper = (0.25, 0.25) if low == high else (-2.3, 9.7)
                        data = (
                            f"{seed[0]} {seed[1]} {stream} {int(antithetic)} "
                            f"{low} {high} {lower} {upper}\n"
                        )
                        run = subprocess.check_output([str(exe)], input=data, text=True, timeout=10)
                        lines = [line.split() for line in run.splitlines() if line.strip()]
                        integers = [list(map(int, line[1:])) for line in lines if line[0] == "I"]
                        uniform = next(line[1:] for line in lines if line[0] == "U")
                        permutation = next(
                            list(map(int, line[1:])) for line in lines if line[0] == "P"
                        )
                        cases.append(
                            dict(
                                language=language,
                                seed=seed,
                                stream=stream,
                                antithetic=antithetic,
                                low=low,
                                high=high,
                                lower=lower,
                                upper=upper,
                                integers=integers,
                                uniform=[float(uniform[0]), *map(int, uniform[1:])],
                                permutation=permutation,
                            )
                        )
    fixture = {"provenance": provenance, "cases": cases}
    Path("tests/fixtures/randlib_sampling.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native sampling cases")


if __name__ == "__main__":
    main()
