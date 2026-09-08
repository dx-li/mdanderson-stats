"""Native C/Fortran binomial values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-binomial").resolve()
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
long a,b,g,anti,x,y,n,z,nn;int i;float p,prob;
if(scanf("%ld %ld %ld %ld %ld %f",&a,&b,&g,&anti,&n,&p)!=6)return 1;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){
nn=n;prob=p;
if(n<0){switch(i%4){case 0:nn=10;prob=.3f;break;
case 1:nn=1000;prob=.3f;break;case 2:nn=10;prob=.7f;break;
default:nn=100;prob=.3f;}}
z=ignbin(nn,prob);getsd(&x,&y);
printf("%ld %ld %ld\\n",z,x,y);}
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
                        "ignbin",
                    ]
                ]
                declarations = "integer ignbin\nexternal ignbin\n"
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_binomial_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_binomial_mod, only: ignbin=>random_binomial
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i,n,z,nn
real p,prob
read(*,*)a,b,g,anti,n,p
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 nn=n
 prob=p
 if(n<0)then
 select case(mod(i-1,4))
 case(0)
 nn=10;prob=.3
 case(1)
 nn=1000;prob=.3
 case(2)
 nn=10;prob=.7
 case(3)
 nn=100;prob=.3
 end select
 endif
 z=ignbin(nn,prob)
 call getsd(x,y)
 write(*,'(3I16)')z,x,y
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
        pairs = [
            (-1, 0.3),  # Mixed-parameter driver case
            (0, 0.3),
            (10, 0),
            (10, 1),
            (1, 0.5),
            (10, 0.3),
            (10, 0.7),
            (100, 0.299),
            (100, 0.3),
            (100, 0.301),
            (1000, 0.05),
            (1000, 0.5),
            (10000, 0.3),
            (1000000, 0.01),
            (1000000, 0.5),
            (1000000000, 1e-8),
        ]
        seeds = [(1234567890, 123456789), (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))]
        seeds.append((776226756, 594208771))  # Final Stirling acceptance
        for seed in seeds:
            for stream in [1, 32]:
                for antithetic in [False, True]:
                    for n, p in pairs:
                        if seed == seeds[-1] and (
                            stream != 1 or antithetic or (n, p) != (10000, 0.3)
                        ):
                            continue
                        output = subprocess.check_output(
                            [str(exe)],
                            input=f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {n} {p}\n",
                            text=True,
                            timeout=10,
                        )
                        rows = [line.split() for line in output.splitlines() if line.strip()]
                        if len(rows) != 20:
                            raise ValueError("expected 20 binomial draws")
                        cases.append(
                            dict(
                                language=language,
                                seed=seed,
                                stream=stream,
                                antithetic=antithetic,
                                n=n,
                                p=p,
                                parameters=(
                                    [[10, 0.3], [1000, 0.3], [10, 0.7], [100, 0.3]] * 5
                                    if n < 0
                                    else [[n, p]] * 20
                                ),
                                values=[int(row[0]) for row in rows],
                                states=[[int(row[1]), int(row[2])] for row in rows],
                            )
                        )
    Path("tests/fixtures/randlib_binomial.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native binomial cases")


if __name__ == "__main__":
    main()
