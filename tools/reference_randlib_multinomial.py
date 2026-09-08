"""Native C/Fortran multinomial values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-multinomial").resolve()
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
long a,b,g,anti,x,y,n,ncat,z[16];int i,j;float p[16];
if(scanf("%ld %ld %ld %ld %ld %ld",&a,&b,&g,&anti,&n,&ncat)!=6)return 1;
if(ncat<2 || ncat>16)return 2;
for(j=0;j<ncat;j++)if(scanf("%f",&p[j])!=1)return 3;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){
genmul(n,p,ncat,z);getsd(&x,&y);
for(j=0;j<ncat;j++)printf("%ld ",z[j]);
printf("%ld %ld\\n",x,y);}
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
                        "genmul",
                    ]
                ]
                declarations = ""
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_binomial_mod",
                        "random_multinomial_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_multinomial_mod, only: genmul=>random_multinomial
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i,n,ncat,z(16)
real p(16)
read(*,*)a,b,g,anti,n,ncat
if(ncat<2.or.ncat>16)stop 1
read(*,*)p(:ncat)
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 call genmul(n,p,ncat,z)
 call getsd(x,y)
 write(*,'(*(I16))')z(:ncat),x,y
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
            (0, [0.2, 0.3, 0.5]),
            (1, [0.9, 0.05, 0.05]),
            (10, [0.2, 0.3, 0.5]),
            (100, [0.2, 0.3, 0.5]),
            (1000, [0.8, 0.1, 0.1]),
            (10000, [0.1] * 10),
            (100, [0, 0.2, 0, 0.8]),
            (100, [0, 0, 1]),
            (100, [0.99999, 0.00001]),
            (1000000000, [1e-8, 1e-8, 1 - 2e-8]),
            (10, [0.3, 0.7]),
            (1000000, [0.2, 0.3, 0.5]),
        ]
        seeds = [(1234567890, 123456789), (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))]
        for seed in seeds:
            for stream in [1, 32]:
                for antithetic in [False, True]:
                    for n, p in pairs:
                        output = subprocess.check_output(
                            [str(exe)],
                            input=f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {n} {len(p)}\n"
                            + " ".join(map(str, p))
                            + "\n",
                            text=True,
                            timeout=10,
                        )
                        rows = [line.split() for line in output.splitlines() if line.strip()]
                        if len(rows) != 20:
                            raise ValueError("expected 20 multinomial draws")
                        cases.append(
                            dict(
                                language=language,
                                seed=seed,
                                stream=stream,
                                antithetic=antithetic,
                                n=n,
                                p=p,
                                values=[[int(x) for x in row[:-2]] for row in rows],
                                states=[[int(row[-2]), int(row[-1])] for row in rows],
                            )
                        )
    Path("tests/fixtures/randlib_multinomial.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native multinomial cases")


if __name__ == "__main__":
    main()
