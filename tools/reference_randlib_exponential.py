"""Native C/Fortran exponential values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-exponential").resolve()
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
long a,b,g,anti,x,y;int i;float mean,z;
if(scanf("%ld %ld %ld %ld %f",&a,&b,&g,&anti,&mean)!=5)return 1;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){z=genexp(mean);getsd(&x,&y);
printf("%.15e %ld %ld\\n",(double)z,x,y);}
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
                        "sexpo",
                        "genexp",
                    ]
                ]
                declarations = "real genexp\nexternal genexp\n"
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_standard_exponential_mod",
                        "random_exponential_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_exponential_mod, only: genexp=>random_exponential
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i
real mean,z
read(*,*)a,b,g,anti,mean
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 z=genexp(mean)
 call getsd(x,y)
 write(*,'(ES24.15,2I16)')z,x,y
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
        half_seed = (
            (1073741825 * pow(40014, -1, 2147483563)) % 2147483563,
            pow(40692, -1, 2147483399),
        )
        maximum_seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
        seeds = [(1234567890, 123456789), half_seed]
        if language != "c":
            seeds.append(maximum_seed)
        for seed in seeds:
            for stream in [1, 2, 32]:
                for antithetic in [False, True]:
                    for mean in [0, 1, 2.3]:
                        if seed == maximum_seed and (stream != 1 or antithetic or mean != 1):
                            continue
                        output = subprocess.check_output(
                            [str(exe)],
                            input=f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {mean}\n",
                            text=True,
                            timeout=10,
                        )
                        rows = [line.split() for line in output.splitlines() if line.strip()]
                        cases.append(
                            dict(
                                language=language,
                                seed=seed,
                                stream=stream,
                                antithetic=antithetic,
                                mean=mean,
                                values=[float(row[0]) for row in rows],
                                states=[[int(row[1]), int(row[2])] for row in rows],
                            )
                        )
    Path("tests/fixtures/randlib_exponential.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native exponential cases")


if __name__ == "__main__":
    main()
