"""Native C/Fortran normal values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-normal").resolve()
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
long a,b,g,anti,x,y;int i;float mean,sd,z;
if(scanf("%ld %ld %ld %ld %f %f",&a,&b,&g,&anti,&mean,&sd)!=6)return 1;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){z=gennor(mean,sd);getsd(&x,&y);
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
                        "snorm",
                        "gennor",
                    ]
                ]
                declarations = "real gennor\nexternal gennor\n"
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_standard_normal_mod",
                        "random_normal_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_normal_mod, only: gennor=>random_normal
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i
real mean,sd,z
read(*,*)a,b,g,anti,mean,sd
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 z=gennor(mean,sd)
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
        # Engineered raw draws exercise every center strip and both tails.
        raw_values = [1, 2, 2147483562, 1073741825]
        raw_values += [
            int((j + fraction) / 64 * 2147483563)
            for j in range(1, 32)
            for fraction in (0.0001, 0.8)
        ]
        seeds = [(1234567890, 123456789)] + [
            (
                ((z + 1) % 2147483563 or 1) * pow(40014, -1, 2147483563) % 2147483563,
                pow(40692, -1, 2147483399),
            )
            for z in raw_values
            if z != 2147483562
        ]
        seeds.append((pow(40014, -1, 2147483563), pow(40692, -1, 2147483399)))
        for seed_index, seed in enumerate(seeds):
            for stream in [1, 2, 32] if seed_index == 0 else [1]:
                for antithetic in [False, True]:
                    for mean, sd in (
                        [(0, 1), (2.3, 0), (-2.3, 3.7)] if seed_index == 0 else [(0, 1)]
                    ):
                        output = subprocess.check_output(
                            [str(exe)],
                            input=f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {mean} {sd}\n",
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
                                sd=sd,
                                values=[float(row[0]) for row in rows],
                                states=[[int(row[1]), int(row[2])] for row in rows],
                            )
                        )
    Path("tests/fixtures/randlib_normal.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native normal cases")


if __name__ == "__main__":
    main()
