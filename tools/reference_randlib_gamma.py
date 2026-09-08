"""Native C/Fortran gamma values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-gamma").resolve()
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
long a,b,g,anti,x,y;int i;float rate,shape,z,r;
if(scanf("%ld %ld %ld %ld %f %f",&a,&b,&g,&anti,&rate ,&shape)!=6)return 1;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){r=shape<0?(i%3==1?0.5f:5.0f):shape;
z=gengam(rate,r);getsd(&x,&y);
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
                        "sexpo",
                        "sgamma",
                        "gengam",
                    ]
                ]
                declarations = "real gengam\nexternal gengam\n"
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
                        "random_gamma_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_gamma_mod, only: gengam=>random_gamma
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i
real rate,shape,z,r
read(*,*)a,b,g,anti,rate,shape
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 r=shape
 if(shape<0)then
  r=5.0
  if(mod(i-1,3)==1)r=0.5
 endif
 z=gengam(rate,r)
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
        seeds = [
            (1234567890, 123456789),
            (1, 1),
            (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399)),
            (572756977, 60805783),  # GD large-quotient acceptance at shape 1
        ]
        shapes = [
            0.0001,
            0.01,
            0.1,
            0.5,
            0.99999994,
            1,
            1.0000001,
            3.686,
            3.6860003,
            13.022,
            13.022001,
            100,
            100000,
        ]
        for seed_index, seed in enumerate(seeds):
            for stream in [1] if seed_index == 0 else [1, 32]:
                for antithetic in [False, True]:
                    for shape in (
                        shapes + ([-1] if not antithetic else [])
                        if seed_index == 0
                        else [0.5, 1, 5, 20]
                    ):
                        if seed_index == 3 and (stream != 1 or antithetic or shape != 1):
                            continue
                        rate = 2.3 if antithetic else 1
                        output = subprocess.check_output(
                            [str(exe)],
                            input=(
                                f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {rate} {shape}\n"
                            ),
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
                                rate=rate,
                                shape=shape,
                                shapes=(
                                    [5 if i % 3 != 1 else 0.5 for i in range(20)]
                                    if shape < 0
                                    else [shape] * 20
                                ),
                                values=[float(row[0]) for row in rows],
                                states=[[int(row[1]), int(row[2])] for row in rows],
                            )
                        )
    Path("tests/fixtures/randlib_gamma.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native gamma cases")


if __name__ == "__main__":
    main()
