"""Native C/Fortran Poisson values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-poisson").resolve()
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
long a,b,g,anti,x,y,z;int i;float mu,value;
if(scanf("%ld %ld %ld %ld %f",&a,&b,&g,&anti,&mu)!=5)return 1;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){
value=mu;
if(mu<0){switch(i%4){case 0:value=2;break;
case 1:value=10;break;case 2:value=9.9f;break;
default:value=100;}}
z=ignpoi(value);getsd(&x,&y);
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
                        "snorm",
                        "sexpo",
                        "ignpoi",
                    ]
                ]
                declarations = "integer ignpoi\nexternal ignpoi\n"
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_standard_normal_mod",
                        "random_standard_exponential_mod",
                        "random_poisson_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_poisson_mod, only: ignpoi=>random_poisson
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i,z
real mu,value
read(*,*)a,b,g,anti,mu
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 value=mu
 if(mu<0)then
 select case(mod(i-1,4))
 case(0)
 value=2
 case(1)
 value=10
 case(2)
 value=9.9
 case(3)
 value=100
 end select
 endif
 z=ignpoi(value)
 call getsd(x,y)
 write(*,'(3I16)')z,x,y
enddo
end program
"""
            )
            # Original Fortran caches l/p/q but omits SAVE for pp(35).
            # Give that table its intended lifetime without editing the source.
            flags = ["-O0", "-std=legacy", "-ffixed-line-length-none", "-fno-automatic"]
            compiler = "gfortran"
            command = [compiler, *flags, *map(str, selected), str(driver), "-o", str(exe)]
        subprocess.run(command, cwd=directory, check=True, capture_output=True)
        provenance[language] = {
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "flags": flags,
            "storage_note": (
                "Static local storage retains the original unsaved pp(35) table"
                if language != "c"
                else "Source static table unchanged"
            ),
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in selected
            },
        }
        means = [-1, 0, 1e-8, 0.1, 2, 5, 9.9, 9.999999, 10, 10.000001, 20, 100, 10000, 1e6]
        seeds = [(1234567890, 123456789), (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))]
        for seed in seeds:
            for stream in [1, 32]:
                for antithetic in [False, True]:
                    for mu in means:
                        output = subprocess.check_output(
                            [str(exe)],
                            input=f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {mu}\n",
                            text=True,
                            timeout=10,
                        )
                        rows = [line.split() for line in output.splitlines() if line.strip()]
                        if len(rows) != 20:
                            raise ValueError("expected 20 poisson draws")
                        cases.append(
                            dict(
                                language=language,
                                seed=seed,
                                stream=stream,
                                antithetic=antithetic,
                                mu=mu,
                                parameters=([2, 10, 9.9, 100] * 5 if mu < 0 else [mu] * 20),
                                values=[int(row[0]) for row in rows],
                                states=[[int(row[1]), int(row[2])] for row in rows],
                            )
                        )
    Path("tests/fixtures/randlib_poisson.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native poisson cases")


if __name__ == "__main__":
    main()
