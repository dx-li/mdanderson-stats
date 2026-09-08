"""Native C/Fortran beta values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-beta").resolve()
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
long a,b,g,anti,x,y;int i;float alpha,beta,z,pa,pb;
if(scanf("%ld %ld %ld %ld %f %f",&a,&b,&g,&anti,&alpha,&beta)!=6)return 1;
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<20;i++){
pa=alpha;pb=beta;
if(alpha<0){switch(i%4){
case 0:pa=2;pb=3;break;case 1:pa=0.5f;pb=0.25f;break;
case 2:pa=3;pb=2;break;default:pa=0.25f;pb=0.5f;}}
z=genbet(pa,pb);getsd(&x,&y);
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
                        "genbet",
                    ]
                ]
                declarations = "real genbet\nexternal genbet\n"
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [
                    folder / f"{name}.f90"
                    for name in [
                        "ecuyer_cote_mod",
                        "random_standard_uniform_mod",
                        "random_beta_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_beta_mod, only: genbet=>random_beta
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i
real alpha,beta,z,pa,pb
read(*,*)a,b,g,anti,alpha,beta
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,20
 pa=alpha
 pb=beta
 if(alpha<0)then
 select case(mod(i-1,4))
 case(0)
 pa=2;pb=3
 case(1)
 pa=0.5;pb=0.25
 case(2)
 pa=3;pb=2
 case(3)
 pa=0.25;pb=0.5
 end select
 endif
 z=genbet(pa,pb)
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
        pairs = [
            (0.1, 0.2),
            (0.2, 0.1),
            (0.5, 0.5),
            (1, 1),
            (1, 5),
            (5, 1),
            (1.0000001, 1.0000001),
            (2, 3),
            (3, 2),
            (100, 200),
            (0.0001, 0.0002),
            (1e-36, 1e-36),
            (1e-36, 2),
            (2, 1e-36),
            (0.01, 100),
            (100, 0.01),
            (-1, 0),
        ]
        pairs.append((1.0000001032014561e-37 if language == "c" else 1e-37, 2))
        settings = [
            ((1234567890, 123456789), stream, anti, pairs)
            for stream in [1, 32]
            for anti in [False, True]
        ]
        # The first raw value is maximal: C RANF rounds it to one.
        settings.append(
            (
                (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399)),
                1,
                False,
                [(2, 3), (0.5, 0.25), (0.0001, 2), (0.2, 0.19), (1e30, 0.2)],
            )
        )
        for seed, stream, antithetic, shape_pairs in settings:
            for alpha, beta in shape_pairs:
                output = subprocess.check_output(
                    [str(exe)],
                    input=(f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {alpha} {beta}\n"),
                    text=True,
                    timeout=10,
                )
                rows = [line.split() for line in output.splitlines() if line.strip()]
                if len(rows) != 20:
                    raise ValueError("expected 20 native values")
                shapes = (
                    [[2, 3], [0.5, 0.25], [3, 2], [0.25, 0.5]] * 5
                    if alpha < 0
                    else [[alpha, beta]] * 20
                )
                cases.append(
                    dict(
                        language=language,
                        seed=seed,
                        stream=stream,
                        antithetic=antithetic,
                        shapes=shapes,
                        mixed=alpha < 0,
                        values=[float(row[0]) for row in rows],
                        states=[[int(row[1]), int(row[2])] for row in rows],
                    )
                )
    Path("tests/fixtures/randlib_beta.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native beta cases")


if __name__ == "__main__":
    main()
