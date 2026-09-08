"""Native C/Fortran multivariate_normal values and consumed generator states."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-multivariate_normal").resolve()
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
long a,b,g,anti,x,y,d,length;int i,j,r;float mean[16],cov[256],parm[153],z[16],work[16];
if(scanf("%ld %ld %ld %ld %ld",&a,&b,&g,&anti,&d)!=5)return 1;
if(d<1 || d>16)return 2;
for(i=0;i<d;i++)if(scanf("%f",&mean[i])!=1)return 3;
for(i=0;i<d;i++)for(j=0;j<d;j++)if(scanf("%f",&cov[i+j*d])!=1)return 4;
setall(a,b);gscgn(1,&g);setant(anti);
setgmn(mean,cov,d,parm);length=d*(d+3)/2+1;
for(i=0;i<length;i++)printf("%.9g ",parm[i]);printf("\\n");
for(r=0;r<20;r++){
genmn(parm,z,work);getsd(&x,&y);
for(j=0;j<d;j++)printf("%.9g ",z[j]);
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
                        "snorm",
                        "sdot",
                        "spofa",
                        "setgmn",
                        "genmn",
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
                        "random_standard_normal_mod",
                        "random_multivariate_normal_mod",
                    ]
                ]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator, setant=>set_antithetic, getsd=>get_seeds
use random_multivariate_normal_mod
"""
                declarations = ""
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,g,anti,x,y,i,j,d,r,length
real mean(16),cov(16,16),parm(153),z(16),work(16)
read(*,*)a,b,g,anti,d
if(d<1.or.d>16)stop 1
read(*,*)mean(:d)
do i=1,d
 read(*,*)cov(i,:d)
enddo
call setall(a,b)
call setcgn(g)
call setant(anti==1)
PREPARE
length=d*(d+3)/2+1
write(*,'(*(ES18.8E3,1X))')PARAMETERS
do r=1,20
 SAMPLE
 call getsd(x,y)
 write(*,'(*(ES18.8E3,1X))',advance='no')z(:d)
 write(*,'(2I16)')x,y
enddo
end program
"""
            )
            driver_text = driver.read_text()
            if language == "f77":
                driver_text = (
                    driver_text.replace("PREPARE", "call setgmn(mean,cov,16,d,parm)")
                    .replace("PARAMETERS", "parm(:length)")
                    .replace("SAMPLE", "call genmn(parm,z,work)")
                )
            else:
                driver_text = (
                    driver_text.replace(
                        "PREPARE",
                        "allocate(param(d*(d+3)/2+1))\n"
                        "call set_random_multivariate_normal(mean(:d),cov(:d,:d),d)",
                    )
                    .replace("PARAMETERS", "param(:length)")
                    .replace("SAMPLE", "call random_multivariate_normal(z(:d))")
                )
            driver.write_text(driver_text)
            flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
            compiler = "gfortran"
            command = [compiler, *flags, *map(str, selected), str(driver), "-o", str(exe)]
        subprocess.run(command, cwd=directory, check=True, capture_output=True)
        provenance[language] = {
            "compiler": subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            "flags": flags,
            "driver_note": (
                "Allocates public module param before setter; source omits allocation"
                if language == "f95"
                else "Caller supplies parameter workspace"
            ),
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in selected
            },
        }
        matrices = [
            ([0], [[1]]),
            ([2.3], [[4]]),
            ([0], [[1e-30]]),
            ([1e20], [[1e30]]),
            ([1, -2], [[4, -1], [-1, 2]]),
            ([0, 0], [[1, 0.999], [0.999, 1]]),
        ]
        for dimension in [3, 7, 8, 12]:
            lower = np.zeros((dimension, dimension))
            for i in range(dimension):
                lower[i, i] = 1 + 0.1 * i
                for j in range(i):
                    lower[i, j] = ((i + 2 * j) % 7) * 0.07 - 0.15
            matrices.append((np.linspace(-2, 3, dimension).tolist(), (lower @ lower.T).tolist()))
        seeds = [(1234567890, 123456789), (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))]
        for seed in seeds:
            for stream in [1, 32]:
                for antithetic in [False, True]:
                    for mean, covariance in matrices:
                        output = subprocess.check_output(
                            [str(exe)],
                            input=f"{seed[0]} {seed[1]} {stream} {int(antithetic)} {len(mean)}\n"
                            + " ".join(map(str, mean))
                            + "\n"
                            + "\n".join(" ".join(map(str, row)) for row in covariance)
                            + "\n",
                            text=True,
                            timeout=10,
                        )
                        rows = [line.split() for line in output.splitlines() if line.strip()]
                        if len(rows) != 21:
                            raise ValueError("expected parameters and 20 multivariate-normal draws")
                        cases.append(
                            dict(
                                language=language,
                                seed=seed,
                                stream=stream,
                                antithetic=antithetic,
                                mean=mean,
                                covariance=covariance,
                                packed_parameters=list(map(float, rows[0])),
                                values=[[float(x) for x in row[:-2]] for row in rows[1:]],
                                states=[[int(row[-2]), int(row[-1])] for row in rows[1:]],
                            )
                        )
    Path("tests/fixtures/randlib_multivariate_normal.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native multivariate_normal cases")


if __name__ == "__main__":
    main()
