"""Compile unchanged C/F77 machine constants and C translation helpers."""

import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256

C_DRIVER = r"""#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "cdflib.h"
int main(void) {
 char op[20]; int i; double a,b,c[4]={1,-2,3,1e99}; long ia,ib;
 if(sizeof(long)!=8) return 2;
 if(scanf("%19s %d",op,&i)!=2) return 2;
 if(!strcmp(op,"ftnstop")) ftnstop("reference stop");
 if(!strcmp(op,"fifmod")) {
   if(sizeof(long)!=8 || scanf("%ld %ld",&ia,&ib)!=2) return 2;
   printf("%ld\n",fifmod(ia,ib)); return 0;
 }
 if(scanf("%lf %lf",&a,&b)!=2) return 2;
 if(!strcmp(op,"ipmpar")) printf("%d\n",ipmpar(&i));
 else if(!strcmp(op,"spmpar")) printf("%.17g\n",spmpar(&i));
 else if(!strcmp(op,"devlpl")) printf("%.17g\n",devlpl(c,&i,&a));
 else if(!strcmp(op,"fifdint")) printf("%.17g\n",fifdint(a));
 else if(!strcmp(op,"fifidint")) printf("%ld\n",fifidint(a));
 else if(!strcmp(op,"fifdmax1")) printf("%.17g\n",fifdmax1(a,b));
 else if(!strcmp(op,"fifdmin1")) printf("%.17g\n",fifdmin1(a,b));
 else if(!strcmp(op,"fifdsign")) printf("%.17g\n",fifdsign(a,b));
 else return 2;
 return 0;
}"""

F_DRIVER = """program probe
implicit none
character(20) op
integer i,ipmpar
real(8) a,b,c(4),spmpar,devlpl
external ipmpar,spmpar,devlpl
c=(/1d0,-2d0,3d0,1d99/)
read(*,*) op,i,a,b
select case(trim(op))
case('ipmpar')
write(*,*) ipmpar(i)
case('spmpar')
write(*,'(ES26.17E3)') spmpar(i)
case('devlpl')
write(*,'(ES26.17E3)') devlpl(c,i,a)
case default
stop 2
end select
end program probe
"""


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Archive hash mismatch")
    profiles = {}
    common = [("ipmpar", i, 0, 0) for i in range(1, 11)]
    common += [("spmpar", i, 0, 0) for i in range(1, 4)]
    common += [("devlpl", n, x, 0) for n in (1, 2, 3) for x in (-2, 0, 0.5, 2)]
    translations = [("ftnstop", 0, 0, 0)]
    translations += [
        (op, 0, x, 0)
        for op in ("fifdint", "fifidint")
        for x in (-1.75, -0.5, -0.0, 0.0, 0.5, 1.75, -(2**63), 2**63 - 1024)
    ]
    translations += [
        (op, 0, a, b)
        for op in ("fifdmax1", "fifdmin1", "fifdsign")
        for a, b in ((-2, 3), (3, -2), (2, 2), (-0.0, 0.0), (0.0, -0.0), (-0.0, -1), (1, -0.0))
    ]
    translations += [
        ("fifmod", 0, a, b)
        for a, b in (
            (-5, 3),
            (5, -3),
            (-5, -3),
            (5, 3),
            (0, -3),
            (2**63 - 1, 3),
            (-(2**63), 3),
            (1, -(2**63)),
        )
    ]
    with (
        tarfile.open(archive) as tar,
        tempfile.TemporaryDirectory(prefix="dcdflib-support-") as tmp,
    ):
        work = Path(tmp)
        for language in ("c", "fortran"):
            if language == "c":
                names = ["dcdflib.c", "ipmpar.c", "cdflib.h"]
                prefix, driver = "source/dcdflib.c/src/", C_DRIVER
                flags = ["cc", "-std=gnu89", "-O0", "-ffp-contract=off"]
                driver_path = work / "driver.c"
            else:
                names = ["ipmpar.f", "spmpar.f", "devlpl.f"]
                prefix, driver = "source/dcdflib.f/src/", F_DRIVER
                flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
                driver_path = work / "driver.f90"
            hashes = {}
            for name in names:
                member = tar.extractfile(prefix + name)
                if member is None:
                    raise RuntimeError(f"Missing source {name}")
                raw = member.read()
                (work / name).write_bytes(raw)
                hashes[prefix + name] = hashlib.sha256(raw).hexdigest()
            driver_path.write_text(driver)
            subprocess.run(
                [
                    *flags,
                    *[n for n in names if not n.endswith(".h")],
                    driver_path.name,
                    *(["-lm"] if language == "c" else []),
                    "-o",
                    "reference",
                ],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=60,
            )
            cases = []
            for op, i, a, b in common + (translations if language == "c" else []):
                data = f"{op} {i} {a} {b}\n"
                run = subprocess.run(
                    [str(work / "reference")], input=data, text=True, capture_output=True, timeout=3
                )
                if op != "ftnstop" and run.returncode:
                    raise RuntimeError(f"Probe failed: {op}: {run.stderr}")
                cases.append(
                    dict(
                        op=op,
                        index=i,
                        a=a,
                        b=b,
                        returncode=run.returncode,
                        stdout=run.stdout,
                        stderr=run.stderr,
                    )
                )
            profiles[language] = dict(
                source_hashes=hashes,
                driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
                flags=flags,
                link_flags=["-lm"] if language == "c" else [],
                compiler=subprocess.check_output([flags[0], "--version"], text=True).splitlines()[
                    0
                ],
                cases=cases,
            )
    result = dict(archive_sha256=ARCHIVE_SHA256, profiles=profiles)
    Path("tests/fixtures/dcdflib_support.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: len(v["cases"]) for k, v in profiles.items()})


if __name__ == "__main__":
    main()
