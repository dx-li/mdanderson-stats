"""Compile unchanged C/F77 normal and t starting/inversion helpers."""

import hashlib
import json
import math
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256

C_DRIVER = """#include <stdio.h>
#include "cdflib.h"
int main(void){int op;double a,p,q,x;
if(scanf("%d %lf %lf %lf",&op,&a,&p,&q)!=4)return 2;
switch(op){case 1:x=stvaln(&p);break;case 2:x=dinvnr(&p,&q);break;
case 3:x=dt1(&p,&q,&a);break;default:return 2;}
printf("%.17g\\n",x);return 0;}
"""
F_DRIVER = """program probe
implicit none
integer op
real(8) a,p,q,x,stvaln,dinvnr,dt1
external stvaln,dinvnr,dt1
read(*,*) op,a,p,q
select case(op)
case(1)
x=stvaln(p)
case(2)
x=dinvnr(p,q)
case(3)
x=dt1(p,q,a)
end select
write(*,'(ES26.17E3)') x
end program probe
"""


def cases():
    rows = []
    for op in [1, 2]:
        for p in [
            0.0,
            5e-324,
            1e-300,
            1e-10,
            0.01,
            0.25,
            math.nextafter(0.5, 0),
            0.5,
            math.nextafter(0.5, 1),
            0.75,
            0.99,
            1 - 1e-10,
            1.0,
        ]:
            rows.append(dict(op=op, df=1.0, p=p, q=1 - p))
        if op == 2:
            for q in [5e-324, 1e-300]:
                rows.append(dict(op=op, df=1.0, p=1.0, q=q))
    for df in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
        for p in [0.01, 0.25, 0.49, 0.5, 0.51, 0.75, 0.99]:
            rows.append(dict(op=3, df=df, p=p, q=1 - p))
    for df in [5e-324, 1e-100, 1e-80, 1e80, 1e308]:
        for p in [0.5, math.nextafter(0.5, 1), 0.99]:
            rows.append(dict(op=3, df=df, p=p, q=1 - p))
    return rows


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Archive hash mismatch")
    profiles = {}
    with (
        tarfile.open(archive) as tar,
        tempfile.TemporaryDirectory(prefix="dcdflib-quantiles-") as tmp,
    ):
        work = Path(tmp)
        for language in ["c", "fortran"]:
            if language == "c":
                names = ["dcdflib.c", "ipmpar.c", "cdflib.h"]
                prefix = "source/dcdflib.c/src/"
                driver, filename = C_DRIVER, "driver.c"
                flags = ["cc", "-std=gnu89", "-O0", "-ffp-contract=off"]
                link = ["-lm"]
            else:
                names = [
                    "stvaln.f",
                    "devlpl.f",
                    "dinvnr.f",
                    "cumnor.f",
                    "spmpar.f",
                    "ipmpar.f",
                    "dt1.f",
                ]
                prefix = "source/dcdflib.f/src/"
                driver, filename = F_DRIVER, "driver.f90"
                flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
                link = []
            hashes = {}
            for name in names:
                member = tar.extractfile(prefix + name)
                if member is None:
                    raise RuntimeError("Missing native source")
                raw = member.read()
                (work / name).write_bytes(raw)
                hashes[prefix + name] = hashlib.sha256(raw).hexdigest()
            (work / filename).write_text(driver)
            subprocess.run(
                [
                    *flags,
                    *[n for n in names if not n.endswith(".h")],
                    filename,
                    *link,
                    "-o",
                    "reference",
                ],
                cwd=work,
                check=True,
                capture_output=True,
                timeout=60,
            )
            records = []
            for row in cases():
                run = subprocess.run(
                    [str(work / "reference")],
                    input=" ".join(str(row[k]) for k in ["op", "df", "p", "q"]) + "\n",
                    text=True,
                    capture_output=True,
                    timeout=3,
                )
                if run.returncode:
                    raise RuntimeError(f"Native call failed: {row}: {run.stderr}")
                value = float(run.stdout)
                records.append(dict(**row, result=value if math.isfinite(value) else str(value)))
            profiles[language] = dict(
                source_hashes=hashes,
                driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
                flags=flags,
                link_flags=link,
                compiler=subprocess.check_output([flags[0], "--version"], text=True).splitlines()[
                    0
                ],
                cases=records,
            )
            print(language, len(records), "cases", flush=True)
    Path("tests/fixtures/dcdflib_quantile_helpers.json").write_text(
        json.dumps(dict(archive_sha256=ARCHIVE_SHA256, profiles=profiles), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
