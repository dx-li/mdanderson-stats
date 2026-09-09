"""Compile unchanged C/F77 root finders and record native termination contracts."""

import hashlib
import json
import math
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256

C_DRIVER = """#include <stdio.h>
#include <math.h>
#include "cdflib.h"
int main(void){int mode,fn,status=0,n=0;unsigned long qleft=0,qhi=0;
double lo,hi,x,target,scale,fx=0,xlo=0,xhi=0,ast,rst,mul,atol,rtol;
if(scanf("%d %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
 &mode,&fn,&lo,&hi,&x,&target,&scale,&ast,&rst,&mul,&atol,&rtol)!=12)return 2;
if(mode==1)dstinv(&lo,&hi,&ast,&rst,&mul,&atol,&rtol);else dstzr(&lo,&hi,&atol,&rtol);
do{if(mode==1)dinvr(&status,&x,&fx,&qleft,&qhi);else dzror(&status,&x,&fx,&xlo,&xhi,&qleft,&qhi);
if(status!=1)break;if(++n>10000)return 3;
switch(fn){case 1:fx=(x-target)*scale;break;case 2:fx=(target-x)*scale;break;
case 3:fx=(x/scale)*(x/scale)-target;break;case 4:fx=target;break;
case 5:fx=exp(x)-target;break;case 6:fx=(x-target)*(x-target)*(x-target)*scale;break;
case 7:fx=x<target?-scale:scale;break;default:return 2;}
}while(1);
printf("%d %d %.17g %.17g %.17g %lu %lu\\n",status,n,x,xlo,xhi,qleft,qhi);return 0;}
"""
F_DRIVER = """program probe
implicit none
integer mode,fn,status,n
logical qleft,qhi
real(8) lo,hi,x,target,scale,fx,xlo,xhi,ast,rst,mul,atol,rtol
read(*,*) mode,fn,lo,hi,x,target,scale,ast,rst,mul,atol,rtol
status=0
n=0
fx=0
xlo=0
xhi=0
qleft=.false.
qhi=.false.
if(mode==1)then
call dstinv(lo,hi,ast,rst,mul,atol,rtol)
else
call dstzr(lo,hi,atol,rtol)
end if
do
if(mode==1)then
call dinvr(status,x,fx,qleft,qhi)
else
call dzror(status,x,fx,xlo,xhi,qleft,qhi)
end if
if(status/=1)exit
n=n+1
if(n>10000)stop 3
select case(fn)
case(1)
fx=(x-target)*scale
case(2)
fx=(target-x)*scale
case(3)
fx=(x/scale)**2-target
case(4)
fx=target
case(5)
fx=exp(x)-target
case(6)
fx=(x-target)**3*scale
case(7)
if(x<target)then
fx=-scale
else
fx=scale
end if
end select
end do
write(*,'(2I6,3ES26.17E3,2L3)') status,n,x,xlo,xhi,qleft,qhi
end program probe
"""
KEYS = [
    "mode",
    "fn",
    "low",
    "high",
    "initial",
    "target",
    "scale",
    "abs_step",
    "rel_step",
    "multiplier",
    "abs_tol",
    "rel_tol",
]


def cases():
    rows = []

    def add(
        mode,
        fn,
        low,
        high,
        initial,
        target,
        scale=1.0,
        abs_step=0.5,
        rel_step=0.5,
        multiplier=5.0,
        abs_tol=1e-10,
        rel_tol=1e-10,
    ):
        rows.append(
            dict(
                zip(
                    KEYS,
                    [
                        mode,
                        fn,
                        low,
                        high,
                        initial,
                        target,
                        scale,
                        abs_step,
                        rel_step,
                        multiplier,
                        abs_tol,
                        rel_tol,
                    ],
                    strict=True,
                )
            )
        )

    for mode in [1, 2]:
        for fn in [1, 2]:
            for target in [-20.0, -10.0, 0.125, 10.0, 20.0]:
                add(mode, fn, -10.0, 10.0, 1.0, target)
        for target in [-1.0, 0.0, 1.0]:
            add(mode, 4, -10.0, 10.0, 1.0, target)
        for atol, rtol in [(1e-4, 0.0), (0.0, 1e-8), (1e-12, 1e-12), (5e-324, 0.0)]:
            add(mode, 3, 0.0, 2.0, 1.0, 2.0, abs_tol=atol, rel_tol=rtol)
        for scale in [1e-300, 1e-100, 1.0, 1e100, 1e300]:
            add(
                mode,
                3,
                0.0,
                2 * scale,
                scale / 2,
                2.0,
                scale=scale,
                abs_step=scale / 2,
                abs_tol=0.0,
                rel_tol=1e-10,
            )
        for scale in [1e-300, 1e300]:
            add(mode, 1, -1.0, 1.0, 0.0, 0.23456789, scale=scale)
        add(mode, 5, 0.0, 2.0, 1.0, 2.0)
        add(mode, 6, -1.0, 1.0, 0.0, 0.23456789)
        add(mode, 7, -1.0, 1.0, 0.0, 0.3, abs_tol=1e-12, rel_tol=0.0)
        add(mode, 1, -1.7e308, 1.7e308, 0.0, 0.125)
        for target in [-1.0, 0.0, 1.0]:
            add(mode, 4, 0.0, 0.0, 0.0, target, abs_step=0.0, rel_step=0.0)
    add(1, 1, -10.0, 10.0, 20.0, 3.0)
    return rows


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Archive hash mismatch")
    profiles = {}
    with (
        tarfile.open(archive) as tar,
        tempfile.TemporaryDirectory(prefix="dcdflib-roots-") as tmp,
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
                names = ["dinvr.f", "dzror.f"]
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
                    input=" ".join(str(row[k]) for k in KEYS) + "\n",
                    text=True,
                    capture_output=True,
                    timeout=3,
                )
                fields = run.stdout.split()
                record = dict(
                    **row, returncode=run.returncode, stderr=run.stderr, stdout=run.stdout
                )
                if run.returncode == 0 and len(fields) == 7:
                    status, n, x, xlo, xhi, left, hi = fields

                    def number(raw):
                        value = float(raw)
                        return value if math.isfinite(value) else str(value)

                    record.update(
                        outcome="completed",
                        status=int(status),
                        evaluations=int(n),
                        x=number(x),
                        xlo=number(xlo),
                        xhi=number(xhi),
                        qleft=left in ["1", "T"],
                        qhi=hi in ["1", "T"],
                    )
                elif run.returncode == 3:
                    record["outcome"] = "driver_budget_exhausted"
                elif row["mode"] == 1 and not row["low"] <= row["initial"] <= row["high"]:
                    record["outcome"] = "native_stop"
                else:
                    raise RuntimeError(f"Unexpected native termination: {record}")
                records.append(record)
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
    Path("tests/fixtures/dcdflib_root.json").write_text(
        json.dumps(dict(archive_sha256=ARCHIVE_SHA256, profiles=profiles), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
