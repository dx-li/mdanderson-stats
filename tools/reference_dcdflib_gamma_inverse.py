"""Compile unchanged C/F77 incomplete-gamma inversion with native status codes."""

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
int main(void){int status=0;double a,p,q,x0,x=0;
if(scanf("%lf %lf %lf %lf",&a,&p,&q,&x0)!=4)return 2;
gaminv(&a,&x,&x0,&p,&q,&status);
printf("%d %.17g\\n",status,x);return 0;}
"""
F_DRIVER = """program probe
implicit none
integer status
real(8) a,p,q,x0,x
read(*,*) a,p,q,x0
x=0
status=0
call gaminv(a,x,x0,p,q,status)
write(*,'(I5,ES26.17E3)') status,x
end program probe
"""


def cases():
    rows = []
    for a in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 500.0]:
        for p in [1e-10, 0.01, 0.5, 0.9, 1 - 1e-10]:
            for x0 in [0.0, -1.0, 1.0, 100.0]:
                rows.append(dict(a=a, p=p, q=1 - p, x0=x0, kind="ordinary"))
    for a in [5e-324, 1e-300, 1e-13, 0.5, 1.0, 2.0, 1e20, 1e100, 1e308]:
        for p, q in [
            (0.0, 1.0),
            (1.0, 0.0),
            (5e-324, 1.0),
            (1.0, 5e-324),
            (1e-300, 1.0),
            (1.0, 1e-300),
            (0.5, 0.5),
        ]:
            for x0 in [0.0, 1.0]:
                rows.append(dict(a=a, p=p, q=q, x0=x0, kind="extreme"))
    for a, p, q, x0 in [
        (0.0, 0.5, 0.5, 0.0),
        (-1.0, 0.5, 0.5, 0.0),
        (1.0, 0.1, 0.1, 0.0),
        (2.0, 0.5, 0.5, 1e-100),
        (2.0, 0.5, 0.5, 1e100),
    ]:
        rows.append(
            dict(a=a, p=p, q=q, x0=x0, kind="invalid" if a <= 0 or p + q != 1 else "poor_hint")
        )
    return rows


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Archive hash mismatch")
    profiles = {}
    with (
        tarfile.open(archive) as tar,
        tempfile.TemporaryDirectory(prefix="dcdflib-gaminv-") as tmp,
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
                names = sorted(
                    Path(m.name).name
                    for m in tar.getmembers()
                    if m.isfile()
                    and m.name.startswith("source/dcdflib.f/src/")
                    and m.name.endswith(".f")
                )
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
                    input=" ".join(str(row[k]) for k in ["a", "p", "q", "x0"]) + "\n",
                    text=True,
                    capture_output=True,
                    timeout=3,
                )
                if run.returncode:
                    raise RuntimeError(f"Native call failed: {row}: {run.stderr}")
                status, raw = run.stdout.split()
                value = float(raw)
                records.append(
                    dict(
                        **row,
                        status=int(status),
                        result=value if math.isfinite(value) else str(value),
                    )
                )
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
    Path("tests/fixtures/dcdflib_gamma_inverse.json").write_text(
        json.dumps(dict(archive_sha256=ARCHIVE_SHA256, profiles=profiles), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
