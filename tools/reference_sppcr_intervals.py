"""Probe original SPPCR transforms and the expressions in print_answers."""

import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from reference_sppcr import ARCHIVE, SHA256

DRIVER = """program probe
use sppcr_aux_mod
implicit none
real(kind(0d0)) :: p,s,c,cs,a,lo,hi
read(*,*) p,s,c,cs
a=arcsin(p)
lo=c-1.959964d0*cs
hi=c+1.959964d0*cs
write(*,'(6ES25.16E3)') inv_arcsin(a-1.959964d0*s), &
 inv_arcsin(a+1.959964d0*s),lo,hi,1d0/hi,1d0/lo
end program
"""


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    with zipfile.ZipFile(ARCHIVE) as z:
        source = z.read("sppcr/source/sppcr_aux_mod.f90")
    records = []
    with tempfile.TemporaryDirectory(prefix="sppcr-intervals-") as tmp:
        w = Path(tmp)
        (w / "sppcr_aux_mod.f90").write_bytes(source)
        (w / "probe.f90").write_text(DRIVER)
        flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
        subprocess.run(
            [*flags, "sppcr_aux_mod.f90", "probe.f90", "-o", "probe"],
            cwd=w,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for p, s, c, cs in [
            (0.1, 0.02, 2, 0.1),
            (0.5, 0.1, 4, 0.5),
            (0.9, 0.02, 1, 0.05),
            (0.001, 0.5, 0.1, 1),
            (0.999, 0.5, 0.1, 1),
        ]:
            output = subprocess.run(
                [str(w / "probe")],
                input=f"{p} {s} {c} {cs}\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=3,
            )
            records.append(
                dict(
                    p=p,
                    transformed_sd=s,
                    calibration=c,
                    calibration_sd=cs,
                    stdout=output.stdout,
                    values=[float(x) for x in output.stdout.split()],
                    interior=s == 0.02 or s == 0.1,
                )
            )
    report = dict(
        archive_sha256=SHA256,
        source_sha256=hashlib.sha256(source).hexdigest(),
        driver=DRIVER,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        scope=(
            "Original transform module; report expressions transcribed into driver, "
            "not the complete report workflow"
        ),
        cases=records,
    )
    Path("tests/fixtures/sppcr_intervals.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
