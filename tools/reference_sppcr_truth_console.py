"""Probe unchanged native truth dialogue and parameter-report output."""

import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from reference_sppcr import ARCHIVE, SHA256

DRIVER = """program probe
use structures_mod
use problem_in_mod, only: generate_parameters_in
implicit none
logical :: from_truth, simulations
open(unit=10,status="scratch",action="write")
call generate_parameters_in(from_truth,simulations,10)
close(10)
write(*,'(A,100ES25.16E3)') 'DNA ',dna
write(*,'(A,100ES25.16E3)') 'WELLS ',n_well
write(*,'(A,100ES25.16E3)') 'FREQUENCY ',true_freq
write(*,'(A,ES25.16E3)') 'CALIBRATION ',true_calibrate
write(*,'(A,2I8)') 'PARENTS ',progenitor
write(*,'(A,2L3)') 'CHOICES ',from_truth,simulations
end program
"""
CASES = {
    "basic": "2 3\n100\n2 5 3\n.25 1\n1.7\n1 2\ny\nn\n",
    "continued_homozygous": "2\n3\n20\n0 1\n0\n.1\n.5\n2\n2 2\ny\ny\n",
    "correct_wells": "1 1\n1001\n1000\n2\n.001\n.4\n1 1\ny\nn\n",
}


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    with zipfile.ZipFile(ARCHIVE) as z:
        root = "sppcr/source/"
        order = re.findall(r"-c (\w+\.f90)", z.read(root + "compile.sppcr").decode())
        sources = {n: z.read(root + n) for n in order if n != "sppcr.f90"}
    records = {}
    flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
    with tempfile.TemporaryDirectory(prefix="sppcr-truth-console-") as tmp:
        w = Path(tmp)
        for name, source in sources.items():
            (w / name).write_bytes(source)
        (w / "probe.f90").write_text(DRIVER)
        subprocess.run(
            [*flags, *sources, "probe.f90", "-o", "probe"],
            cwd=w,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for name, stdin in CASES.items():
            result = subprocess.run(
                [str(w / "probe")],
                input=stdin,
                text=True,
                capture_output=True,
                check=False,
                timeout=5,
            )
            if result.returncode:
                raise RuntimeError(f"{name}: {result.stderr}\n{result.stdout}")
            records[name] = dict(
                input=stdin,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
    Path("tests/fixtures/sppcr_truth_console.json").write_text(
        json.dumps(
            dict(
                archive_sha256=SHA256,
                source_sha256={n: hashlib.sha256(b).hexdigest() for n, b in sources.items()},
                compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[
                    0
                ],
                flags=flags,
                driver=DRIVER,
                scope=(
                    "Unchanged truth dialogue and report; "
                    "clock-seeded counts excluded from comparison"
                ),
                cases=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
