"""Record native SPPCR truth normalization and seeded experiment generation."""

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
use generate_mod, only: set_generate, generate
use ecuyer_cote_mod, only: set_all_seeds
implicit none
integer :: i,j,r
read(*,*) n_dna,n_allele
call initialize
read(*,*) dna
read(*,*) n_well
read(*,*) true_freq
read(*,*) true_calibrate
true_freq=true_freq/sum(true_freq)
write(*,'(100ES25.16E3)') true_freq
call set_all_seeds(1234567890,123456789)
call set_generate(.true.)
do r=1,3
 call generate
 do i=1,n_dna
  write(*,'(100I12)') (int(seen(i,j,boot)),j=1,n_allele)
 end do
end do
end program
"""
CASES = [
    dict(dna=[0.25, 1, 2], wells=[100, 100, 100], weights=[2, 5, 3], calibration=1.7),
    dict(dna=[0.1, 0.5], wells=[20, 50], weights=[0, 1, 0], calibration=2),
    dict(dna=[0.001, 10], wells=[1000, 25], weights=[1], calibration=0.4),
]


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    with zipfile.ZipFile(ARCHIVE) as z:
        root = "sppcr/source/"
        order = re.findall(r"-c (\w+\.f90)", z.read(root + "compile.sppcr").decode())
        sources = {n: z.read(root + n) for n in order if n != "sppcr.f90"}
    records = []
    flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
    with tempfile.TemporaryDirectory(prefix="sppcr-truth-") as tmp:
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
        for case in CASES:
            rows = [f"{len(case['dna'])} {len(case['weights'])}"]
            rows.extend(" ".join(map(str, case[k])) for k in ("dna", "wells", "weights"))
            rows.append(str(case["calibration"]))
            stdin = "\n".join(rows) + "\n"
            result = subprocess.run(
                [str(w / "probe")],
                input=stdin,
                text=True,
                capture_output=True,
                check=True,
                timeout=3,
            )
            lines = result.stdout.splitlines()
            records.append(
                dict(
                    **case,
                    stdin=stdin,
                    stdout=result.stdout,
                    frequency=[float(x) for x in lines[0].split()],
                    seen=[[int(x) for x in line.split()] for line in lines[1:]],
                )
            )
    Path("tests/fixtures/sppcr_truth.json").write_text(
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
                    "Native set_generate(true) and generate; "
                    "normalization expression from truth entry"
                ),
                cases=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
