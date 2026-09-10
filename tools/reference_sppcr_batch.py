"""Native SPPCR batch-reader probes with controlled unused input storage."""

import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from reference_sppcr import ARCHIVE, SHA256

DRIVER = """program probe
use problem_in_mod, only: read_one_bat
use data_in_struct_mod, only: seen_in,size_in,n_well_in
use structures_mod
implicit none
integer :: i,tail
read(*,*) tail
seen_in=0
size_in=0
n_well_in=0
seen_in(50,3)=tail
call read_one_bat(5)
write(*,'(A,2I8)') 'dimensions ',n_dna,n_allele
write(*,'(A,100ES25.16E3)') 'dna ',dna
write(*,'(A,100ES25.16E3)') 'wells ',n_well
write(*,'(A,100I8)') 'sizes ',allele_size
write(*,'(A,2I8)') 'parents ',progenitor
do i=1,n_dna
write(*,'(A,100ES25.16E3)') 'seen ',seen(i,:,data)
end do
end program
"""
BASE = """nallele 3
nrun 2
nwell 20 30
allelesizes 100 102 104
progenitor 100 102
run 0.5 4 8 2
run 1 10 15 5
"""


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    with zipfile.ZipFile(ARCHIVE) as z:
        contents = {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}
    prefix = "sppcr/source/"
    order = re.findall(r"-c (\w+\.f90)", contents[prefix + "compile.sppcr"].decode())
    records = {}
    cases = {
        "basic": (BASE, 0),
        "continued_comments": (
            BASE.upper()
            .replace("NWELL 20 30", "NWELL 20 # continue\n30 / done")
            .replace("ALLELESIZES 100 102 104", "ALLELESIZES\n100 102\n104"),
            0,
        ),
        "exponent_dna": (BASE.replace("0.5", "5d-1").replace("run 1 ", "run 1e0 "), 0),
        "drop_unseen": (BASE.replace("4 8 2", "4 8 0").replace("10 15 5", "10 15 0"), 0),
        "unseen_parent": (BASE.replace("4 8 2", "0 8 2").replace("10 15 5", "0 15 5"), 0),
        "stale_unused_row": (BASE.replace("4 8 2", "4 8 0").replace("10 15 5", "10 15 0"), 1),
        "negative_seen": (BASE.replace("4 8 2", "-4 8 2"), 0),
    }
    with tempfile.TemporaryDirectory(prefix="sppcr-batch-") as tmp:
        w = Path(tmp)
        for n in order:
            (w / n).write_bytes(contents[prefix + n])
        (w / "probe.f90").write_text(DRIVER)
        flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
        subprocess.run([*flags, "-c", *order], cwd=w, check=True, capture_output=True, timeout=60)
        subprocess.run(
            [
                *flags,
                "probe.f90",
                *[str(Path(n).with_suffix(".o")) for n in order if n != "sppcr.f90"],
                "-o",
                "probe",
            ],
            cwd=w,
            check=True,
            capture_output=True,
            timeout=60,
        )
        for name, (value, tail) in cases.items():
            r = subprocess.run(
                [str(w / "probe")],
                input=f"{tail}\n{value}",
                text=True,
                capture_output=True,
                timeout=3,
            )
            records[name] = dict(
                input=value,
                unused_row_value=tail,
                returncode=r.returncode,
                stdout=r.stdout,
                stderr=r.stderr,
            )
    report = dict(
        archive_sha256=SHA256,
        driver=DRIVER,
        scope=(
            "Original batch reader and conversion; unused storage explicitly initialized, "
            "with one injected stale-row case"
        ),
        cases=records,
    )
    Path("tests/fixtures/sppcr_batch.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
