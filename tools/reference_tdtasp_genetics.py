"""Generate family probabilities with the unmodified TDTASP offspring routine."""

import hashlib
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use offspring_mod
implicit none
real(8) theta,penetrance(0:2),nhet,affected,trans,sharing
integer f1,f2,m1,m2
logical father,mother
read(*,*) theta,penetrance(2),penetrance(1),penetrance(0)
do f1=0,3
do f2=0,3
do m1=0,3
do m2=0,3
call offspring(penetrance,theta,f1,f2,m1,m2,father,mother,nhet,affected,exp_p_trans=trans)
call offspring(penetrance,theta,f1,f2,m1,m2,father,mother,nhet,affected,exp_p_same=sharing)
write(*,'(2I2,4ES26.17E3)') merge(1,0,father),merge(1,0,mother),nhet,affected,trans,sharing
end do
end do
end do
end do
end program reference
"""


def main():
    source = Path("research/raw/TDTASP/source/source/tdtasp_1.1/source/offspring_mod.f90").resolve()
    archive = Path("research/raw/TDTASP/TDTASP  _V1.tar.gz")
    work = Path("research/raw/reference/tdtasp-genetics").resolve()
    work.mkdir(parents=True, exist_ok=True)
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    executable = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        str(source),
        str(driver),
        "-o",
        str(executable),
    ]
    subprocess.run(command, cwd=work, check=True)
    cases = []
    for penetrance in (
        [0.8, 0.3, 0.01],
        [1, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0.5, 0.5, 0.5],
        [0, 0, 0],
        [1e-55] * 3,
    ):
        for theta in (0, 0.1, 0.5):
            output = subprocess.check_output(
                [str(executable)],
                cwd=work,
                text=True,
                input=" ".join(map(str, [theta, *penetrance])) + "\n",
                timeout=30,
            )
            rows = [[float(x) for x in line.split()] for line in output.splitlines()]
            if len(rows) != 256 or any(len(row) != 6 for row in rows):
                raise RuntimeError("unexpected native output dimensions")
            cases.append({"penetrance": penetrance, "recombination": theta, "rows": rows})
    result = {
        "provenance": {
            "archive_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/"
            "TDTASP/TDTASP%20%20_V1.tar.gz",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[
                0
            ],
            "build_command": command,
            "driver": DRIVER,
            "source_changes": "None",
            "row_order": "f1,f2,m1,m2 nested 0..3, codes Bd,BD,Ad,AD",
            "columns": [
                "father_heterozygous",
                "mother_heterozygous",
                "n_heterozygous",
                "affected_probability",
                "transmission_probability",
                "sharing_probability",
            ],
            "note": "Sharing retains original duplicate transmission weighting and cutoff.",
        },
        "cases": cases,
    }
    Path("tests/fixtures/tdtasp_genetics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(cases)} models, {256 * len(cases)} parental-family rows")


if __name__ == "__main__":
    main()
