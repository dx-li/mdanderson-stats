"""Run original TDTASP fixed-observation binomial power and critical regions."""

import hashlib
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use power_bin1_mod
implicit none
real(8) n,pa,sig,result,critical
read(*,*) n,pa,sig
result=power_bin1(.5d0,pa,n,sig)
if(n>0)then
critical=crbin1(.5d0,pa,n,sig)
else
critical=-1
end if
write(*,'(2ES26.17E3)') result,critical
end program reference
"""


def main():
    source = Path("research/raw/TDTASP/source/source/tdtasp_1.1/source").resolve()
    work = Path("research/raw/reference/tdtasp-power").resolve()
    work.mkdir(parents=True, exist_ok=True)
    names = [
        "zero_finder",
        "cdf_aux_mod",
        "beta_gamma_mod",
        "cdf_beta_mod",
        "cdf_binomial_mod",
        "power_bin1_mod",
    ]
    paths = [source / f"{name}.f90" for name in names]
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    executable = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, paths),
        str(driver),
        "-o",
        str(executable),
    ]
    subprocess.run(command, cwd=work, check=True)
    cases, errors = [], []
    for n in [0, 1, 5, 10, 25, 100, 1000]:
        for p in [0.2, 0.5, 0.8]:
            for alpha in [0.01, 0.05, 0.25]:
                process = subprocess.run(
                    [str(executable)],
                    input=f"{n} {p} {alpha}\n",
                    cwd=work,
                    text=True,
                    timeout=30,
                    capture_output=True,
                    check=True,
                )
                if "Input argument out of range" in process.stdout:
                    errors.append(
                        {
                            "observations": n,
                            "answer_probability": p,
                            "alpha": alpha,
                            "stdout": process.stdout,
                            "stderr": process.stderr,
                        }
                    )
                    continue
                power, critical = map(float, process.stdout.split())
                cases.append(
                    {
                        "observations": n,
                        "answer_probability": p,
                        "alpha": alpha,
                        "power": power,
                        "critical": critical,
                    }
                )
    data = {
        "provenance": {
            "source_changes": "None; original source routines compiled with a separate driver.",
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "build_command": command,
            "driver": DRIVER,
            "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[
                0
            ],
            "archive_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TDTASP/TDTASP%20%20_V1.tar.gz",
            "archive_sha256": hashlib.sha256(
                Path("research/raw/TDTASP/TDTASP  _V1.tar.gz").read_bytes()
            ).hexdigest(),
        },
        "cases": cases,
        "native_errors": errors,
    }
    Path("tests/fixtures/tdtasp_fixed_power.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {len(cases)} native results and {len(errors)} recorded native errors")


if __name__ == "__main__":
    main()
