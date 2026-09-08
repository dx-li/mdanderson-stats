"""Native CTA SENSPEC including predictive values from its text report."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/CTA/source/cta0298/cta0298.f")
    work = Path("research/raw/reference/cta-diagnostic").resolve()
    work.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    numerical = work / "numerical.f"
    numerical.write_text(text[text.index("      SUBROUTINE bincomp") :])
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
real ob(100,100),sens,vsens,spec,vspec
integer i
ob=0
do i=1,2
 read(*,*)ob(i,1:2)
enddo
open(unit=7,file='report.txt',status='replace')
call senspec(ob,2,100,sens,vsens,spec,vspec)
close(7)
write(*,'(A,4ES24.15)')'RESULT ',sens,vsens,spec,vspec
end program
""")
    executable = work / "reference"
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    subprocess.run(
        ["gfortran", *flags, str(numerical), str(driver), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    cases = []
    for table in [[[80, 10], [20, 90]], [[1, 9], [5, 15]], [[0.5, 2], [3, 4]], [[10, 0], [0, 20]]]:
        for axis in ["rows", "columns"]:
            for pos in [0, 1]:
                data = (
                    "\n".join(" ".join(map(str, row)) for row in table)
                    + f"\n{axis[0]}\n{pos + 1}\n"
                )
                run = subprocess.run(
                    [str(executable)],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True,
                    cwd=work,
                )
                line = next(line for line in run.stdout.splitlines() if line.startswith("RESULT "))
                raw = list(map(float, line.split()[1:]))
                report = (work / "report.txt").read_text().splitlines()
                ppv = list(
                    map(
                        float,
                        next(line for line in report if "Positive Predicitive" in line).split()[
                            -2:
                        ],
                    )
                )
                npv = list(
                    map(
                        float,
                        next(line for line in report if "Negative Predicitive" in line).split()[
                            -2:
                        ],
                    )
                )
                cases.append(
                    {
                        "observed": table,
                        "standard": axis,
                        "positive_index": pos,
                        "probabilities": [raw[0], raw[2], ppv[0], npv[0]],
                        "source_errors": [raw[1], raw[3], ppv[1], npv[1]],
                    }
                )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
    }
    Path("tests/fixtures/cta_diagnostic.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native SENSPEC cases")


if __name__ == "__main__":
    main()
