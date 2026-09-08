"""Native CTA MCNEMAR output; source remains an unbundled local reference."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/CTA/source/cta0298/cta0298.f")
    work = Path("research/raw/reference/cta-mcnemar").resolve()
    work.mkdir(parents=True, exist_ok=True)
    numerical = work / "numerical.f"
    raw = source.read_text()
    numerical.write_text(raw[raw.index("      SUBROUTINE bincomp") :])
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
real ob(100,100),ct,c1,ch,pt,p1,ph
integer n,i
read(*,*)n
ob=0
do i=1,n
 read(*,*)ob(i,1:n)
enddo
open(unit=7,status='scratch')
call mcnemar(ob,n,100,ct,c1,ch,pt,p1,ph)
write(*,'(A,6ES24.15)')'RESULT ',ct,c1,ch,pt,p1,ph
end program
""")
    executable = work / "reference"
    subprocess.run(
        [
            "gfortran",
            "-O0",
            "-std=legacy",
            "-ffixed-line-length-none",
            str(numerical),
            str(driver),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    tables = [
        [[10, 5], [7, 12]],
        [[0, 1], [9, 0]],
        [[2, 4], [4, 3]],
        [[3, 1, 7], [2, 6, 3], [4, 8, 9]],
        [[2, 1, 2], [1, 3, 3], [2, 3, 4]],
        [[1, 0.5, 2.5], [1.5, 2, 3.5], [4.5, 5.5, 3]],
        [[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13]],
    ]
    cases = []
    for table in tables:
        data = str(len(table)) + "\n" + "\n".join(" ".join(map(str, row)) for row in table) + "\n"
        result = subprocess.run(
            [str(executable)], input=data, text=True, capture_output=True, check=True, cwd=work
        )
        line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
        cases.append({"observed": table, "result": list(map(float, line.split()[1:]))})
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": ["-O0", "-std=legacy", "-ffixed-line-length-none"],
        "columns": ["summed", "pooled", "heterogeneity", "summed_p", "pooled_p", "heterogeneity_p"],
        "cases": cases,
    }
    Path("tests/fixtures/cta_mcnemar.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native MCNEMAR cases")


if __name__ == "__main__":
    main()
