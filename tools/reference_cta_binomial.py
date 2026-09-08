"""Capture unchanged CTA BINCOMP event choices and reported probabilities."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/CTA/source/cta0298/cta0298.f")
    work = Path("research/raw/reference/cta-binomial").resolve()
    work.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    numerical = work / "numerical.f"
    numerical.write_text(text[text.index("      SUBROUTINE bincomp") :])
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
real ob(100,2),pl,pu,ptot
integer i,m1,m2
ob=0
do i=1,2
 read(*,*)ob(i,1:2)
enddo
open(unit=7,status='scratch')
call bincomp(ob,2,100,m1,m2,pl,pu,ptot)
close(7)
write(*,'(A,2I10,3ES24.15)')'RESULT ',m1,m2,pl,pu,ptot
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
    for table in [
        [[1, 9], [5, 15]],
        [[0, 10], [3, 17]],
        [[10, 10], [10, 10]],
        [[9, 1], [15, 5]],
        [[0, 10], [0, 20]],
        [[4, 6], [5, 95]],
        [[5, 95], [4, 6]],
        [[25, 75], [30, 70]],
    ]:
        for groups in ["rows", "columns"]:
            oriented = (
                table if groups == "rows" else [list(row) for row in zip(*table, strict=True)]
            )
            for reverse in [False, True]:
                ob = (
                    oriented[::-1]
                    if reverse and groups == "rows"
                    else ([row[::-1] for row in oriented] if reverse else oriented)
                )
                data = "\n".join(" ".join(map(str, row)) for row in ob)
                data += f"\n{groups[0]}\ny\nn\n"
                run = subprocess.run(
                    [str(executable)],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True,
                    cwd=work,
                )
                result = next(
                    line for line in run.stdout.splitlines() if line.startswith("RESULT ")
                ).split()
                cases.append(
                    {
                        "observed": ob,
                        "groups": groups,
                        "events": list(map(int, result[1:3])),
                        "reported_tails": list(map(float, result[3:5])),
                        "pvalue": float(result[5]),
                    }
                )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
    }
    Path("tests/fixtures/cta_binomial.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native BINCOMP cases")


if __name__ == "__main__":
    main()
