"""Run CTA CHISQT unchanged in its archived Fortran, excluding the interactive main."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/CTA/source/cta0298/cta0298.f")
    work = Path("research/raw/reference/cta").resolve()
    work.mkdir(parents=True, exist_ok=True)
    numerical = work / "numerical.f"
    text = source.read_text()
    numerical.write_text(text[text.index("      SUBROUTINE bincomp") :])
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
real ob(100,100),ex(100,100),rows(100),cols(100),eps,obmin,c,y,cc,p,yp,cp
integer kr(100),kc(100),m,n,i,j,ind
read(*,*)m,n
ob=0
ex=0
do i=1,m
 read(*,*)ob(i,1:n)
enddo
ind=0
eps=5
cc=-1
cp=-1
open(unit=7,status='scratch')
call chisqt(ob,ex,m,n,eps,kr,kc,cols,rows,100,100,obmin,c,y,cc,p,yp,cp,ind)
write(*,'(A,7ES24.15)')'RESULT ',c,y,cc,p,yp,cp,obmin
do i=1,m
 write(*,'(A,100ES24.15)')'EXPECTED ',ex(i,1:n)
enddo
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
        [[12, 5], [7, 16]],
        [[1, 9], [11, 3]],
        [[10, 10], [10, 10]],
        [[3, 0], [0, 5]],
        [[2, 3, 5], [7, 11, 13]],
        [[1, 2, 3], [4, 6, 5], [7, 8, 10]],
        [[0.5, 2.5], [4.5, 6.5]],
        [[100, 200], [300, 400]],
        [[7, 2], [2, 7]],
    ]
    cases = []
    for table in tables:
        raw = (
            f"{len(table)} {len(table[0])}\n"
            + "\n".join(" ".join(map(str, row)) for row in table)
            + "\ny\n"
        )
        result = subprocess.run(
            [str(executable)], input=raw, text=True, capture_output=True, check=True, cwd=work
        )
        lines = result.stdout.splitlines()
        values = next(line for line in lines if line.startswith("RESULT ")).split()[1:]
        expected = [
            list(map(float, line.split()[1:])) for line in lines if line.startswith("EXPECTED ")
        ]
        cases.append(
            {"observed": table, "expected": expected, "statistics": list(map(float, values))}
        )
    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": ["-O0", "-std=legacy", "-ffixed-line-length-none"],
        "columns": ["pearson", "yates", "cochran", "p", "yates_p", "cochran_p", "minimum_expected"],
        "cases": cases,
    }
    Path("tests/fixtures/cta_chisqt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Generated {len(cases)} native CHISQT cases")


if __name__ == "__main__":
    main()
