"""Native CTA FISHXT tail sums and number of terms before its cutoff."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/CTA/source/cta0298/cta0298.f")
    work = Path("research/raw/reference/cta-fisher").resolve()
    work.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    numerical = work / "numerical.f"
    numerical.write_text(text[text.index("      SUBROUTINE bincomp") :])
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
real ob(100,2),x(50000),spr
integer i
ob=0
do i=1,2
 read(*,*)ob(i,1:2)
enddo
open(unit=7,status='scratch')
call fishxt(ob,spr,x,100)
close(7)
write(*,'(A,ES24.15)')'RESULT ',spr
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
        [[1, 9], [11, 3]],
        [[12, 5], [7, 16]],
        [[10, 10], [10, 10]],
        [[0, 4], [5, 0]],
        [[1, 2], [3, 4]],
        [[1, 5], [10, 20]],
        [[30, 40], [50, 60]],
        [[0, 0], [3, 7]],
        [[0, 0], [0, 0]],
        [[100, 100], [100, 100]],
    ]:
        for transpose in [False, True]:
            for reverse in [False, True]:
                ob = [list(row) for row in zip(*table, strict=True)] if transpose else table
                ob = ob[::-1] if reverse else ob
                data = "\n".join(" ".join(map(str, row)) for row in ob) + "\n"
                run = subprocess.run(
                    [str(executable)],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True,
                    cwd=work,
                )
                lines = run.stdout.splitlines()
                result = next(line for line in lines if line.startswith("RESULT "))
                cases.append(
                    {
                        "observed": ob,
                        "pvalue": float(result.split()[1]),
                        "terms": sum("PR=" in line for line in lines),
                    }
                )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
    }
    Path("tests/fixtures/cta_fisher.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native FISHXT cases")


if __name__ == "__main__":
    main()
