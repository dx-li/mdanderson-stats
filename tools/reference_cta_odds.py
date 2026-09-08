"""Compile unchanged CTA RELRISK and capture its odds ratio and legacy limits."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/CTA/source/cta0298/cta0298.f")
    work = Path("research/raw/reference/cta-odds").resolve()
    work.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    numerical = work / "numerical.f"
    numerical.write_text(text[text.index("      SUBROUTINE bincomp") :])
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
real ob(100,100),rr,se,lo,hi
integer i,istat
ob=0
istat=1
do i=1,2
 read(*,*)ob(i,1:2)
enddo
open(unit=7,status='scratch')
call relrisk(ob,2,100,rr,se,lo,hi,istat)
close(7)
write(*,'(A,4ES24.15)')'RESULT ',rr,se,lo,hi
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
        [[80, 10], [20, 90]],
        [[1, 9], [5, 15]],
        [[0.5, 2], [3, 4]],
        [[10, 10], [10, 10]],
    ]:
        for axis in ["rows", "columns"]:
            for response in [0, 1]:
                for alpha in [0.01, 0.05, 0.2]:
                    data = "\n".join(" ".join(map(str, row)) for row in table)
                    data += f"\n{axis[0]}\n{response + 1}\n{alpha}\n"
                    run = subprocess.run(
                        [str(executable)],
                        input=data,
                        text=True,
                        capture_output=True,
                        check=True,
                        cwd=work,
                    )
                    line = next(
                        line for line in run.stdout.splitlines() if line.startswith("RESULT ")
                    )
                    cases.append(
                        {
                            "observed": table,
                            "risk_factor": axis,
                            "response_index": response,
                            "alpha": alpha,
                            "result": list(map(float, line.split()[1:])),
                        }
                    )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
    }
    Path("tests/fixtures/cta_odds.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native RELRISK cases")


if __name__ == "__main__":
    main()
