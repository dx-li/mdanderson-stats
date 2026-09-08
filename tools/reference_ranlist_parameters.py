"""Native fixed-width records using MKLST's archived WRITE formats."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from mdanderson_stats import RanlistSession, RanlistSpecification, ranlist_parameter_text


def main():
    source = Path("research/raw/RANLIST/source/source/ranlist.f")
    work = Path("research/raw/reference/ranlist-parameters").resolve()
    work.mkdir(parents=True, exist_ok=True)
    driver = work / "driver.f90"
    driver.write_text("""program records
implicit none
integer mode,seed(2),nstrat,ntreat,counters(2),counts(3),i
real weights(3)
character header*25,title*80,phrase*31,strata(2)*30,treat(3)*30
logical restricted
read(*,*)mode
header='Parameter file for RANLST'
title='Native parameter fixture'
phrase='abc'
seed=(/8241812,190815113/)
nstrat=2
ntreat=3
strata=(/'North                         ','South                         '/)
treat(1)='Control'
treat(2)='Low'
treat(3)='High'
counters=(/12,345/)
counts=(/1,2,3/)
weights=(/0.0625,0.1,2.3456/)
restricted=mode==1
write(*,'(A25)')header
write(*,'(I1)')1
write(*,'(A80)')title
write(*,'(A31,I10,I10,I2,I2,L1)')phrase,seed,nstrat,ntreat,restricted
write(*,'(A30)')strata
write(*,'(A30)')treat
if(restricted)then
 write(*,'(I1)')counts
 write(*,'(2I3)')1,4
else
 write(*,'(F6.3)')weights
 write(*,'(2I3)')0,0
endif
write(*,'(I6)')counters
end program
""")
    exe = work / "reference"
    flags = ["-O0", "-std=legacy"]
    subprocess.run(["gfortran", *flags, str(driver), "-o", str(exe)], check=True)
    cases = []
    for mode in [1, 2]:
        text = subprocess.check_output([str(exe)], input=f"{mode}\n", text=True)
        cases.append({"restricted": mode == 1, "text": text})
    full_exe = work / "ranlist"
    full_flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    subprocess.run(
        ["gfortran", *full_flags, str(source.resolve()), "-o", str(full_exe)], check=True
    )
    workflows = []
    for restricted in [False, True]:
        for named in [False, True]:
            state = RanlistSession(
                RanlistSpecification(
                    (1, 2),
                    restricted=restricted,
                    legacy=True,
                    strata=("North", "South") if named else ("", ""),
                    balance=(1, 3) if restricted else (1, 1),
                ),
                (2, 3),
            )
            initial = ranlist_parameter_text(state)
            parameter = work / "workflow.par"
            parameter.write_text(initial)
            arrivals = [2, 1, 2, 2, 1]
            inputs = ["", "2", "workflow.par", "P", "Y", "N", "1"]
            for index, stream in enumerate(arrivals):
                inputs.extend([str(stream), "N" if index == len(arrivals) - 1 else "Y"])
            inputs.extend(["2", "2", "4", "N", "0", "0"])
            run = subprocess.run(
                [str(full_exe)],
                cwd=work,
                input="\n".join(inputs) + "\n",
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            )
            assignments = list(map(int, re.findall(r"is treatment\s+(\d+)", run.stdout)))
            if len(assignments) != len(arrivals) + 1:
                raise RuntimeError("native workflow did not produce all enrollment/inquiry results")
            workflows.append(
                {
                    "initial": initial,
                    "arrivals": arrivals,
                    "treatments": assignments[:-1],
                    "inquiry": assignments[-1],
                    "updated": parameter.read_text(),
                }
            )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
        "workflow_flags": full_flags,
        "workflows": workflows,
    }
    Path("tests/fixtures/ranlist_parameters.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print("Generated two native record fixtures and four complete-program workflows")


if __name__ == "__main__":
    main()
