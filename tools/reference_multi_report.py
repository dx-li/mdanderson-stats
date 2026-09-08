"""Run the original MULTI PDISP report writer on existing native adjustment fixtures."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/MULTI/source/multi3.f")
    text = source.read_text()
    start = text.index("      SUBROUTINE PDISP(")
    end = text.index("      SUBROUTINE PROMPT", start)
    directory = Path("research/raw/reference/multi-report")
    directory.mkdir(parents=True, exist_ok=True)
    native = directory / "pdisp.f"
    native.write_text(text[start:end])
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,i
integer,allocatable :: ord(:),rej(:)
double precision,allocatable :: x(:),p(:)
logical,allocatable :: q(:)
character(120) header(1)
read(*,*) n
allocate(ord(n),rej(n),x(n),p(n),q(n))
read(*,*) x
read(*,*) p
read(*,*) ord
read(*,*) rej
q=rej/=0
call pdisp(6,x,p,ord,q,n,1,header,0)
end program
subroutine vctbwk(start,down,n,headers,nh,row,nrow,next,done)
logical start,done
integer down,n,nh,nrow,next
character(*) headers(*),row
done=.true.
end subroutine
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(native), str(driver), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    previous = json.loads(Path("tests/fixtures/multi.json").read_text())
    cases = []
    for case in previous["adjustments"]:
        x = np.array(case["pvalues"])
        if x.size < 4:
            continue
        order = np.argsort(x, kind="stable")
        for method, adjusted in case["sorted_adjusted"].items():
            adjusted = np.array(adjusted)
            reject = adjusted <= 0.05
            data = f"{len(x)}\n"
            for array in (x[order], adjusted, order + 1, reject.astype(int)):
                data += " ".join(map(str, array)) + "\n"
            run = subprocess.run(
                [str(executable)], input=data, text=True, capture_output=True, check=True
            )
            rows = []
            for row in run.stdout.splitlines():
                fields = row.split()
                rows.append(
                    [
                        int(fields[0]),
                        int(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                        len(fields) == 5,
                    ]
                )
            cases.append(dict(pvalues=x.tolist(), method=method, rows=rows))
    record = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        notes="Original PDISP unchanged. VCTBWK terminal navigation stubbed to finished; "
        "report writer runs. Input adjustments come from existing native multi.json fixtures. "
        "Values above .999999 are printed as one by the source.",
        cases=cases,
    )
    Path("tests/fixtures/multi_report.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(f"{len(cases)} native report cases")


if __name__ == "__main__":
    main()
