"""Native CUMINC CINC estimates and variances, including the supplied dataset."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np


def main():
    source = Path("research/raw/CUMINC/source/S/cuminc.f")
    match = re.search(r"^      SUBROUTINE cinc\(.*?^      END\s*$", source.read_text(), re.M | re.S)
    if match is None:
        raise RuntimeError("CINC not found")
    directory = Path("research/raw/reference/cuminc").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    numerical = directory / "cinc.f"
    numerical.write_text(match.group() + "\n")
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer n,i,used
integer,allocatable::status(:),failed(:),target(:)
real(8),allocatable::t(:),x(:),f(:),v(:)
read(*,*) n
allocate(t(n),status(n),failed(n),target(n),x(2*n+2),f(2*n+2),v(2*n+2))
do i=1,n
 read(*,*)t(i),status(i)
enddo
failed=merge(1,0,status/=0)
target=merge(1,0,status==1)
call cinc(t,failed,target,n,x,f,v,used)
do i=1,used
 write(*,'(*(ES27.17E3,1X))')x(i),f(i),v(i)
enddo
end program
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(numerical), str(driver), "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    inputs = [
        ([1, 2, 3, 4], [1, 2, 0, 1]),
        ([1, 1, 2, 2, 3, 3], [1, 2, 0, 1, 1, 2]),
        ([1, 2, 3], [0, 0, 0]),
        ([1, 2, 3], [2, 2, 2]),
        ([0, 0, 1, 2], [1, 2, 0, 1]),
        ([1], [1]),
        ([1], [0]),
        ([1, 1, 1, 1], [1, 1, 2, 2]),
        ([1, 2, 3, 4], [1, 1, 1, 1]),
    ]
    data_path = Path("research/raw/CUMINC/source/S/test.data")
    data = np.loadtxt(data_path)
    for group in np.unique(data[:, 2]):
        for cause in [1, 2]:
            rows = data[data[:, 2] == group]
            codes = np.where(rows[:, 1] == 0, 0, np.where(rows[:, 1] == cause, 1, 2))
            inputs.append((rows[:, 0].tolist(), codes.tolist()))
    cases = []
    for times, codes in inputs:
        ordered = sorted(zip(times, codes, strict=True))
        output = subprocess.run(
            [str(executable)],
            input=str(len(times)) + "\n" + "\n".join(f"{t} {int(c)}" for t, c in ordered) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=5,
        )
        rows = [list(map(float, line.split())) for line in output.stdout.splitlines()]
        cases.append(dict(time=times, event=codes, corners=rows))
    fixture = dict(
        source=(
            "Unchanged CINC, independent driver; "
            "original supplied-data group/cause subsets included"
        ),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(driver.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/cuminc.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native curves")


if __name__ == "__main__":
    main()
