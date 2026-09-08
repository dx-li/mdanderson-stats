"""Compile archived IGTRT/GENPRM/IGNUIN to verify restricted allocation."""

import hashlib
import json
import subprocess
from pathlib import Path

from reference_ranlist_random import extract_units


def main():
    source = Path("research/raw/RANLIST/source/source/ranlist.f")
    work = Path("research/raw/reference/ranlist-restricted").resolve()
    work.mkdir(parents=True, exist_ok=True)
    numerical = work / "numerical.f"
    numerical.write_text(
        extract_units(
            source.read_text(),
            {
                "getcgn",
                "ignlgi",
                "initgn",
                "inrgcm",
                "mltmod",
                "qrgnin",
                "ranf",
                "setall",
                "igtrt",
                "ignuin",
                "genprm",
            },
        )
    )
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
integer nt,s1,s2,g,lo,hi,i,j,n,k,m,igtrt,ignuin,counts(20),block(500)
read(*,*)nt,s1,s2,g,lo,hi
read(*,*)counts(1:nt)
n=0
do i=1,nt
 do j=1,counts(i)
  n=n+1
  block(n)=i
 enddo
enddo
call setall(s1,s2)
call setcgn(g)
call initgn(-1)
m=ignuin(lo,hi)
write(*,'(A,I10)')'MULT ',m
do i=1,2*n*m+3
 k=igtrt(block,n,g,i,lo,hi)
 write(*,'(A,2I10)')'RESULT ',i,k
enddo
end program
""")
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    exe = work / "reference"
    subprocess.run(["gfortran", *flags, str(numerical), str(driver), "-o", str(exe)], check=True)
    # First raw draw is the largest possible integer: forces a rejection for
    # source width two, whereas the unbiased sampler accepts the same draw.
    rejection_seed = (pow(40014, -1, 2147483563), pow(40692, -1, 2147483399))
    cases = []
    for counts, balance in [
        ([1], (1, 1)),
        ([1, 1], (1, 1)),
        ([1, 2, 3], (2, 2)),
        ([2, 3, 2], (1, 4)),
        ([1] * 20, (1, 2)),
        ([1, 1], (1, 2)),
    ]:
        for seed in [(1234567890, 123456789), rejection_seed]:
            for stream in [1, 2, 20, 32]:
                data = f"{len(counts)} {seed[0]} {seed[1]} {stream} {balance[0]} {balance[1]}\n"
                data += " ".join(map(str, counts)) + "\n"
                run = subprocess.run(
                    [str(exe)], input=data, text=True, capture_output=True, check=True
                )
                rows = [
                    line.split()[1:]
                    for line in run.stdout.splitlines()
                    if line.startswith("RESULT ")
                ]
                mult = next(
                    int(line.split()[1])
                    for line in run.stdout.splitlines()
                    if line.startswith("MULT ")
                )
                cases.append(
                    {
                        "counts": counts,
                        "balance": balance,
                        "seed": seed,
                        "stream": stream,
                        "multiplier": mult,
                        "patients": [int(row[0]) for row in rows],
                        "treatments": [int(row[1]) for row in rows],
                    }
                )
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
    }
    Path("tests/fixtures/ranlist_restricted.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native restricted allocation cases")


if __name__ == "__main__":
    main()
