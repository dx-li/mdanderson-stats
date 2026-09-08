"""Native IGTUT and the sequential weight accumulation used by GENLST/WRKLST."""

import hashlib
import json
import subprocess
from pathlib import Path

from reference_ranlist_random import extract_units


def main():
    source = Path("research/raw/RANLIST/source/source/ranlist.f")
    work = Path("research/raw/reference/ranlist-unrestricted").resolve()
    work.mkdir(parents=True, exist_ok=True)
    numerical = work / "numerical.f"
    numerical.write_text(
        extract_units(
            source.read_text(),
            {"getcgn", "ignlgi", "initgn", "inrgcm", "mltmod", "qrgnin", "ranf", "setall", "igtut"},
        )
    )
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
integer nt,s1,s2,g,j,k,igtut,assigned
real weights(20),cumpr(20),total
read(*,*)nt,s1,s2,g
read(*,*)weights(1:nt)
total=0
do j=1,nt
 total=total+weights(j)
enddo
cumpr(1)=weights(1)/total
do j=2,nt
 cumpr(j)=cumpr(j-1)+weights(j)/total
enddo
write(*,'(A,20ES24.15)')'CDF ',cumpr(1:nt)
call setall(s1,s2)
do j=1,100
 assigned=igtut(cumpr,g,j)
 if(j<=10.or.j==25.or.j==50.or.j==100)then
  write(*,'(A,2I10)')'RESULT ',j,assigned
 endif
enddo
end program
""")
    exe = work / "reference"
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    subprocess.run(
        ["gfortran", *flags, str(numerical), str(driver), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    cases = []
    boundary_seed = (
        (1073741825 * pow(40014, -1, 2147483563)) % 2147483563,
        pow(40692, -1, 2147483399),
    )
    for weights in [[1], [1, 1], [1, 2, 3], [0.125, 0.375, 0.5], [1] * 20]:
        for seed in [(1234567890, 123456789), boundary_seed]:
            for stream in [1, 2, 20, 32]:
                data = (
                    f"{len(weights)} {seed[0]} {seed[1]} {stream}\n"
                    + " ".join(map(str, weights))
                    + "\n"
                )
                run = subprocess.run(
                    [str(exe)], input=data, text=True, capture_output=True, check=True
                )
                lines = run.stdout.splitlines()
                cdf = next(line.split()[1:] for line in lines if line.startswith("CDF "))
                rows = [line.split()[1:] for line in lines if line.startswith("RESULT ")]
                cases.append(
                    {
                        "weights": weights,
                        "seed": seed,
                        "stream": stream,
                        "cumulative": list(map(float, cdf)),
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
    Path("tests/fixtures/ranlist_unrestricted.json").write_text(
        json.dumps(fixture, indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native unrestricted allocation cases")


if __name__ == "__main__":
    main()
