"""Record KSBIN1 transition and expected-sample-size reference calculations."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/KSBIN1/source/ksbin190_1.0/source/ksbin1_aux_mod.f90")
    archive = Path("research/raw/KSBIN1/KSBIN1_V1.tar.gz")
    directory = Path("research/raw/reference/ksbin1").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = source.read_text()
    blocks = []
    for name, kind in [
        ("NXTSTG", "SUBROUTINE"),
        ("BINDEN", "FUNCTION"),
        ("EXPCOR", "FUNCTION"),
        ("EXPOV", "FUNCTION"),
    ]:
        match = re.search(
            r"^\s*" + kind + r" " + name + r"\s*\(.*?END " + kind + " " + name,
            original,
            re.M | re.S | re.I,
        )
        if match is None:
            raise RuntimeError(name)
        blocks.append(match.group())
    numerical = directory / "numerical.f90"
    numerical.write_text(
        "module numerical\nimplicit none\ninteger,parameter::dpkind=kind(1d0)\n"
        "real(dpkind),parameter::zero=0d0,one=1d0\ncontains\n"
        + "\n".join(blocks)
        + "\nend module\n"
    )
    driver = directory / "driver.f90"
    driver.write_text("""program reference
use numerical
implicit none
integer stages, direction, n(10), critical(10), quit(10), i,j,previous,total
real(dpkind) p, last(0:200), mass(0:200), rej(10), qt(10), cont(10)
logical reject, stop
read(*,*) stages,direction,p
read(*,*) n(:stages)
read(*,*) critical(:stages)
read(*,*) quit(:stages)
last=zero
last(0)=one
previous=0
do i=1,stages
 total=n(i)
 mass=zero
 call nxtstg(last(0:previous),0,previous,mass(0:total),total-previous,p)
 rej(i)=zero
 qt(i)=zero
 cont(i)=zero
 last=zero
 do j=0,total
  reject=critical(i)>=0.and.((direction==1.and.j<=critical(i)).or.(direction==2.and.j>=critical(i)))
  stop=quit(i)>=0.and.((direction==1.and.j>=quit(i)).or.(direction==2.and.j<=quit(i)))
  if(i==stages) stop=.not.reject
  if(reject) then
   rej(i)=rej(i)+mass(j)
  else if(stop) then
   qt(i)=qt(i)+mass(j)
  else
   cont(i)=cont(i)+mass(j)
   last(j)=mass(j)
  endif
 enddo
 previous=total
enddo
write(*,'(*(ES27.17E3,1X))') rej(:stages),qt(:stages),cont(:stages), &
 expov(n,stages,rej,qt),expcor(n,stages,rej),expcor(n,stages,qt)
end program
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(numerical), str(driver), "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for totals, critical, quit in [
        ([14, 28, 42], [0, 1, 3], [3, 4]),
        ([5, 10], [-1, 3], [-1]),
        ([20], [5], []),
    ]:
        for greater in (False, True):
            cuts = (
                [n - c if c >= 0 else -1 for n, c in zip(totals, critical)] if greater else critical
            )
            quits = [n - q if q >= 0 else -1 for n, q in zip(totals, quit)] if greater else quit
            for p in [0.06, 0.2, 0.7]:
                data = f"{len(totals)} {2 if greater else 1} {p}\n"
                for row in [totals, cuts, [*quits, -1]]:
                    data += " ".join(map(str, row)) + "\n"
                result = subprocess.run(
                    [str(executable)],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                values = list(map(float, result.stdout.split()))
                cases.append(
                    dict(
                        totals=totals,
                        critical=cuts,
                        quit=quits,
                        alternative="greater" if greater else "less",
                        probability=p,
                        rejection=values[: len(totals)],
                        quitting=values[len(totals) : 2 * len(totals)],
                        continuation=values[2 * len(totals) : 3 * len(totals)],
                        expected=values[-3:],
                    )
                )
    fixture = dict(
        source=(
            "Unchanged NXTSTG, BINDEN, EXPCOR, EXPOV; independent driver applies fixed boundaries"
        ),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        extracted_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=["-O2", "-std=legacy"],
        cases=cases,
    )
    Path("tests/fixtures/ksbin1.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native cases")


if __name__ == "__main__":
    main()
