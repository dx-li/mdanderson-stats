"""Build unchanged KSB1CI numerical routines and record native reference cases."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def main():
    archive = Path("research/raw/KSB1CI/KSB1CI_V1.tar.gz")
    source = Path("research/raw/KSB1CI/source/ksb1ci.f")
    directory = Path("research/raw/reference/ksb1ci").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    names = [
        "alfact",
        "fprstb",
        "phicon",
        "plocon",
        "prstat",
        "prstby",
        "prstlo",
        "qrzero",
        "scbicf",
        "scbitm",
        "setks",
    ]
    blocks = []
    for name in names:
        match = re.search(
            r"^      (?:REAL FUNCTION|LOGICAL FUNCTION|SUBROUTINE) "
            + name
            + r"\(.*?^      END\s*$",
            text,
            re.M | re.S,
        )
        if match is None:
            raise RuntimeError(f"Missing routine {name}")
        blocks.append(match.group())
    numerical = directory / "numerical.f"
    numerical.write_text("\n".join(blocks) + "\n")
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
integer k,stage,events,n(10),lo(10),hi(10)
real p,level,plocon,phicon,prstby
read(*,*) k,stage,events,p,level
read(*,*) n(:k)
read(*,*) lo(:k)
read(*,*) hi(:k)
call setks(k,n,lo,hi)
write(*,'(3(ES24.16E3,1X))') prstby(stage,events,p), &
plocon(stage,events,level), phicon(stage,events,level)
end program
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(numerical), str(driver), "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    cases = []
    designs = [
        ([20], [], [], [(1, 0), (1, 5), (1, 10), (1, 19)]),
        ([14, 14, 14], [0, 1], [3, 4], [(1, 2), (2, 6), (3, 3), (3, 5)]),
        ([3, 4, 5], [-1, 1], [3, -1], [(2, 2), (3, 4)]),
        ([10, 10], [-1], [-1], [(2, 5), (2, 15)]),
    ]
    for sizes, low, high, stops in designs:
        for stage, events in stops:
            for p, level in [(0.06, 0.8), (0.2, 0.95), (0.7, 0.99)]:
                data = f"{len(sizes)} {stage} {events} {p} {level}\n"
                for row in (sizes, [*low, -1], [*high, -1]):
                    data += " ".join(map(str, row)) + "\n"
                result = subprocess.run(
                    [str(executable)],
                    input=data,
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                cdf, lower, upper = map(float, result.stdout.split())
                cases.append(
                    dict(
                        sizes=sizes,
                        low=low,
                        high=high,
                        stage=stage,
                        events=events,
                        probability=p,
                        confidence=level,
                        cdf=cdf,
                        lower=lower,
                        upper=upper,
                    )
                )
    fixture = dict(
        source="KSB1CI Version 1.1; numerical routines extracted unchanged",
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        numerical_sha256=hashlib.sha256(numerical.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=["-O2", "-std=legacy"],
        cases=cases,
    )
    Path("tests/fixtures/ksb1ci.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native cases")


if __name__ == "__main__":
    main()
