"""Native RANLIST seed hashing, integer streams, blocks and RANF rounding."""

import hashlib
import json
import re
import subprocess
from pathlib import Path


def extract_units(text: str, names: set[str]) -> str:
    units = []
    for match in re.finditer(
        r"^      (?:SUBROUTINE|(?:INTEGER|REAL|LOGICAL) FUNCTION) (\w+)", text, re.MULTILINE
    ):
        if match[1] in names:
            end = re.search(r"^[ 0-9]{5} END[ \t]*$", text[match.start() :], re.MULTILINE)
            units.append(text[match.start() : match.start() + end.end()] + "\n")
    if len(units) != len(names):
        raise RuntimeError("missing native RNG units")
    return "".join(units)


def main():
    source = Path("research/raw/RANLIST/source/source/ranlist.f")
    work = Path("research/raw/reference/ranlist-random").resolve()
    work.mkdir(parents=True, exist_ok=True)
    text = source.read_text()
    names = {
        "getcgn",
        "ignlgi",
        "initgn",
        "inrgcm",
        "mltmod",
        "qrgnin",
        "ranf",
        "setall",
        "phrtsd",
        "lennob",
    }
    numerical = work / "numerical.f"
    numerical.write_text(extract_units(text, names))
    driver = work / "driver.f90"
    driver.write_text("""program reference
implicit none
integer s1,s2,g,block,i,j,z(1000),ignlgi
real u,ranf
read(*,*)s1,s2,g,block
call setall(s1,s2)
call setcgn(g)
call initgn(-1)
do j=1,block
 call initgn(1)
enddo
do i=1,1000
 z(i)=ignlgi()
enddo
call initgn(-1)
do j=1,block
 call initgn(1)
enddo
do i=1,1000
 u=ranf()
 if(i==1.or.i==2.or.i==3.or.i==10.or.i==100.or.i==1000)then
  write(*,'(A,2I14,ES24.15)')'RESULT ',i,z(i),u
 endif
enddo
end program
""")
    phrase_driver = work / "phrase.f90"
    phrase_driver.write_text("""program reference
implicit none
character(256) phrase
integer a,b
read(*,'(A)')phrase
call phrtsd(phrase,a,b)
write(*,'(A,2I14)')'RESULT ',a,b
end program
""")
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    for input_file, exe in [(driver, work / "reference"), (phrase_driver, work / "phrase")]:
        subprocess.run(
            ["gfortran", *flags, str(numerical), str(input_file), "-o", str(exe)],
            check=True,
            capture_output=True,
        )
    cases = []
    for seed in [(1234567890, 123456789), (1, 1), (2147483562, 2147483398)]:
        for stream in [1, 2, 17, 32]:
            for block in [0, 1, 3]:
                run = subprocess.run(
                    [str(work / "reference")],
                    input=f"{seed[0]} {seed[1]} {stream} {block}\n",
                    text=True,
                    capture_output=True,
                    check=True,
                )
                rows = [
                    line.split()[1:]
                    for line in run.stdout.splitlines()
                    if line.startswith("RESULT ")
                ]
                cases.append(
                    {
                        "seed": seed,
                        "stream": stream,
                        "positions": [block * 2**30 + int(row[0]) for row in rows],
                        "integers": [int(row[1]) for row in rows],
                        "uniform": [float(row[2]) for row in rows],
                    }
                )
    phrases = []
    for phrase in [
        "",
        "   ",
        "abc",
        "abc   ",
        "ABC",
        "a b",
        "!@#$%^&*()_+[];:'\"<>?,./",
        "-=\\",
        "trial 123",
    ]:
        run = subprocess.run(
            [str(work / "phrase")], input=phrase + "\n", text=True, capture_output=True, check=True
        )
        row = next(
            line.split()[1:] for line in run.stdout.splitlines() if line.startswith("RESULT ")
        )
        phrases.append({"phrase": phrase, "seed": list(map(int, row))})
    fixture = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": flags,
        "cases": cases,
        "phrases": phrases,
    }
    Path("tests/fixtures/ranlist_random.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native stream/block cases and {len(phrases)} phrase cases")


if __name__ == "__main__":
    main()
