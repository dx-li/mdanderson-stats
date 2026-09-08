"""Compile native MULTI QLEX and record token/value oracles for RDDATA input."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/MULTI/source/multi3.f")
    original = source.read_text()
    start = original.index("      LOGICAL FUNCTION QLEX(")
    end = original.index("      LOGICAL FUNCTION qrgnin()", start)
    directory = Path("research/raw/reference/multi-input")
    directory.mkdir(parents=True, exist_ok=True)
    lexer = directory / "qlex.f"
    lexer.write_text(original[start:end])
    driver = directory / "driver.f90"
    driver.write_text("""program reference
implicit none
character(212) line,cval
character(2) kind
integer lc,i,ov
real r
double precision d
logical first,qlex,more
read(*,'(A)') line
first=.true.
do
more=qlex(line,first,kind,cval,lc,i,r,d,ov)
if(.not.more) exit
write(*,'(A,1X,I1,1X,ES27.17E3,1X,A)') kind,ov,d,cval(:lc)
enddo
end program
integer function i1mach(i)
integer i
i1mach=huge(1)
end function
real function r1mach(i)
integer i
r1mach=huge(1.)
end function
double precision function d1mach(i)
integer i
d1mach=huge(1d0)
end function
""")
    executable = directory / "reference"
    subprocess.run(
        ["gfortran", "-O2", "-std=legacy", str(lexer), str(driver), "-o", str(executable)],
        check=True,
    )
    lines = [
        "0 .1 0.2 1",
        "0.1,0.2;0.3:0.4",
        "1e-2 1.e-2 .1D+1 0.05E-1",
        "-1 -0 -0.1 + 0.2 +.3",
        "0.1 + .2 - .3 0.4",
        "\"0.2\" 'don''t' \"q\"",
        "p=.1 q=.2 a_b .3 $bad .4",
        "0.1\t0.2 ! 0.3 @ 0.4",
        ". .e1 .e 0. .2",
        "1.0e999 1.0e-999 2.0 -2.0",
        "0.1 q 0.2 Quit 0.3",
        "(0.1)[0.2]{0.3} / 0.4",
        "0.1234567890123456789012345 0.1.2 0.3",
        "0.1 'unfinished",
        "0.1 +",
        "0.1 --0.2 ++0.3",
        "- x + y 0.5",
        "",
    ]
    cases = []
    for line in lines:
        result = subprocess.run(
            [str(executable)], input=line + "\n", text=True, capture_output=True, check=True
        )
        tokens = []
        for row in result.stdout.splitlines():
            kind, overflow, value, *text = row.split(maxsplit=3)
            tokens.append(
                dict(
                    kind=kind,
                    overflow=int(overflow),
                    value=float(value),
                    text=text[0].rstrip() if text else "",
                )
            )
        cases.append(dict(line=line, tokens=tokens))
    record = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        notes="Original QLEX extracted unchanged; IEEE binary64/single and 32-bit integer "
        "maximum machine constants. 212-character lines. Numeric value undefined when "
        "overflow=3. Integer sign bug retained.",
        compiler=subprocess.run(
            ["gfortran", "--version"], text=True, capture_output=True, check=True
        ).stdout.splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/multi_input.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
