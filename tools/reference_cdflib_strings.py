"""Record unchanged F95 case conversion and stateful lexical analysis."""

import hashlib
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use biomath_strings_mod
implicit none
integer mode,n,padding,width,i,j,lcval,ival,indov,calls
integer,allocatable :: codes(:)
real(8) dval
logical qstart,found
character(:),allocatable :: input,cval,lower,upper
character(2) curtyp
read(*,*) mode,n,padding,width
allocate(codes(n))
if(n>0) read(*,*) codes
allocate(character(n+padding)::input)
input=repeat(' ',n+padding)
do i=1,n
 input(i:i)=achar(codes(i))
end do
if(mode==1) then
 do i=1,n
  write(*,'(3(I0,1X))') codes(i), &
   iachar(lower_case_char(input(i:i))),iachar(upper_case_char(input(i:i)))
 end do
else if(mode==2) then
 lower=input
 upper=input
 call lower_case_string(lower)
 call upper_case_string(upper)
 write(*,'(*(I0,1X))') (iachar(lower(i:i)),i=1,len(lower))
 write(*,'(*(I0,1X))') (iachar(upper(i:i)),i=1,len(upper))
else
 allocate(character(width)::cval)
 qstart=.true.
 do calls=1,10000
  found=qlex(input,qstart,curtyp,cval,lcval,ival,dval,indov)
  if(.not.found) then
   write(*,'(A,1X,L1)') 'END',qstart
   stop
  end if
  write(*,'(A,1X,A,1X,3(I0,1X),ES26.17E3,1X,L1)') 'TOKEN',curtyp,lcval,ival,indov,dval,qstart
  write(*,'(*(I0,1X))') (iachar(cval(j:j)),j=1,min(lcval,len(cval)))
 end do
 write(*,'(A)') 'DRIVER_TOKEN_LIMIT'
end if
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    paths = [source / f"{name}.f90" for name in ("biomath_constants_mod", "biomath_strings_mod")]
    work = Path("research/raw/reference/cdflib-strings").resolve()
    work.mkdir(parents=True, exist_ok=True)
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    exe = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, paths),
        str(driver),
        "-o",
        str(exe),
    ]
    subprocess.run(command, cwd=work, check=True)

    def run(mode, text, *, padding=2, width=None):
        codes = list(text) if isinstance(text, bytes) else list(map(ord, text))
        width = max(1, len(codes) + padding) if width is None else width
        payload = f"{mode} {len(codes)} {padding} {width}\n" + " ".join(map(str, codes)) + "\n"
        row = dict(mode=mode, codes=codes, padding=padding, width=width)
        try:
            r = subprocess.run(
                [str(exe)], input=payload, text=True, capture_output=True, timeout=3, check=True
            )
        except subprocess.TimeoutExpired:
            row.update(execution_outcome="timeout", timeout_seconds=3)
        except subprocess.CalledProcessError as error:
            row.update(
                execution_outcome="process_error",
                exit_code=error.returncode,
                stdout=error.stdout,
                stderr=error.stderr,
            )
        else:
            row["execution_outcome"] = "completed"
            lines = r.stdout.splitlines()
            if mode == 1:
                row["result"] = [list(map(int, line.split())) for line in lines]
            elif mode == 2:
                row["result"] = [list(map(int, line.split())) for line in lines]
            else:
                tokens = []
                while lines and lines[0].startswith("TOKEN "):
                    fields = lines.pop(0).split()
                    value = float(fields[5])
                    token = dict(
                        kind=fields[1],
                        length=int(fields[2]),
                        integer=int(fields[3]),
                        overflow=int(fields[4]),
                        real=value if math.isfinite(value) else str(value),
                        qstart=fields[6] == "T",
                        codes=list(map(int, lines.pop(0).split())),
                    )
                    tokens.append(token)
                row.update(tokens=tokens, termination=lines)
        return row

    conversion = [run(1, bytes(range(256)), padding=0)]
    for text in [b"", b"AbC xyz  ", bytes(range(256)), b"AZ\taz\0 Qq\r\n", b"LongABC" * 100]:
        conversion.append(run(2, text, padding=0))
    ordinary = [
        "",
        "   ",
        "0 1 -2 +3",
        "1.25 -0.5 .25 5. 1e3 1D-3",
        "abc ABC a1 a.b a_b a$b",
        "()[]{},:;",
        "+ - * / < > = <= >= == **",
        "a+2 a-2 a + 2 a - 2",
        "(-2) [+3] x=-4",
        "a+.5 a + .5",
        "'hello' \"world\"",
        "'' \"\"",
        "'don''t'",
        '"a""b"',
        "'a b' 'a\tb'",
        "'unterminated",
        "123abc . _ $",
        "1e 1e+ 1e- 1.2.3",
        "abc!def @#$% \\ | ~",
        "\t\r\n",
        "1,2;3:4",
        "+   12 -   .25",
        "a\0b",
    ]
    numeric = [
        "2147483647 2147483648 -2147483648 -2147483649",
        "9007199254740991 9007199254740992 9007199254740993",
        "1e38 1e39 1e308 1e309",
        "1e-300 1e-320 1e-324",
        "0e999 0e-999",
        "1e2147483647",
        "1e-1000000000",
        "9" * 310,
        "0." + "1" * 350,
    ]
    lexical = [dict(label=f"ordinary-{i}", **run(3, t)) for i, t in enumerate(ordinary)]
    lexical += [dict(label=f"numeric-{i}", **run(3, t)) for i, t in enumerate(numeric)]
    for text in ["", "abc", "123", "'x'", "(", "  "]:
        lexical.append(dict(label="exact-buffer", **run(3, text, padding=0)))
    for text in [bytes([128]), bytes([255]), b"'\xff'", b"abc"]:
        lexical.append(dict(label="buffer-or-byte-bound", **run(3, text, width=1)))
    lexical += [dict(label=f"ascii-{i}", **run(3, bytes([i]))) for i in range(128)]
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    result = dict(
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        command=command,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        driver=DRIVER,
        adaptations=[],
        conversion=conversion,
        lexical=lexical,
    )
    Path("tests/fixtures/cdflib_strings.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        {
            out: sum(r["execution_outcome"] == out for r in lexical)
            for out in ["completed", "process_error", "timeout"]
        }
    )


if __name__ == "__main__":
    main()
