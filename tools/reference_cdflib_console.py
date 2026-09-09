"""Record unchanged CDFLIB console routines with bounded input transcripts."""

import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256, F95_ROOT

DRIVER = """program probe
use biomath_constants_mod
use biomath_interface_mod
implicit none
character(20) :: mode
character(80) :: message(1), answer, fmt
real(dpkind) :: x, values(5)
real(spkind) :: sx, sv(3)
integer :: ix, iv(3)
integer :: choice, nlist
logical :: failed
call get_command_argument(1,mode)
report_unit=0
always_print=.true.
print_off=.false.
num_subs=0
select case(trim(mode))
case('number')
call get_numbers(x=x,lo=zero,hi=one,failed=failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,'(ES26.17E3)') x
case('double_array')
call get_numbers(x=values,number_wanted=3,lo=[zero],hi=[one],failed=failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,'(3ES26.17E3)') values(1:3)
case('real')
call get_numbers(x=sx,failed=failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,'(ES26.17E3)') sx
case('real_array')
call get_numbers(x=sv,failed=failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,'(3ES26.17E3)') sv
case('integer')
call get_numbers(x=ix,failed=failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,*) ix
case('integer_array')
call get_numbers(x=iv,failed=failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,*) iv
case('character')
call get_character("(1X,'choose')",'abc',3,choice,failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,*) choice
case('string')
message(1)="(1X,'text')"
call get_string(message,1,answer,.false.,failed)
write(*,*) 'RESULT ',failed
if (.not.failed) write(*,'(A)') trim(answer)
case('list')
values(1:3)=[3.0_dpkind,one,one]
nlist=3
call get_list_double(list=values,n_list=nlist,failed=failed)
write(*,*) 'RESULT ',failed,nlist
if (.not.failed) write(*,'(5ES26.17E3)') values(1:nlist)
case('array')
values=[zero,1.25_dpkind,0.0001_dpkind,10000.0_dpkind,-two]
fmt='5F11.6'
call write_array(values,fmt)
end select
end program probe
"""

CASES = [
    ("number", "0.5\n"),
    ("number", "nan\n"),
    ("number", "bad\n-1\n0.25\n"),
    ("number", ""),
    ("character", " B\n"),
    ("character", "x\nx\nx\n"),
    ("string", "# comment\nhello # tail\n"),
    ("string", ""),
    ("array", ""),
    ("list", "8\n"),
    ("list", "7\n8\n"),
    ("list", "5\n2\n8\n"),
    ("list", "6\n1 2\n8\n"),
    ("list", "2\n0 1 1\n8\n"),
    ("list", "3\n1 100 1\n8\n"),
    ("list", "1\n2\n4 5\n8\n"),
    ("list", "4\n8\n"),
    ("list", "4\n\n8\n"),
    ("list", "6\n1 3\n1\n2\n1e308 1.5e308\n7\n8\n"),
    ("double_array", "0.1,0.2,0.3\n"),
    ("double_array", "0.1\n0.2 0.3\n"),
    ("double_array", "3*0.5\n"),
    ("real", "0.1\n"),
    ("real", "1+2\n"),
    ("real", "1-2\n"),
    ("real_array", "0.1 1e-40 1e30\n"),
    ("integer", "2147483647\n"),
    ("integer", "2147483648\n-2147483648\n"),
    ("integer_array", "1 -2 3\n"),
    ("number", "1D-2\n"),
    ("number", "bad\nbad\nbad\n"),
]


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("CDFLIB90 archive hash mismatch")
    names = [
        "biomath_constants_mod",
        "biomath_strings_mod",
        "biomath_sort_mod",
        "biomath_interface_mod",
    ]
    sources = {}
    with tarfile.open(archive) as tar:
        for name in names:
            member = tar.extractfile(F95_ROOT + "source/" + name + ".f90")
            if member is None:
                raise RuntimeError(f"Missing source {name}")
            sources[name] = member.read()
    flags = ["-O0", "-ffp-contract=off", "-fcheck=all"]
    records = []
    with tempfile.TemporaryDirectory(prefix="cdflib-console-") as directory:
        work = Path(directory)
        for name, raw in sources.items():
            (work / (name + ".f90")).write_bytes(raw)
        (work / "driver.f90").write_text(DRIVER)
        subprocess.run(
            ["gfortran", *flags, *[n + ".f90" for n in names], "driver.f90", "-o", "reference"],
            cwd=work,
            check=True,
            timeout=60,
        )
        for mode, data in CASES:
            try:
                run = subprocess.run(
                    [str(work / "reference"), mode],
                    input=data,
                    text=True,
                    capture_output=True,
                    timeout=3,
                )
                records.append(
                    dict(
                        mode=mode,
                        input=data,
                        returncode=run.returncode,
                        stdout=run.stdout,
                        stderr=re.sub(
                            r"0x[0-9a-fA-F]+", "<address>", run.stderr.replace(str(work), "<build>")
                        ),
                    )
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"Console reference timed out: {mode}") from error
    report = dict(
        archive_sha256=ARCHIVE_SHA256,
        source_hashes={n: hashlib.sha256(r).hexdigest() for n, r in sources.items()},
        driver_sha256=hashlib.sha256(DRIVER.encode()).hexdigest(),
        flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=records,
    )
    Path("tests/fixtures/cdflib_console.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Recorded {len(records)} console transcripts")


if __name__ == "__main__":
    main()
