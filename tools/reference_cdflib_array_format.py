"""Compile native console formatting and output-routing contracts."""

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
character(80) :: mode,fmt,flag
real(dpkind) :: values(5)
call get_command_argument(1,mode)
call get_command_argument(2,fmt)
call get_command_argument(3,flag)
report_unit=0
always_print=.true.
print_off=.false.
num_subs=0
if(trim(mode)=='array') then
read(*,*) values
call write_array(values,fmt)
else
message_format=fmt
open(unit=42,file='report.txt',status='replace')
select case(trim(flag))
case('unit')
call print_message_format(unit=42)
case('unit_false')
call print_message_format(unit=42,unit_only=.false.)
case('unit_true')
call print_message_format(unit=42,unit_only=.true.)
case('off')
print_off=.true.
call print_message_format()
case('force')
print_off=.true.
call print_message_format(force=.true.)
case default
call print_message_format()
end select
close(42)
end if
end program probe
"""

CASES = [
    ("message", "(1X,'Hello')", "default", ""),
    ("message", "(1X,'Hello')", "unit", ""),
    ("message", "(1X,'Hello')", "unit_false", ""),
    ("message", "(1X,'Hello')", "unit_true", ""),
    ("message", "(1X,'Hello')", "off", ""),
    ("message", "(1X,'Hello')", "force", ""),
    ("array", "5F11.6", "", "0 1 1e-4 1e4 -1\n"),
    ("array", "5F11.3", "", "0 1 1e-4 1e4 -1\n"),
    ("array", "5F11.6", "", "1e-3 1e3 5e-324 1e308 -1e308\n"),
    ("array", "5E11.3", "", "0 1 1e-4 1e4 -1\n"),
    ("array", "5F18.6", "", "0 1 1e-4 1e4 -1\n"),
    ("array", "5f11.6", "", "0 1 1e-4 1e4 -1\n"),
    ("array", "5F11.0", "", "0 1e-4 5e4 9e4 -9e4\n"),
    ("array", "5F11.1", "", "0 1e-4 5e4 9e4 -9e4\n"),
    ("array", "5F11.6", "", "-0 -1e-4 0.0010000000474974513 1000.0000001 999.9999999\n"),
    ("array", "5F11.1", "", "1.25 1.75 -1.25 -1.75 9.95\n"),
    ("array", "5F11.0", "", "1 2 -1 0.4 -0.4\n"),
    ("array", "5F10.6", "", "999.9999999 -999.9999999 0.1 -0.1 1000\n"),
    ("array", "2F11.6 3X 3F11.6", "", "1 2 3 4 5\n"),
    ("array", "5F11.9", "", "0.5 -0.5 1.25 -1.25 0.0011\n"),
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
    with tempfile.TemporaryDirectory(prefix="cdflib-format-") as directory:
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
        for mode, fmt, flag, data in CASES:
            run = subprocess.run(
                [str(work / "reference"), mode, fmt, flag],
                cwd=work,
                input=data,
                text=True,
                capture_output=True,
                timeout=3,
            )
            report = (work / "report.txt").read_text() if mode == "message" else None
            records.append(
                dict(
                    mode=mode,
                    format=fmt,
                    flag=flag,
                    input=data,
                    returncode=run.returncode,
                    stdout=run.stdout,
                    stderr=re.sub(
                        r"0x[0-9a-fA-F]+", "<address>", run.stderr.replace(str(work), "<build>")
                    ),
                    report=report,
                )
            )
    result = dict(
        archive_sha256=ARCHIVE_SHA256,
        source_hashes={n: hashlib.sha256(r).hexdigest() for n, r in sources.items()},
        driver_sha256=hashlib.sha256(DRIVER.encode()).hexdigest(),
        flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=records,
    )
    Path("tests/fixtures/cdflib_array_format.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Recorded {len(records)} formatting cases")


if __name__ == "__main__":
    main()
