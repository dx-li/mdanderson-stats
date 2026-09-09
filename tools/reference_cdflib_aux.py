"""Compile unchanged CDFLIB auxiliary metadata and defined failure contracts."""

import hashlib
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

from audit_cdflib90 import ARCHIVE_SHA256, F95_ROOT


def main():
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("CDFLIB90 archive hash mismatch")
    modules = [
        "biomath_constants_mod",
        "biomath_strings_mod",
        "biomath_sort_mod",
        "biomath_interface_mod",
        "zero_finder",
        "cdf_aux_mod",
    ]
    sources = {}
    with tarfile.open(archive) as tar:
        for name in modules:
            member = tar.extractfile(F95_ROOT + "source/" + name + ".f90")
            if member is None:
                raise RuntimeError(f"Missing source {name}")
            sources[name] = member.read()
    names = re.findall(
        r"TYPE \(the_distribution\), PARAMETER :: (\w+)", sources["cdf_aux_mod"].decode()
    )
    code = [
        "program reference",
        "use cdf_aux_mod",
        "use, intrinsic :: ieee_arithmetic",
        "implicit none",
        "type(the_distribution) :: d",
        "integer :: i, status",
        "real(dpkind) :: p(6), v",
        "logical :: ok",
    ]
    for name in names:
        code += [
            f"d = {name}",
            "write(*,'(A)') trim(d%name)",
            "write(*,*) d%max_which, d%nparam",
            "do i=1,6",
            "write(*,'(A)') trim(d%parameters(i)%name)",
            "write(*,*) d%parameters(i)%no_check",
            "write(*,'(2ES26.17E3)') d%parameters(i)%low_bound, d%parameters(i)%high_bound",
            "end do",
            "p = 0.0_dpkind",
            "call validate_parameters(d,0,p,status)",
            "write(*,*) status",
            "call validate_parameters(d,2,p,status)",
            "write(*,*) status",
        ]
    # Logical outcomes are defined on success; optional INTENT(OUT) status is not.
    for expression in (
        "-1.0_dpkind",
        "0.0_dpkind",
        "0.5_dpkind",
        "1.0_dpkind",
        "2.0_dpkind",
        "ieee_value(0.0_dpkind,ieee_quiet_nan)",
    ):
        code += [
            f"v={expression}",
            "ok=dbl_in_range(v,zero,one,'audit','v',-7,status)",
            "write(*,*) ok",
            "if (.not.ok) write(*,*) status",
        ]
    for x, y in [(0, 1), (0.25, 0.75), (0, 0), (1, 1), (-1, 2)]:
        code += [
            f"ok=add_to_one({float(x)!r}_dpkind,{float(y)!r}_dpkind,'audit','x','y',3,status)",
            "write(*,*) ok",
            "if (.not.ok) write(*,*) status",
        ]
    code += ["end program reference"]
    driver = "\n".join(code) + "\n"
    flags = ["-O0", "-ffp-contract=off", "-fcheck=all"]
    with tempfile.TemporaryDirectory(prefix="cdflib-aux-") as directory:
        work = Path(directory)
        for name, raw in sources.items():
            (work / (name + ".f90")).write_bytes(raw)
        (work / "driver.f90").write_text(driver)
        subprocess.run(
            [
                "gfortran",
                *flags,
                *(name + ".f90" for name in modules),
                "driver.f90",
                "-o",
                "reference",
            ],
            cwd=work,
            check=True,
            timeout=60,
        )
        output = iter(
            subprocess.check_output([str(work / "reference")], text=True, timeout=5).splitlines()
        )
    records = {}
    for name in names:
        native_name = next(output).strip()
        max_which, nparam = map(int, next(output).split())
        parameters = []
        for _ in range(6):
            label = next(output).strip()
            no_check = int(next(output))
            low, high = map(float, next(output).split())
            parameters.append(dict(name=label, no_check=no_check, low_bound=low, high_bound=high))
        records[name] = dict(
            name=native_name,
            max_which=max_which,
            nparam=nparam,
            parameters=parameters,
            invalid_which_status=int(next(output)),
            zero_pair_status=int(next(output)),
        )
    ranges = []
    for value in [-1, 0, 0.5, 1, 2, "NaN"]:
        ok = next(output).strip() == "T"
        ranges.append(dict(value=value, ok=ok, status=None if ok else int(next(output))))
    complements = []
    for x, y in [(0, 1), (0.25, 0.75), (0, 0), (1, 1), (-1, 2)]:
        ok = next(output).strip() == "T"
        complements.append(dict(x=x, y=y, ok=ok, status=None if ok else int(next(output))))
    if list(output):
        raise RuntimeError("Unexpected auxiliary driver output")
    result = dict(
        archive_sha256=ARCHIVE_SHA256,
        source_hashes={n: hashlib.sha256(r).hexdigest() for n, r in sources.items()},
        driver_sha256=hashlib.sha256(driver.encode()).hexdigest(),
        flags=flags,
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        distributions=records,
        ranges=ranges,
        complements=complements,
    )
    Path("tests/fixtures/cdflib_aux.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Recorded {len(records)} distributions and {len(ranges) + len(complements)} helper cases"
    )


if __name__ == "__main__":
    main()
