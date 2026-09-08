"""Native Fortran 77/95 RANDLIB state transitions and antithetic draws."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    f77 = root / "source/randlib.f/src"
    f95 = root / "RANDLIB90/source/randlib90/source"
    work = Path("research/raw/reference/randlib-streams").resolve()
    work.mkdir(parents=True, exist_ok=True)
    body = """implicit none
integer a,b,g,k,i,z,x,y
read(*,*)a,b,g,k
call setall(a,b)
call setcgn(g)
do i=1,4
 call capture()
enddo
call setant(.true.)
call capture()
call capture()
call initgn(0)
call capture()
call initgn(1)
call capture()
call advnst(k)
call capture()
call initgn(-1)
call capture()
call setsd(123,456)
call capture()
call setant(.false.)
call capture()
call setcgn(1)
call capture()
call setall(9876,54321)
call capture()
contains
subroutine capture()
z=ignlgi()
call getsd(x,y)
write(*,'(3I16)')z,x,y
end subroutine
end program
"""
    files = [
        f77 / f"{name}.f"
        for name in [
            "advnst",
            "getcgn",
            "getsd",
            "ignlgi",
            "initgn",
            "inrgcm",
            "mltmod",
            "qrgnin",
            "setall",
            "setant",
            "setsd",
        ]
    ]
    flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
    cases = []
    sources = {}
    for language in ["f77", "f95"]:
        directory = work / language
        directory.mkdir(exist_ok=True)
        driver = directory / "driver.f90"
        prefix = "program reference\n"
        if language == "f95":
            prefix += """use ecuyer_cote_mod, only: ignlgi=>random_large_integer, &
setall=>set_all_seeds, setcgn=>set_current_generator, setant=>set_antithetic, &
initgn=>reinitialize_current_generator, advnst=>advance_state, &
setsd=>set_current_seed, getsd=>get_seeds
"""
            selected = [f95 / "ecuyer_cote_mod.f90"]
            text = body
        else:
            selected = files
            text = body.replace(
                "integer a,b,g,k,i,z,x,y", "integer a,b,g,k,i,z,x,y,ignlgi\nexternal ignlgi"
            )
        driver.write_text(prefix + text)
        exe = directory / "reference"
        subprocess.run(
            ["gfortran", *flags, *map(str, selected), str(driver), "-o", str(exe)],
            cwd=directory,
            check=True,
        )
        sources.update(
            {
                str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in selected
            }
        )
        for seed in [(1234567890, 123456789), (1, 1)]:
            for stream in [1, 2, 17, 32]:
                for exponent in [0, 5, 30, 80]:
                    output = subprocess.check_output(
                        [str(exe)], input=f"{seed[0]} {seed[1]} {stream} {exponent}\n", text=True
                    )
                    rows = [
                        list(map(int, line.split())) for line in output.splitlines() if line.strip()
                    ]
                    if len(rows) != 14:
                        raise RuntimeError("missing native state transitions")
                    cases.append(
                        {
                            "language": language,
                            "seed": seed,
                            "stream": stream,
                            "exponent": exponent,
                            "rows": rows,
                        }
                    )
    fixture = {
        "source_sha256": sources,
        "flags": flags,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "cases": cases,
    }
    Path("tests/fixtures/randlib_streams.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Generated {len(cases)} native stream/control cases")


if __name__ == "__main__":
    main()
