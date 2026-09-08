"""Build KSBIN1 XBIN1 reference brackets with explicit auxiliary-status repair."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def main():
    original = Path("research/raw/KSBIN1/source/ksbin190_1.0/source")
    directory = Path("research/raw/reference/binomial-power").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    for p in original.iterdir():
        if p.suffix == ".f90" or p.name == "Makefile":
            shutil.copyfile(p, directory / p.name)
    path = directory / "cdf_aux_mod.f90"
    code = path.read_text()
    helpers = [
        "add_to_one",
        "check_complements",
        "dbl_in_range",
        "int_in_range",
        "validate_parameters",
    ]
    for name in helpers:
        match = re.search(r"(?:FUNCTION|SUBROUTINE) " + name + r"\(", code)
        if match is None:
            raise RuntimeError(name)
        start = code.index("! .. Executable", match.start())
        end = code.index("\n", start) + 1
        code = code[:end] + "\n        IF (PRESENT(status)) status = 0\n" + code[end:]
    path.write_text(code)
    subprocess.run(
        ["make", "F90=gfortran", "F90FLAGS=-O2 -std=legacy"],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    driver = directory / "driver.f90"
    driver.write_text("""program reference
use solve_binomial_one_sample_mod
use biomath_constants_mod
implicit none
real(dpkind) n,p0,pa,alpha,c1,s1,w1,c2,s2,w2
read(*,*) n,p0,pa,alpha
s1=alpha
call xbin1(p0,pa,n,c1,s1,w1,c2,s2,w2)
write(*,'(*(ES27.17E3,1X))') c1,s1,w1,c2,s2,w2
end program
""")
    executable = directory / "reference"
    objects = [str(p) for p in directory.glob("*.o") if p.name != "ksbin1_main.o"]
    subprocess.run(
        ["gfortran", "-O2", str(driver), *objects, "-o", str(executable)],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    cases = []
    for n in [20, 50, 100]:
        for p0, pa in [(0.2, 0.06), (0.3, 0.6), (0.7, 0.4), (0.8, 0.94)]:
            for alpha in [0.01, 0.05, 0.1]:
                result = subprocess.run(
                    [str(executable)],
                    input=f"{n} {p0} {pa} {alpha}\n",
                    text=True,
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
                rows = [
                    line
                    for line in result.stdout.splitlines()
                    if re.fullmatch(r"[\s0-9Ee+.\-]+", line) and len(line.split()) == 6
                ]
                values = list(map(float, rows[-1].split())) if rows else None
                cases.append(
                    dict(
                        trials=n,
                        null=p0,
                        alternative=pa,
                        alpha=alpha,
                        bracket=values,
                        diagnostic=result.stdout if values is None else None,
                    )
                )
    fixture = dict(
        source="KSBIN1 XBIN1; unchanged statistical routines",
        status_patch={"helpers": helpers, "change": "Initialize optional status=0 at entry"},
        source_sha256=hashlib.sha256(
            (original / "solve_binomial_one_sample_mod.f90").read_bytes()
        ).hexdigest(),
        auxiliary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        cases=cases,
    )
    Path("tests/fixtures/binomial_power.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(cases)} native brackets")


if __name__ == "__main__":
    main()
