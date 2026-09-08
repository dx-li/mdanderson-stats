"""Record original ONESAMPLE calculation-module tests and confidence bounds."""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


def main():
    source = Path("research/raw/ONESAMPLE/source/one_sample_1.0/source").resolve()
    original_source = source
    directory = Path("research/raw/reference/onesample").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "portable"
    source.mkdir(exist_ok=True)
    for path in original_source.iterdir():
        if path.suffix == ".f90" or path.name == "Makefile":
            shutil.copyfile(path, source / path.name)
    auxiliary = source / "cdf_aux_mod.f90"
    code = auxiliary.read_text()
    for name in (
        "add_to_one",
        "check_complements",
        "dbl_in_range",
        "int_in_range",
        "validate_parameters",
    ):
        match = re.search(r"(?:FUNCTION|SUBROUTINE) " + name + r"\(", code)
        if match is None:
            raise RuntimeError(name)
        start = code.index("! .. Executable", match.start())
        end = code.index("\n", start) + 1
        code = code[:end] + "\n        IF (PRESENT(status)) status = 0\n" + code[end:]
    auxiliary.write_text(code)
    subprocess.run(
        ["make", "F90=gfortran", "F90FLAGS=-O2 -std=legacy"],
        cwd=source,
        check=True,
        capture_output=True,
    )
    driver = directory / "reference.f90"
    driver.write_text("""program reference
use one_sample_calc_mod
implicit none
integer family,k,f
real(dpkind) null,time,level,estimate,lo,hi,greater,less
read(*,*) family,k,f,null,time,level
if(family==1) then
call binomial_conf(k,f,level,estimate,lo,hi)
call binomial_test(k,f,null,greater,less)
else
call poisson_conf(k,time,level,estimate,lo,hi)
call poisson_test(k,time,null,greater,less)
endif
write(*,'(*(ES27.17E3,1X))') estimate,lo,hi,less,greater
end program
""")
    executable = directory / "reference"
    objects = sorted(str(p) for p in source.glob("*.o") if p.name != "one_sample.o")
    subprocess.run(
        ["gfortran", "-O2", "-I", str(source), str(driver), *objects, "-o", str(executable)],
        check=True,
        capture_output=True,
    )
    cases = []
    for confidence in (0.8, 0.95, 0.99):
        inputs = [
            ("binomial", k, 30 - k, p, 1)
            for k, p in [(0, 0.3), (1, 0.3), (12, 0.3), (30, 0.3), (1, 1e-12), (29, 1 - 1e-12)]
        ]
        inputs += [
            ("poisson", k, 0, p, t)
            for k, p, t in [
                (0, 2, 1),
                (1, 2, 1),
                (10, 2, 5),
                (10000, 2, 5000),
                (1, 1e-10, 1),
                (10, 1, 100),
            ]
        ]
        for distribution, k, f, null, time in inputs:
            text = f"{1 if distribution == 'binomial' else 2} {k} {f} {null} {time} {confidence}\n"
            result = subprocess.run(
                [str(executable)],
                input=text,
                text=True,
                capture_output=True,
                timeout=10,
                check=True,
            )
            values = list(map(float, result.stdout.split()))
            if len(values) != 5:
                raise RuntimeError(result.stdout)
            cases.append(
                dict(
                    distribution=distribution,
                    events=k,
                    failures=f,
                    null=null,
                    exposure=time,
                    confidence=confidence,
                    estimate=values[0],
                    bounds=values[1:3],
                    p_less=values[3],
                    p_greater=values[4],
                )
            )
    record = dict(
        archive_sha256=hashlib.sha256(
            Path("research/raw/ONESAMPLE/ONESAMPLE_V1.tar.gz").read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(
            (original_source / "one_sample_calc_mod.f90").read_bytes()
        ).hexdigest(),
        compiler=subprocess.run(
            ["gfortran", "--version"], text=True, capture_output=True, check=True
        ).stdout.splitlines()[0],
        notes="Unmodified one_sample_calc_mod, compiled -O2 -std=legacy. Reference-only "
        "cdf_aux_mod patch initializes optional status outputs to zero on entry to "
        "add_to_one, check_complements, dbl_in_range, int_in_range and validate_parameters; "
        "source success branches otherwise leave them undefined. Numerical algorithms "
        "and cutoffs unchanged.",
        cases=cases,
    )
    Path("tests/fixtures/onesample.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(f"{len(cases)} native cases")


if __name__ == "__main__":
    main()
