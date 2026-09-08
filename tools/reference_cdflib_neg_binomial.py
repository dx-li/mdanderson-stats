"""Generate original CDFLIB90 negative-binomial references with a separate driver."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use cdf_neg_binomial_mod
implicit none
integer which,status
real(8) p,q,f,s,pr,cpr
read(*,*) which,p,q,f,s,pr,cpr
call cdf_neg_binomial(which,cum=p,ccum=q,f=f,s=s,pr=pr,cpr=cpr,status=status)
write(*,'(I5,6ES26.17E3)') status,p,q,f,s,pr,cpr
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    work = Path("research/raw/reference/cdflib-neg-binomial").resolve()
    work.mkdir(parents=True, exist_ok=True)
    names = [
        "biomath_constants_mod",
        "biomath_strings_mod",
        "biomath_sort_mod",
        "biomath_interface_mod",
        "biomath_mathlib_mod",
        "zero_finder",
        "cdf_aux_mod",
        "cdf_beta_mod",
        "cdf_neg_binomial_mod",
    ]
    paths = [source / f"{name}.f90" for name in names]
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    executable = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, paths),
        str(driver),
        "-o",
        str(executable),
    ]
    subprocess.run(command, cwd=work, check=True)

    def run(values):
        completed = subprocess.run(
            [str(executable)],
            input=" ".join(map(str, values)) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        fields = completed.stdout.split()
        if len(fields) != 7:
            raise RuntimeError(f"unexpected native output: {completed.stdout}")
        return int(fields[0]), [float(x) for x in fields[1:]]

    cases = []
    for failures, successes, pr in itertools.product(
        (0.0, 0.5, 3.0, 5.0), (0.2, 1.0, 5.0), (0.1, 0.5, 0.9)
    ):
        inputs = [1, 0.0, 0.0, failures, successes, pr, 1 - pr]
        status, result = run(inputs)
        cases.append({"input": inputs, "status": status, "result": result})
        p, q = result[:2]
        if min(p, q) < 1e-8:
            continue
        for which in (2, 3, 4):
            inputs = [which, p, q, failures, successes, pr, 1 - pr]
            status, result = run(inputs)
            cases.append({"input": inputs, "status": status, "result": result})
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    fixture = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command,
        "driver": DRIVER,
        "adaptations": [],
        "input_order": ["which", "cum", "ccum", "f", "s", "pr", "cpr"],
        "result_order": ["cum", "ccum", "f", "s", "pr", "cpr"],
        "cases": cases,
    }
    target = Path("tests/fixtures/cdflib_neg_binomial.json")
    target.write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        f"Recorded {len(cases)} native negative-binomial cases; "
        f"{sum(c['status'] != 0 for c in cases)} nonzero statuses"
    )


if __name__ == "__main__":
    main()
