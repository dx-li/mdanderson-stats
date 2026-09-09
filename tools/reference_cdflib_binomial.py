"""Generate original CDFLIB90 binomial references with a separate driver."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use cdf_binomial_mod
implicit none
integer which,status
real(8) p,q,s,n,pr,cpr
read(*,*) which,p,q,s,n,pr,cpr
call cdf_binomial(which,cum=p,ccum=q,s=s,n=n,pr=pr,cpr=cpr,status=status)
write(*,'(I5,6ES26.17E3)') status,p,q,s,n,pr,cpr
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
        "cdf_binomial_mod",
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
    for fraction, trials, pr in itertools.product(
        (0.0, 0.2, 0.5), (1.0, 5.0, 20.0), (0.1, 0.5, 0.9)
    ):
        successes = fraction * trials
        inputs = [1, 0.0, 0.0, successes, trials, pr, 1 - pr]
        status, result = run(inputs)
        cases.append({"input": inputs, "status": status, "result": result})
        p, q = result[:2]
        if min(p, q) < 1e-8:
            continue
        for which in (2, 3, 4):
            inputs = [which, p, q, successes, trials, pr, 1 - pr]
            native_input = inputs.copy()
            if which == 4:
                native_input[-2:] = [0.123, 0.877]
            status, result = run(native_input)
            cases.append(
                {"input": inputs, "native_input": native_input, "status": status, "result": result}
            )
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    fixture = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command,
        "driver": DRIVER,
        "adaptations": [],
        "input_order": ["which", "cum", "ccum", "s", "n", "pr", "cpr"],
        "result_order": ["cum", "ccum", "s", "n", "pr", "cpr"],
        "cases": cases,
    }
    backup_source = source / "#cdf_binomial_mod.f90#"
    backup_work = work / "backup"
    backup_work.mkdir(exist_ok=True)
    renamed = backup_work / "cdf_binomial_mod.f90"
    renamed.write_bytes(backup_source.read_bytes())
    backup_paths = paths[:-1] + [renamed]
    executable = backup_work / "reference"
    backup_command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, backup_paths),
        str(driver),
        "-o",
        str(executable),
    ]
    subprocess.run(backup_command, cwd=backup_work, check=True)
    backup_cases = []
    for case in cases:
        native_input = case.get("native_input", case["input"])
        status, result = run(native_input)
        backup_cases.append(
            {
                "input": case["input"],
                "native_input": native_input,
                "status": status,
                "result": result,
            }
        )
    fixture["backup_reference"] = {
        "archived_source": str(backup_source),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in backup_paths},
        "command": backup_command,
        "adaptations": [
            "Backup renamed to a .f90 filename for compilation; source bytes unchanged"
        ],
        "cases": backup_cases,
    }
    target = Path("tests/fixtures/cdflib_binomial.json")
    target.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    print(
        f"Recorded {len(cases)} native binomial cases; "
        f"{sum(c['status'] != 0 for c in cases)} nonzero statuses"
    )


if __name__ == "__main__":
    main()
