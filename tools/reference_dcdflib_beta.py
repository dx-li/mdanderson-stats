"""Compile both archived legacy languages and record their beta contracts."""

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

C_DRIVER = r"""#include <stdio.h>
#include "cdflib.h"
int main(void) {
 int which,status=-999; double p,q,x,y,a,b,bound=0;
 if(scanf("%d %lf %lf %lf %lf %lf %lf",&which,&p,&q,&x,&y,&a,&b)!=7) return 1;
 cdfbet(&which,&p,&q,&x,&y,&a,&b,&status,&bound);
 printf("%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",status,p,q,x,y,a,b,bound);
 return 0;
}
"""
F_DRIVER = """program reference
implicit none
integer which,status
real(8) p,q,x,y,a,b,bound
read(*,*) which,p,q,x,y,a,b
bound=0
status=-999
call cdfbet(which,p,q,x,y,a,b,status,bound)
write(*,'(I5,7ES26.17E3)') status,p,q,x,y,a,b,bound
end program reference
"""


def main():
    root = Path("research/raw/CDFLIB90/source/source").resolve()
    profiles = {}
    for language in ("c", "fortran"):
        work = Path("research/raw/reference/dcdflib-beta", language).resolve()
        work.mkdir(parents=True, exist_ok=True)
        exe = work / "reference"
        if language == "c":
            source = root / "dcdflib.c/src"
            paths = [source / "dcdflib.c", source / "ipmpar.c"]
            header = source / "cdflib.h"
            driver = work / "driver.c"
            driver.write_text(C_DRIVER)
            command = [
                "cc",
                "-std=gnu89",
                "-O0",
                "-ffp-contract=off",
                "-I",
                str(source),
                *map(str, paths),
                str(driver),
                "-lm",
                "-o",
                str(exe),
            ]
            version = subprocess.check_output(["cc", "--version"], text=True).splitlines()[0]
            hashes = paths + [header]
        else:
            paths = sorted((root / "dcdflib.f/src").glob("*.f"))
            driver = work / "driver.f90"
            driver.write_text(F_DRIVER)
            command = [
                "gfortran",
                "-std=legacy",
                "-O0",
                "-ffp-contract=off",
                "-fcheck=all",
                *map(str, paths),
                str(driver),
                "-o",
                str(exe),
            ]
            version = subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]
            hashes = paths
        subprocess.run(command, cwd=work, check=True)
        profiles[language] = {
            "command": command,
            "compiler": version,
            "source_sha256": {
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes
            },
            "driver": driver.read_text(),
            "adaptations": [],
            "cases": [],
        }

    def run(language, values, *, timeout=30):
        result = subprocess.run(
            [profiles[language]["command"][-1]],
            input=" ".join(map(str, values)) + "\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        fields = result.stdout.split()
        if len(fields) != 8:
            raise RuntimeError(f"unexpected native output: {result.stdout}")
        values = [float(v) for v in fields[1:]]
        return int(fields[0]), [v if math.isfinite(v) else str(v) for v in values]

    def record(language, inputs, *, timeout=30):
        try:
            status, result = run(language, inputs, timeout=timeout)
        except subprocess.TimeoutExpired:
            return dict(
                input=inputs,
                status=None,
                result=None,
                execution_outcome="timeout",
                timeout_seconds=timeout,
            )
        except subprocess.CalledProcessError as error:
            return dict(
                input=inputs,
                status=None,
                result=None,
                execution_outcome="process_error",
                exit_code=error.returncode,
                stdout=error.stdout,
                stderr=error.stderr,
            )
        return dict(input=inputs, status=status, result=result, execution_outcome="completed")

    for a, b, x in itertools.product(
        (0.5, 1.0, 2.0, 10.0), (0.5, 1.0, 2.0, 10.0), (0.01, 0.2, 0.5, 0.9)
    ):
        inputs = [1, 0.0, 0.0, x, 1 - x, a, b]
        for language in profiles:
            profiles[language]["cases"].append(record(language, inputs))
        p, q = profiles["fortran"]["cases"][-1]["result"][:2]
        for which in (2, 3, 4):
            inputs = [which, p, q, x, 1 - x, a, b]
            for language in profiles:
                profiles[language]["cases"].append(record(language, inputs))
    for language in profiles:
        profiles[language]["invalid_cases"] = [
            record(language, inputs)
            for inputs in (
                [0, 0.5, 0.5, 0.5, 0.5, 1, 1],
                [5, 0.5, 0.5, 0.5, 0.5, 1, 1],
                [2, -0.1, 1.1, 0.5, 0.5, 1, 1],
                [2, 0.5, -0.1, 0.5, 0.5, 1, 1],
                [1, 0, 0, -0.1, 1.1, 1, 1],
                [1, 0, 0, 0.5, -0.1, 1, 1],
                [1, 0, 0, 0.5, 0.5, 0, 1],
                [1, 0, 0, 0.5, 0.5, 1, 0],
                [2, 0.4, 0.4, 0.5, 0.5, 1, 1],
                [1, 0, 0, 0.4, 0.4, 1, 1],
            )
        ]
        profiles[language]["boundary_cases"] = [
            record(language, inputs)
            for inputs in (
                [1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 1, 1],
                [2, 0, 1, 0.5, 0.5, 1, 1],
                [2, 1, 0, 0.5, 0.5, 1, 1],
                [3, 0, 1, 0, 1, 1, 1],
                [4, 1, 0, 1, 0, 1, 1],
                [3, 0, 1, 0.5, 0.5, 1, 1],
                [4, 1, 0, 0.5, 0.5, 1, 1],
            )
        ]
        profiles[language]["wide_cases"] = [
            record(language, inputs, timeout=3)
            for inputs in (
                [1, 0, 0, 0.5, 0.5, 1e308, 1e308],
                [1, 0, 0, 0.5, 0.5, 1e200, 1e200],
                [1, 0, 0, 0.5, 0.5, 1e-308, 1],
                [1, 0, 0, 0.5, 0.5, 1, 1e-308],
                [1, 0, 0, 0.2, 0.8, 1e-308, 2e-308],
                [1, 0, 0, 1e-200, 1, 1, 1e200],
                [2, 1e-100, 1, 0.5, 0.5, 1, 1],
                [2, 1, 1e-100, 0.5, 0.5, 1, 1],
                [3, 1, 1e-100, 0.5, 0.5, 1, 1],
                [4, 1e-100, 1, 0.5, 0.5, 1, 1],
                [3, 0.5, 0.5, 0.5, 0.5, 1, 1e90],
            )
        ]
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    result = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "input_order": ["which", "p", "q", "x", "cx", "a", "b"],
        "result_order": ["p", "q", "x", "cx", "a", "b", "bound"],
        "status_policy": "Driver initializes status to -999.",
        "bound_policy": "Driver initializes bound to zero; undefined on native success.",
        "profiles": profiles,
    }
    Path("tests/fixtures/dcdflib_beta.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print({name: len(p["cases"]) for name, p in profiles.items()})


if __name__ == "__main__":
    main()
