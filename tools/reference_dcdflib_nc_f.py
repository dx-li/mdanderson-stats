"""Compile both archived legacy languages and record their noncentral F contracts."""

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

C_DRIVER = r"""#include <stdio.h>
#include "cdflib.h"
int main(void) {
 int which,status; double p,q,f,dfn,dfd,nc,bound=0;
 if(scanf("%d %lf %lf %lf %lf %lf %lf",&which,&p,&q,&f,&dfn,&dfd,&nc)!=7) return 1;
 cdffnc(&which,&p,&q,&f,&dfn,&dfd,&nc,&status,&bound);
 printf("%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",status,p,q,f,dfn,dfd,nc,bound);
 return 0;
}
"""
F_DRIVER = """program reference
implicit none
integer which,status
real(8) p,q,f,dfn,dfd,nc,bound
read(*,*) which,p,q,f,dfn,dfd,nc
bound=0
call cdffnc(which,p,q,f,dfn,dfd,nc,status,bound)
write(*,'(I5,7ES26.17E3)') status,p,q,f,dfn,dfd,nc,bound
end program reference
"""


def main():
    root = Path("research/raw/CDFLIB90/source/source").resolve()
    profiles = {}
    for language in ("c", "fortran"):
        work = Path("research/raw/reference/dcdflib-nc-f", language).resolve()
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

    def run(language, values):
        result = subprocess.run(
            [profiles[language]["command"][-1]],
            input=" ".join(map(str, values)) + "\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        fields = result.stdout.split()
        if len(fields) != 8:
            raise RuntimeError(f"unexpected native output: {result.stdout}")
        values = [float(v) for v in fields[1:]]
        return int(fields[0]), [v if math.isfinite(v) else str(v) for v in values]

    for f, nn, dd, nc in itertools.product(
        (0.1, 1.0, 10.0), (0.5, 2.0, 10.0), (0.5, 2.0, 10.0), (0.0, 0.5, 4.0)
    ):
        inputs = [1, 0.0, 0.0, f, nn, dd, nc]
        for language in profiles:
            status, result = run(language, inputs)
            profiles[language]["cases"].append(
                {"input": inputs, "status": status, "result": result}
            )
        p, q = profiles["fortran"]["cases"][-1]["result"][:2]
        if min(p, q) < 1e-8:
            continue
        for which in (2, 3, 4, 5):
            if which == 5 and nc == 0:
                continue
            inputs = [which, p, q, f, nn, dd, nc]
            for language in profiles:
                status, result = run(language, inputs)
                profiles[language]["cases"].append(
                    {"input": inputs, "status": status, "result": result}
                )
    for language in profiles:
        profiles[language]["invalid_cases"] = []
        for inputs in (
            [0, 0.5, 0.5, 1, 2, 2, 1],
            [1, 0, 0, -1, 2, 2, 1],
            [1, 0, 0, 1, 0, 2, 1],
            [1, 0, 0, 1, 2, 0, 1],
            [1, 0, 0, 1, 2, 2, -1],
            [2, 1, 0, 1, 2, 2, 1],
            [2, -0.1, 0, 1, 2, 2, 1],
        ):
            status, result = run(language, inputs)
            profiles[language]["invalid_cases"].append(
                {"input": inputs, "status": status, "result": result}
            )
        profiles[language]["ignored_q_cases"] = []
        for q in (-123.0, 0.0, 0.8, 42.0):
            inputs = [2, 0.4, q, 1, 2, 2, 1]
            status, result = run(language, inputs)
            profiles[language]["ignored_q_cases"].append(
                {"input": inputs, "status": status, "result": result}
            )
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    result = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "input_order": ["which", "p", "q", "f", "dfn", "dfd", "pnonc"],
        "result_order": ["p", "q", "f", "dfn", "dfd", "pnonc", "bound"],
        "bound_policy": (
            "Driver initializes bound to zero; only defined by native contract on failure."
        ),
        "profiles": profiles,
    }
    Path("tests/fixtures/dcdflib_nc_f.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print({name: len(p["cases"]) for name, p in profiles.items()})


if __name__ == "__main__":
    main()
