"""Native noncentral chi-square evidence, retaining original and repaired profiles."""

import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path

DRIVER = """program reference
use cdf_nc_chisq_mod
implicit none
integer which,status
real(8) p,q,x,df,nc
read(*,*) which,p,q,x,df,nc
call cdf_nc_chisq(which,cum=p,ccum=q,x=x,df=df,pnonc=nc,status=status)
write(*,'(I5,5ES26.17E3)') status,p,q,x,df,nc
end program reference
"""


def main():
    source = Path("research/raw/CDFLIB90/source/CDFLIB90/source/cdflib90_1.2/source").resolve()
    names = [
        "biomath_constants_mod",
        "biomath_strings_mod",
        "biomath_sort_mod",
        "biomath_interface_mod",
        "biomath_mathlib_mod",
        "zero_finder",
        "cdf_aux_mod",
        "cdf_gamma_mod",
        "cdf_chisq_mod",
        "cdf_nc_chisq_mod",
    ]
    paths = [source / f"{name}.f90" for name in names]
    profiles = {}
    for profile in ["original", "central_status_repaired"]:
        work = Path("research/raw/reference/cdflib-nc-chisq", profile).resolve()
        work.mkdir(parents=True, exist_ok=True)
        compiled = []
        patches = []
        for path in paths:
            raw = path.read_bytes()
            if profile != "original" and path.stem == "cdf_chisq_mod":
                before = b"CALL cdf_finalize_status(local,status)"
                after = b"IF (which>1) CALL cdf_finalize_status(local,status)"
                assert raw.count(before) == 1
                raw = raw.replace(before, after)
                patches.append(
                    {
                        "file": path.name,
                        "before": before.decode(),
                        "after": after.decode(),
                        "compiled_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            target = work / path.name
            target.write_bytes(raw)
            compiled.append(target)
        driver = work / "driver.f90"
        driver.write_text(DRIVER)
        exe = work / "reference"
        command = [
            "gfortran",
            "-O0",
            "-ffp-contract=off",
            "-fcheck=all",
            *map(str, compiled),
            str(driver),
            "-o",
            str(exe),
        ]
        subprocess.run(command, cwd=work, check=True)
        profiles[profile] = {"command": command, "patches": patches, "cases": []}

    def run(profile, inputs):
        command = profiles[profile]["command"]
        output = subprocess.run(
            [command[-1]],
            input=" ".join(map(str, inputs)) + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        fields = output.split()
        if len(fields) != 6:
            raise RuntimeError(f"unexpected native output: {output}")
        values = [float(v) for v in fields[1:]]
        return int(fields[0]), [v if math.isfinite(v) else str(v) for v in values]

    for x, df, nc in itertools.product(
        (0.2, 2.0, 10.0, 30.0), (0.5, 2.0, 10.0), (0.0, 0.5, 4.0, 20.0)
    ):
        inputs = [1, 0.0, 0.0, x, df, nc]
        for profile in profiles:
            status, result = run(profile, inputs)
            profiles[profile]["cases"].append({"input": inputs, "status": status, "result": result})
        # Paired probabilities from the documented repaired native profile.
        p, q = profiles["central_status_repaired"]["cases"][-1]["result"][:2]
        if not isinstance(p, float) or not isinstance(q, float) or min(p, q) < 1e-6:
            continue
        for which in (2, 3, 4):
            if which == 4 and nc == 0:
                continue
            inputs = [which, p, q, x, df, nc]
            for profile in profiles:
                status, result = run(profile, inputs)
                profiles[profile]["cases"].append(
                    {"input": inputs, "status": status, "result": result}
                )
    archive = Path("research/raw/CDFLIB90/CDFLIB90  _V90.tar.gz")
    output = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "driver": DRIVER,
        "input_order": ["which", "cum", "ccum", "x", "df", "pnonc"],
        "result_order": ["cum", "ccum", "x", "df", "pnonc"],
        "profiles": profiles,
    }
    Path("tests/fixtures/cdflib_nc_chisq.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print({name: len(profile["cases"]) for name, profile in profiles.items()})


if __name__ == "__main__":
    main()
