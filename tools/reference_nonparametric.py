"""Record the original S NPFIT outputs and diagnose known numerical failures."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from reference_multi import SOURCE, build, evaluate
from reference_numerics import RAW


def main() -> None:
    executable = build()
    rng = np.random.default_rng(182)
    families = {
        "published": [float(v) for v in (SOURCE / "pvals.txt").read_text().split()],
        "tied": [0.01] * 10 + [0.1, 0.2, 0.3, 0.5, 0.8, 0.9],
        "grid20": np.arange(1, 21) / 21,
        "grid40": np.arange(1, 41) / 41,
        "quadratic": np.sqrt(np.arange(1, 31) / 31),
        "narrow": 0.75 + np.arange(1, 21) * 2.0**-30,
        "endpoints": [0, 0.1, 0.1, 0.3, 0.4, 0.6, 0.8, 1],
    }
    for n in (4, 10, 40, 100):
        families[f"random{n}"] = rng.uniform(size=n)
    cases = []
    for name, values in families.items():
        values = list(values)
        rows = evaluate(executable, 15, values).splitlines()
        columns = [row.split() for row in rows[1:]]
        cases.append(
            {
                "name": name,
                "pvalues": values,
                "bandwidth": float(rows[0]),
                "sorted_scores": [float(c[0]) for c in columns],
                "sorted_density": [float(c[1]) for c in columns],
                "regression_status": [int(c[2]) for c in columns],
                "window_starts_one_based": [int(c[3]) for c in columns],
                "window_counts": [int(c[4]) for c in columns],
            }
        )
    output = {
        "source_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/MULTI/MULTI_V1.tar.gz",
        "sha256": hashlib.sha256((RAW / "MULTI/MULTI_V1.tar.gz").read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "flags": ["-O2", "-std=legacy", "-fallow-argument-mismatch"],
        "notes": "S NPFIT and WDTHMX symbols renamed to distinguish desktop routines; "
        "CUMBIN extracted unchanged. Existing identical desktop smoothing helpers reused. "
        "Additional density/window diagnostics use the same original kernels. "
        "Narrow data expose unstable normal equations; random10 gives invalid beta shapes.",
        "cases": cases,
    }
    Path("tests/fixtures/nonparametric.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} original nonparametric fits")


if __name__ == "__main__":
    main()
