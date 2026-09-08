"""Record original fixed-shape minim.S grids with the archived FGH/DRMNHB."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from reference_stukel_fit import build_native


def main():
    source = Path("research/raw/STUKEL/S/stukel")
    executable = build_native(scan=True)
    alpha1, alpha2 = [-0.2, 0.0, 0.2], [-0.3, 0.0, 0.3, 0.6]
    cases = []
    for dataset in ("beetles", "warsaw"):
        values = np.loadtxt(source / f"{dataset}.dat")
        x = np.column_stack((np.ones(len(values)), values[:, 0]))
        objectives, statuses = [], []
        for first in alpha1:
            objective_row, status_row = [], []
            for second in alpha2:
                data = f"{len(x)} 2 0\n"
                for array in (x.ravel(order="F"), values[:, 1], values[:, 2], [first, second]):
                    data += " ".join(map(str, array)) + "\n"
                result = subprocess.run(
                    [str(executable)], input=data, text=True, capture_output=True, check=True
                )
                rows = result.stdout.splitlines()
                status_row.append(int(rows[0]))
                objective_row.append(float(rows[1]))
            objectives.append(objective_row)
            statuses.append(status_row)
        cases.append(
            dict(
                dataset=dataset,
                x=values[:, :1].tolist(),
                successes=values[:, 1].tolist(),
                trials=values[:, 2].tolist(),
                alpha1=alpha1,
                alpha2=alpha2,
                objective=objectives,
                status=statuses,
            )
        )
    record = {
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("all.f", "dgay.f", "minim.S", "scan.stukel.S")
        },
        "compiler": subprocess.run(
            ["gfortran", "--version"], text=True, capture_output=True, check=True
        ).stdout.splitlines()[0],
        "notes": "Original FGH/DRMNHB, minim.S zero starts, +/-1000 beta bounds and default "
        "DIVSET tolerances, 500 function evaluations. Only machine constants replaced for "
        "IEEE binary64/32-bit integers. Statuses 3..6 indicate native convergence.",
        "cases": cases,
    }
    Path("tests/fixtures/stukel_scan.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print([(c["dataset"], c["status"]) for c in cases])


if __name__ == "__main__":
    main()
