"""Record S/desktop nonparametric outputs and diagnose numerical failures."""

import hashlib
import json
import subprocess
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from reference_multi import SOURCE, build, evaluate
from reference_numerics import RAW


def precise_derivative(values, n, center, width, start, count):
    """Independently solve the uncentered normal equations at 70-digit precision.

    Window bounds and width come from the Fortran trace, not the Python port.
    A common weight normalization factor cancels from the normal equations.
    """
    with localcontext() as context:
        context.prec = 70
        x = [Decimal.from_float(float(v)) for v in values[start : start + count]]
        y = [Decimal.from_float(float((i + 1) / n)) for i in range(start, start + count)]
        c, h = Decimal.from_float(float(center)), Decimal.from_float(float(width))
        weights = [1 / (1 - ((v - c) / h) ** 2) ** 2 for v in x]
        powers = [[Decimal(1), v, v * v, v**3, v**4] for v in x]
        matrix = [
            [sum(w * v[i + j] for w, v in zip(weights, powers, strict=True)) for j in range(3)]
            for i in range(3)
        ]
        rhs = [
            sum(w * v[i] * z for w, v, z in zip(weights, powers, y, strict=True)) for i in range(3)
        ]
        for i in range(3):
            pivot = max(range(i, 3), key=lambda j: abs(matrix[j][i]))
            matrix[i], matrix[pivot] = matrix[pivot], matrix[i]
            rhs[i], rhs[pivot] = rhs[pivot], rhs[i]
            for j in range(i + 1, 3):
                factor = matrix[j][i] / matrix[i][i]
                for k in range(i, 3):
                    matrix[j][k] -= factor * matrix[i][k]
                rhs[j] -= factor * rhs[i]
        beta = [Decimal(0)] * 3
        for i in range(2, -1, -1):
            beta[i] = (rhs[i] - sum(matrix[i][j] * beta[j] for j in range(i + 1, 3))) / matrix[i][i]
        return float(beta[1] + 2 * beta[2] * c)


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
    desktop_cases = []
    for name in ("published", "grid40", "endpoints"):
        values = list(families[name])
        for fitted in sorted({0, 1, 2, 5, 6, 10, 11, 25, len(values)}):
            if fitted > len(values):
                continue
            estimate = len(values) - fitted + 0.7
            for alpha in (0.05, 0.2):
                rows = evaluate(executable, 16, values, alpha, estimate).splitlines()
                length, status = map(int, rows[0].split())
                columns = [row.split() for row in rows[1:]]
                case = {
                    "name": name,
                    "pvalues": values,
                    "null_estimate": estimate,
                    "alpha": alpha,
                    "fitted_points": length,
                    "status": status,
                    "sorted_scores": [float(c[0]) for c in columns],
                    "sorted_density": [float(c[1]) for c in columns[:length]],
                    "sorted_reject": [bool(int(c[3])) for c in columns],
                }
                if length > 10 and status == 0:
                    ordered = sorted(values)
                    case["bandwidths"] = [float(c[2]) for c in columns[:length]]
                    case["high_precision_density"] = [
                        precise_derivative(
                            ordered, len(values), ordered[i], float(c[2]), int(c[4]) - 1, int(c[5])
                        )
                        for i, c in enumerate(columns[:length])
                    ]
                desktop_cases.append(case)
    output["desktop_cases"] = desktop_cases
    output["desktop_notes"] = (
        "Original NP1P with output sentinels initialized by the driver; "
        "TRACE_SMOOTH adds only a selected-width output to a copy of SMOOTH. "
        "High-precision densities solve uncentered reciprocal-weight normal equations "
        "at Decimal precision 70 using the Fortran-selected windows and widths. "
        "Nearly tied cross-validation objectives can choose different widths in "
        "floating-point arithmetic, as on the exact linear grids."
    )
    Path("tests/fixtures/nonparametric.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} S and {len(desktop_cases)} desktop nonparametric fits")


if __name__ == "__main__":
    main()
