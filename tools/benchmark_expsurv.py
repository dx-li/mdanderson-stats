"""EXPSURV NumPy recurrence and query timings, with numerical equality checks."""

import json
import platform
from pathlib import Path

import numpy as np
from benchmark_numerics import measure

from mdanderson_stats import exploratory_survival


def sequential_product(status: np.ndarray) -> np.ndarray:
    survival = 1.0
    result = []
    for i, event in enumerate(status):
        survival *= 1 - float(event) / (len(status) - i)
        result.append(survival)
    return np.array(result)


def main() -> None:
    time = np.arange(20000, dtype=float)
    status = (np.arange(time.size) % 4 != 0).astype(float)
    fit = exploratory_survival(time, status)
    query = np.linspace(-1, time[-1] + 1, 10000)
    results = {
        "sequential_km": measure(
            lambda: exploratory_survival(time, status, legacy=True).survival,
            lambda: sequential_product(status),
            "20000 sorted observations, 25% censored; "
            "full NumPy fit versus scalar Python KM recurrence",
        ),
        "survival_queries": measure(
            lambda: fit.at(query),
            lambda: np.array([fit.at(t) for t in query]),
            "10000 right-continuous queries on a precomputed 20000-observation curve",
        ),
    }
    report = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "repetitions": 3,
        "comparison": "Python scalar references, not the archived XLISP-STAT executable; "
        "no GUI timing claim",
        "results": results,
    }
    Path("docs/expsurv-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
