"""Reproducible batch/scalar comparisons; timings are evidence, not test gates."""

import json
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy

from mdanderson_stats import chi_square_gof, normal_tails


def measure(batch: Callable, scalar: Callable, workload: str) -> dict:
    batch_times, scalar_times = [], []
    for _ in range(3):
        start = time.perf_counter()
        batch_result = batch()
        batch_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        scalar_result = scalar()
        scalar_times.append(time.perf_counter() - start)
        np.testing.assert_allclose(batch_result, scalar_result, rtol=1e-13, atol=1e-15)
    return {
        "workload": workload,
        "batch_seconds": statistics.median(batch_times),
        "scalar_seconds": statistics.median(scalar_times),
        "speedup": statistics.median(scalar_times) / statistics.median(batch_times),
    }


def main() -> None:
    x = np.linspace(-100, 100, 10000)
    observations = np.arange(20000).reshape(1000, 20) % 29
    weights = np.arange(1, 21)
    normal = measure(
        lambda: np.stack(normal_tails(x, log=True), axis=-1),
        lambda: np.array([normal_tails(float(v), log=True) for v in x]),
        "10000 pairs of log normal tails, x evenly spaced from -100 to 100",
    )
    gof = measure(
        lambda: chi_square_gof(observations, weights).pvalue,
        lambda: np.array([chi_square_gof(row, weights).pvalue for row in observations]),
        "1000 Pearson goodness-of-fit tests, 20 categories, "
        "deterministic unequal counts and weights",
    )
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "repetitions": 3,
        "comparison": "Same Python API called once on arrays versus once per scalar or row; "
        "not a comparison against the archived executables",
        "results": {"normal_tails": normal, "chi_square_gof": gof},
    }
    Path("docs/numerical-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
