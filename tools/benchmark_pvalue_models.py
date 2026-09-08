"""Compare CLUSTP's structured covariance simulation to a dense matrix method."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from mdanderson_stats import clustered_pvalues


def main() -> None:
    rows, k, rho, seed = 10000, 200, 0.4, 241
    # Precompute the same symmetric covariance root outside the dense timing.
    root = np.full((k, k), (np.sqrt(1 + (k - 1) * rho) - np.sqrt(1 - rho)) / k)
    root[np.diag_indices(k)] += np.sqrt(1 - rho)
    structured_times, dense_times = [], []
    for _ in range(5):
        start = time.perf_counter()
        structured = clustered_pvalues(rows, cluster_size=k, correlation=rho, rng=seed)
        structured_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        z = np.random.default_rng(seed).standard_normal((rows, k)) @ root
        dense = ndtr(-z)
        dense_times.append(time.perf_counter() - start)
        np.testing.assert_allclose(structured, dense, rtol=5e-13, atol=2e-15)
    result = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "workload": {"clusters": rows, "cluster_size": k, "correlation": rho, "seed": seed},
        "repetitions": 5,
        "structured_seconds": statistics.median(structured_times),
        "dense_seconds": statistics.median(dense_times),
        "speedup": statistics.median(dense_times) / statistics.median(structured_times),
        "comparison": "Same normal draws and covariance root; dense root precomputed; "
        "both include random generation and CDF conversion. Not original Fortran throughput.",
    }
    Path("docs/pvalue-models-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
