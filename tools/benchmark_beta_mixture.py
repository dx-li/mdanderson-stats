"""Measure batched mixture posteriors against repeated scalar calls."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import scipy

from mdanderson_stats import BetaMixture


def main() -> None:
    model = BetaMixture(0.5, [0.3, 0.2], [0.4, 2], [8, 3])
    x = np.random.default_rng(223).uniform(size=5000)
    batched_times, scalar_times = [], []
    for _ in range(3):
        start = time.perf_counter()
        batch = model.null_posterior(x)
        batched_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        scalar = np.array([model.null_posterior(value) for value in x])
        scalar_times.append(time.perf_counter() - start)
        np.testing.assert_allclose(batch, scalar, rtol=1e-14, atol=0)
    output = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "repetitions": 3,
        "workload": "5000 p-values, a uniform plus two-beta model, seed 223",
        "batch_seconds": statistics.median(batched_times),
        "scalar_seconds": statistics.median(scalar_times),
        "speedup": statistics.median(scalar_times) / statistics.median(batched_times),
        "comparison": "One vectorized posterior call versus 5000 scalar calls to the same "
        "Python API; not original Fortran throughput or an EM fitting benchmark.",
    }
    Path("docs/beta-mixture-benchmark.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
