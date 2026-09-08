"""Compare batched intervals with repeated scalar calls to the same Python API."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import binomial_interval


def main() -> None:
    k = np.arange(10000) % 101
    n = np.full(10000, 100)
    vector, scalar = [], []
    for _ in range(3):
        start = time.perf_counter()
        a = binomial_interval(k, n)
        vector.append(time.perf_counter() - start)
        start = time.perf_counter()
        b = np.array([binomial_interval(int(x), 100) for x in k])
        scalar.append(time.perf_counter() - start)
        np.testing.assert_allclose(np.stack(a, axis=-1), b, rtol=1e-13, atol=1e-15)
    result = {
        "workload": "10000 Clopper-Pearson intervals, n=100, k cycles through 0..100",
        "repetitions": 3,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "vectorized_seconds": statistics.median(vector),
        "scalar_seconds": statistics.median(scalar),
        "speedup": statistics.median(scalar) / statistics.median(vector),
        "comparison": "same Python API called once on arrays versus 10000 scalar calls; "
        "not a legacy executable comparison",
    }
    Path("docs/interval-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
