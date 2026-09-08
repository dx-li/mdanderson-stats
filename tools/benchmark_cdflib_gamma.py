"""Measure gamma batching against repeated calls to the same Python API."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdf_gamma


def main():
    cases = []
    for size in (128, 1024):
        x = np.geomspace(0.01, 100, size)
        rate = np.geomspace(10, 0.01, size)
        p = -np.expm1(-x * rate)
        q = np.exp(-x * rate)
        for operation in ("tails", "shape"):

            def evaluate(index):
                if operation == "tails":
                    return cdf_gamma(x=x[index], shape=1, rate=rate[index]).cum
                return cdf_gamma(3, x=x[index], cum=p[index], ccum=q[index], rate=rate[index]).shape

            batch_times, scalar_times = [], []
            for _ in range(3):
                start = perf_counter()
                batch = evaluate(slice(None))
                batch_times.append(perf_counter() - start)
                start = perf_counter()
                individual = np.array([float(evaluate(i)) for i in range(size)])
                scalar_times.append(perf_counter() - start)
                np.testing.assert_allclose(batch, individual, rtol=1e-11, atol=1e-15)
                np.testing.assert_allclose(
                    batch, p if operation == "tails" else 1, rtol=1e-10, atol=1e-15
                )
            batch_time, scalar_time = median(batch_times), median(scalar_times)
            cases.append(
                {
                    "operation": operation,
                    "size": size,
                    "batch_seconds": batch_time,
                    "scalar_seconds": scalar_time,
                    "speedup": scalar_time / batch_time,
                }
            )
    output = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "repetitions": 3,
        "inputs": (
            "x=geomspace(.01,100,n); rate=geomspace(10,.01,n); shape=1; "
            "p=-expm1(-x*rate); q=exp(-x*rate)"
        ),
        "comparison": (
            "One broadcast call versus n scalar calls to the same Python API; not native Fortran"
        ),
        "cases": cases,
    }
    Path("docs/cdflib-gamma-benchmark.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
