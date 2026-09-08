"""Measure beta batching against repeated calls to the same Python API."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdf_beta


def main():
    cases = []
    for size in (128, 1024):
        x = np.linspace(0.05, 0.9, size)
        a = np.geomspace(0.2, 20, size)
        p = np.exp(a * np.log(x))
        q = -np.expm1(a * np.log(x))
        for operation in ("tails", "shape_a"):

            def evaluate(index):
                if operation == "tails":
                    return cdf_beta(x=x[index], a=a[index], b=1).cum
                return cdf_beta(3, x=x[index], cum=p[index], ccum=q[index], b=1).a

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
                    batch, p if operation == "tails" else a, rtol=1e-10, atol=1e-15
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
        "inputs": "x=linspace(.05,.9,n); a=geomspace(.2,20,n); b=1; p=x**a; q=-expm1(a*log(x))",
        "comparison": (
            "One broadcast call versus n scalar calls to the same Python API; not native Fortran"
        ),
        "cases": cases,
    }
    Path("docs/cdflib-beta-benchmark.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
