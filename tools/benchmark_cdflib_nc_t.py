"""Measure noncentral t batching against repeated calls to the same Python API."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import cdf_nc_t


def main():
    cases = []
    for size in (32, 128):
        x = np.linspace(1, 5, size)
        nc = np.linspace(0.5, 2, size)
        reference = cdf_nc_t(t=x, df=5, pnonc=nc)
        p, q = reference.cum, reference.ccum
        for operation in ("tails", "pnonc", "negative_tails"):

            def evaluate(index):
                if operation == "tails":
                    return cdf_nc_t(t=x[index], df=5, pnonc=nc[index]).cum
                if operation == "negative_tails":
                    return cdf_nc_t(t=-x[index], df=10, pnonc=4).cum
                return cdf_nc_t(4, t=x[index], df=5, cum=p[index], ccum=q[index]).pnonc

            batch_times, scalar_times = [], []
            for _ in range(3):
                start = perf_counter()
                batch = evaluate(slice(None))
                batch_times.append(perf_counter() - start)
                start = perf_counter()
                individual = np.array([float(evaluate(i)) for i in range(size)])
                scalar_times.append(perf_counter() - start)
                np.testing.assert_allclose(batch, individual, rtol=1e-11, atol=1e-15)
                if operation != "negative_tails":
                    np.testing.assert_allclose(
                        batch, p if operation == "tails" else nc, rtol=1e-10, atol=1e-15
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
            "t=linspace(1,5,n); pnonc=linspace(.5,2,n); df=5; "
            "negative_tails: -t,df=10,nc=4; p/q from the forward distribution"
        ),
        "comparison": (
            "One broadcast call versus n scalar calls to the same Python API; not native Fortran"
        ),
        "cases": cases,
    }
    Path("docs/cdflib-nc-t-benchmark.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
