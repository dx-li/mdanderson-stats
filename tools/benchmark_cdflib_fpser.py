"""Measure fpser batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import fpser


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("ordinary", lambda x: fpser(0.5, 1e-20, x)),
            ("tiny_shapes", lambda x: fpser(1e-20, 1e-40, x)),
            ("large_shape", lambda x: fpser(100, 1e-20, x)),
            ("unit_shape", lambda x: fpser(1, 1e-20, x)),
            ("subnormal", lambda x: fpser(0.99, 1e-15, x * 1e-308)),
            ("loose_tolerance", lambda x: fpser(0.5, 0.2, x, 2)),
        ]:
            x = np.linspace(0.1, 0.5, size)
            batch, scalar = [], []
            for _ in range(3):
                start = perf_counter()
                result = function(x)
                batch.append(perf_counter() - start)
                start = perf_counter()
                expected = np.array([float(function(v)) for v in x])
                scalar.append(perf_counter() - start)
                np.testing.assert_array_equal(result, expected)
            b, s = median(batch), median(scalar)
            cases.append(
                dict(operation=name, size=size, batch_seconds=b, scalar_seconds=s, speedup=s / b)
            )
    data = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        platform=platform.platform(),
        repetitions=3,
        comparison=(
            "One broadcast call versus repeated scalar calls to the same API; not native timing"
        ),
        cases=cases,
    )
    Path("docs/cdflib-fpser-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
