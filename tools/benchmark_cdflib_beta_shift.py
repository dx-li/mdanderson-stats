"""Measure beta-shift batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import bup


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("ordinary", lambda x: bup(2, 3, x, n=100)),
            ("tiny_shapes", lambda x: bup(1e-309, 1e-309, x, n=100)),
            ("huge_center", lambda x: bup(x * 1e308, x * 1e308, 0.5, n=2**31 - 1)),
            ("unit_large_shift", lambda x: bup(1, 1, 1 - x * 1e-12, x * 1e-12, n=2**31 - 1)),
            ("tail_difference", lambda x: bup(2, 3, x, n=2**31 - 1)),
        ]:
            x = np.linspace(0.1, 0.9, size)
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
    Path("docs/cdflib-beta-shift-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
