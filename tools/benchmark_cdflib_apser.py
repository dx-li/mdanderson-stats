"""Measure apser batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import apser


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("ordinary", lambda x: apser(1e-20, 0.5, x)),
            ("small_shapes", lambda x: apser(1e-20, 0.1, x)),
            ("tiny_companion", lambda x: apser(5e-324, 1e-309, x)),
            ("large_companion", lambda x: apser(1e-309, 1e308, x * 1e-308)),
            ("unit_shape", lambda x: apser(1e-20, 1, x)),
            ("subnormal_coordinate", lambda x: apser(0.01, 0.5, x * 1e-308, 0.1)),
            ("loose_tolerance", lambda x: apser(0.01, 2, x, 0.1)),
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
    Path("docs/cdflib-apser-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
