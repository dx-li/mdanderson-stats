"""Measure beta-factor batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import brcmp1, brcomp


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("ordinary", lambda x: brcomp(2, 3, x)),
            ("tiny_shapes", lambda x: brcomp(1e-309, 1e-309, x)),
            ("large_center", lambda x: brcomp(1e30, 2e30, 1 / 3 + (x - 0.5) * 1e-15)),
            ("scaled_tiny", lambda x: brcmp1(1000, 1e-309, 1e-309, x)),
            ("scaled_large", lambda x: brcmp1(-1000, x * 1e300, x * 1e300, 0.5)),
            ("negative_shape", lambda x: brcomp(-0.5, 1, x)),
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
    Path("docs/cdflib-beta-factors-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
