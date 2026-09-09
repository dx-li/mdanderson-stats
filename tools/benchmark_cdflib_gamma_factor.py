"""Measure vectorized gamma scaling factors against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import rcomp


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("ordinary", lambda x: rcomp(2.5, x)),
            ("tiny_shape", lambda x: rcomp(1e-309, x)),
            ("large_center", lambda x: rcomp(1e20, 1e20 + x * 1e10)),
            ("negative_shape", lambda x: rcomp(-0.5, x)),
        ]:
            x = np.linspace(0.1, 1.2, size)
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
    Path("docs/cdflib-gamma-factor-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
