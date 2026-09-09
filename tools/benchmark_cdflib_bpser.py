"""Measure bpser batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import bpser


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("positive_series", lambda x: bpser(0.5, 0.25, x)),
            ("signed_series", lambda x: bpser(2.0, 8.0, x / 8)),
            ("tiny_shapes", lambda x: bpser(1e-309, 5e-324, x)),
            ("large_companion", lambda x: bpser(100.0, 1e308, x / 1e308)),
            ("near_one", lambda x: bpser(0.5, 1e-20, 1 - x * 1e-12)),
            ("unit_shape", lambda x: bpser(1.0, 0.5, x)),
            ("subnormal_coordinate", lambda x: bpser(0.01, 0.5, x * 1e-308)),
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
    Path("docs/cdflib-bpser-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
