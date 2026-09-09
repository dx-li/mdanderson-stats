"""Measure basym batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import basym


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("ordinary", lambda x: basym(50.0, 100.0, x * 10)),
            ("far_tail_fallback", lambda x: basym(50.0, 15.0, x * 50)),
            ("huge_center_displacement", lambda x: basym(1e308, 1e308, x * 1e146)),
            ("huge_shape_tail", lambda x: basym(1e308, 1e308, x * 1e154)),
            ("subnormal_coordinate_fallback", lambda x: basym(15.0, 1e308, 15 - x * 1e-14)),
            ("underflow_bound", lambda x: basym(1e6, 1e6, x * 1e5)),
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
    Path("docs/cdflib-basym-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
