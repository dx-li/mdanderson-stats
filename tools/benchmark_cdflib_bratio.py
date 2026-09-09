"""Measure bratio batching against repeated scalar calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import bratio


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("ordinary_tails", lambda x: np.stack(bratio(20.0, 50.0, x), axis=-1)),
            ("small_shape_tails", lambda x: np.stack(bratio(0.5, 100.0, x), axis=-1)),
            ("subnormal_shapes", lambda x: np.stack(bratio(1e-309, 5e-324, x), axis=-1)),
            ("subnormal_tail", lambda x: np.stack(bratio(100.0, 1e-20, 0.001 + x * 1e-5), axis=-1)),
            ("huge_shapes", lambda x: np.stack(bratio(1e308, 1e308, x), axis=-1)),
            ("tiny_upper_tail", lambda x: np.stack(bratio(1e-20, 100.0, x * 0.01), axis=-1)),
        ]:
            x = np.linspace(0.1, 0.5, size)
            batch, scalar = [], []
            for _ in range(3):
                start = perf_counter()
                result = function(x)
                batch.append(perf_counter() - start)
                start = perf_counter()
                expected = np.array([function(v) for v in x])
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
    Path("docs/cdflib-bratio-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
