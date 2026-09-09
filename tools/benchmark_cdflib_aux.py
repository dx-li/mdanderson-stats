"""Measure batched descriptor validation against repeated scalar API calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mdanderson_stats import cdflib_aux as aux


def main():
    cases = []
    for size in (256, 10000):
        values = np.tile([0.5, 0.5, 0.25, 0.75, 2, 3], (size, 1))
        values[::3, 4] = -1
        values[::7, :2] = 0
        batch, scalar = [], []
        for _ in range(3):
            start = perf_counter()
            result = aux.validate_parameters(aux.the_beta, 2, values)
            batch.append(perf_counter() - start)
            start = perf_counter()
            expected = np.array([aux.validate_parameters(aux.the_beta, 2, row) for row in values])
            scalar.append(perf_counter() - start)
            np.testing.assert_array_equal(result, expected)
        b, s = median(batch), median(scalar)
        cases.append(dict(size=size, batch_seconds=b, scalar_seconds=s, speedup=s / b))
    report = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        repetitions=3,
        comparison="One batch versus repeated scalar calls to the same API; not native timing",
        cases=cases,
    )
    Path("docs/cdflib-aux-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
