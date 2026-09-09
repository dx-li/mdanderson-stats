"""Measure exact integer remainder batching versus repeated scalar API calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mdanderson_stats import dcdflib_support as support


def main():
    rng = np.random.default_rng(315)
    cases = []
    for size in (256, 10000):
        a = rng.integers(-(2**62), 2**62, size)
        b = rng.integers(1, 2**62, size)
        b[::2] *= -1
        batch, scalar = [], []
        for _ in range(3):
            start = perf_counter()
            result = support.fifmod(a, b)
            batch.append(perf_counter() - start)
            start = perf_counter()
            expected = np.array([support.fifmod(x, y) for x, y in zip(a, b, strict=True)])
            scalar.append(perf_counter() - start)
            np.testing.assert_array_equal(result, expected)
        bt, st = median(batch), median(scalar)
        cases.append(dict(size=size, batch_seconds=bt, scalar_seconds=st, speedup=st / bt))
    report = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        repetitions=3,
        comparison="One batch versus repeated scalar calls to the same API; not native timing",
        cases=cases,
    )
    Path("docs/dcdflib-support-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
