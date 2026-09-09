"""Compare batched gamma quantiles with repeated scalar API calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mdanderson_stats import dcdflib_support as legacy


def main():
    cases = []
    for size in [64, 1000]:
        p = np.linspace(0.001, 0.999, size)
        a = np.linspace(0.5, 100, size)
        batch, scalar = [], []
        for _ in range(3):
            start = perf_counter()
            result = legacy.gaminv(a, p, 1 - p)
            batch.append(perf_counter() - start)
            start = perf_counter()
            expected = np.array([legacy.gaminv(v, x, 1 - x) for x, v in zip(p, a, strict=True)])
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
    Path("docs/dcdflib-gamma-inverse-benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
