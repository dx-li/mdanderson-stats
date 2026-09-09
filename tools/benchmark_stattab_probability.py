"""Compare broadcast discrete terms with repeated calls to the same Python API."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import (
    stattab_binomial_term,
    stattab_negative_binomial_term,
    stattab_poisson_term,
)


def main():
    records = []
    for size in [64, 1024]:
        counts = np.arange(size, dtype=float) % 100 + 0.5
        for fn, parameters in [
            (stattab_binomial_term, (200, 0.25)),
            (stattab_negative_binomial_term, (20, 0.25)),
            (stattab_poisson_term, (20,)),
        ]:
            expected = np.array([fn(float(k), *parameters) for k in counts])
            batched = []
            scalar = []
            for _ in range(3):
                start = time.perf_counter()
                actual = fn(counts, *parameters)
                batched.append(time.perf_counter() - start)
                np.testing.assert_array_equal(actual, expected)
                start = time.perf_counter()
                again = np.array([fn(float(k), *parameters) for k in counts])
                scalar.append(time.perf_counter() - start)
                np.testing.assert_array_equal(again, expected)
            a, b = statistics.median(batched), statistics.median(scalar)
            records.append(
                dict(
                    function=fn.__name__,
                    rows=size,
                    batch_seconds=a,
                    scalar_seconds=b,
                    speedup=b / a,
                )
            )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope="Three-run medians; batching versus repeated same-Python calls, not native Fortran",
        cases=records,
    )
    Path("docs/stattab-probability-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
