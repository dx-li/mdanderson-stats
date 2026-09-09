"""Measure STATTAB batching against repeated calls to the same Python API."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import stattab_solve


def main():
    records = []
    for name, size, key, values, fixed in [
        ("normal", 1024, "x", np.linspace(-3, 3, 1024), dict(mean=0, sd=1)),
        ("gamma", 1024, "x", np.linspace(0.1, 10, 1024), dict(rate=2, shape=3)),
        ("binomial", 1024, "s", np.arange(1024) % 100 + 0.5, dict(n=200, pr=0.25)),
        ("poisson", 1024, "s", np.arange(1024) % 100 + 0.5, dict(mean=20)),
        ("normal", 1024, "ccum", np.linspace(0.01, 0.99, 1024), dict(compute="x", mean=0, sd=1)),
        ("poisson", 64, "cum", np.linspace(0.1, 0.9, 64), dict(compute="s", mean=3)),
    ]:
        reference = np.stack(
            [stattab_solve(name, **fixed, **{key: float(v)}).values for v in values]
        )
        batch_times = []
        scalar_times = []
        for _ in range(3):
            start = time.perf_counter()
            result = stattab_solve(name, **fixed, **{key: values})
            batch_times.append(time.perf_counter() - start)
            np.testing.assert_array_equal(result.values, reference)
            start = time.perf_counter()
            repeated = np.stack(
                [stattab_solve(name, **fixed, **{key: float(v)}).values for v in values]
            )
            scalar_times.append(time.perf_counter() - start)
            np.testing.assert_array_equal(repeated, reference)
        batch, scalar = statistics.median(batch_times), statistics.median(scalar_times)
        records.append(
            dict(
                distribution=name,
                compute=fixed.get("compute", "cum"),
                rows=size,
                batch_seconds=batch,
                scalar_seconds=scalar,
                speedup=scalar / batch,
            )
        )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope="Three-run medians; same-Python batching, not native Fortran",
        cases=records,
    )
    Path("docs/stattab-results-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
