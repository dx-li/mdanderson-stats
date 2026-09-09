"""Measure gamma-tail batching, including the subnormal recovery branches."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import grat1, gratio, rcomp


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("gratio_ordinary", lambda x: gratio(10, 10 * x)),
            ("gratio_tiny_shape", lambda x: gratio(1e-100, x)),
            ("gratio_large_center", lambda x: gratio(1e20, 1e20 + x * 1e10)),
            ("gratio_lower_subnormal", lambda x: gratio(200, 2 + 0.1 * x)),
            ("gratio_upper_subnormal", lambda x: gratio(0.5, 730 + x)),
            ("grat1_fraction", lambda x: grat1(0.1, 2 + x, rcomp(0.1, 2 + x))),
        ]:
            x = np.linspace(0.1, 1.2, size)
            batch, scalar = [], []
            for _ in range(3):
                start = perf_counter()
                result = function(x)
                batch.append(perf_counter() - start)
                start = perf_counter()
                expected = np.stack([function(v) for v in x], axis=1)
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
    Path("docs/cdflib-incomplete-gamma-benchmark.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
