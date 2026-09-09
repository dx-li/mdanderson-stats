"""Measure vectorized error-function/exponential helpers against repeated scalar API calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import erf, erfc1, esum, exparg


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("erf", erf),
            ("erfc", lambda x: erfc1(0, x)),
            ("erfc_scaled", lambda x: erfc1(1, x)),
            ("erfc_subnormal", lambda x: erfc1(0, x + 27)),
            ("esum", lambda x: esum(1000, x - 1000)),
            ("exparg", lambda x: exparg(x > 0)),
        ]:
            x = np.linspace(-0.1, 0.1, size)
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
            "One broadcast call versus repeated scalar calls to the same Python API; "
            "not native timing"
        ),
        cases=cases,
    )
    Path("docs/cdflib-error-exponential-benchmark.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
