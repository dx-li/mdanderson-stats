"""Measure vectorized gamma-ratio helpers against repeated scalar API calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import algdiv, bcorr, gsumln


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("algdiv", lambda x: algdiv(x, 8)),
            ("algdiv_negative", lambda x: algdiv(-x, 8)),
            ("algdiv_tiny_ratio", lambda x: algdiv(x * 1e-100, 1e308)),
            ("bcorr", lambda x: bcorr(8 + x, 1e308)),
            ("bcorr_subnormal", lambda x: bcorr(x * 1e308, 1e308)),
            ("gsumln", lambda x: gsumln(1, 1 + x / 2)),
        ]:
            x = np.linspace(0.1, 1.2, size)
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
    Path("docs/cdflib-gamma-ratios-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
