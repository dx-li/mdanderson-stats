"""Measure vectorized elementary helpers against repeated scalar API calls."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mdanderson_stats import alnrel, evaluate_polynomial, rexp, rlog, rlog1


def main():
    cases = []
    for size in (64, 256):
        for name, function in [
            ("alnrel", alnrel),
            ("rexp", rexp),
            ("rlog", rlog),
            ("rlog1", rlog1),
            ("polynomial", lambda x: evaluate_polynomial([1.0, -2.0, 3.0, 4.0], x)),
        ]:
            x = np.linspace(-0.1, 0.1, size) + (1 if name == "rlog" else 0)
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
        platform=platform.platform(),
        repetitions=3,
        comparison=(
            "One broadcast call versus repeated scalar calls to the same Python API; "
            "not native timing"
        ),
        cases=cases,
    )
    Path("docs/cdflib-elementary-benchmark.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
