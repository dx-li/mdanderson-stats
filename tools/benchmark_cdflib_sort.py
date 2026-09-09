"""Compare the numeric sorting port with Python sorted and array conversion."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from mdanderson_stats import sort_list


def main():
    cases = []
    for size in (10_000, 100_000):
        for dtype in (np.int64, np.float32, np.float64):
            rng = np.random.default_rng(871 + size)
            values = rng.integers(-size, size, size=size).astype(dtype)
            if np.issubdtype(dtype, np.floating):
                values /= 7
            batch, python = [], []
            for _ in range(5):
                start = perf_counter()
                actual = sort_list(values)
                batch.append(perf_counter() - start)
                start = perf_counter()
                expected = np.asarray(sorted(values.tolist()), dtype=dtype)
                python.append(perf_counter() - start)
                np.testing.assert_array_equal(actual, expected)
            b, p = median(batch), median(python)
            cases.append(
                dict(
                    size=size,
                    dtype=np.dtype(dtype).name,
                    numpy_seconds=b,
                    python_seconds=p,
                    speedup=p / b,
                )
            )
    result = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        repetitions=5,
        comparison=(
            "sort_list(array) versus np.asarray(sorted(array.tolist()), dtype=array.dtype); "
            "no native executable timing"
        ),
        cases=cases,
    )
    Path("docs/cdflib-sort-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(cases, indent=2))


if __name__ == "__main__":
    main()
