"""Measure SPPCR frequency-summary batching against the same scalar Python API."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import sppcr_frequencies


def main():
    records = []
    for size in [32, 1024]:
        means = (np.arange(size * 8).reshape(size, 8) % 13 + 1) / 10
        variances = means / 100
        bt, st = [], []
        for _ in range(3):
            start = time.perf_counter()
            batch = sppcr_frequencies(means, variances, progenitor=(0, 1))
            bt.append(time.perf_counter() - start)
            start = time.perf_counter()
            separate = [
                sppcr_frequencies(m, v, progenitor=(0, 1))
                for m, v in zip(means, variances, strict=True)
            ]
            st.append(time.perf_counter() - start)
            for name in ["frequency", "mutant", "calibration"]:
                for attr in ["value", "variance", "standard_error"]:
                    np.testing.assert_array_equal(
                        getattr(getattr(batch, name), attr),
                        np.stack([getattr(getattr(s, name), attr) for s in separate]),
                    )
            for name in ["frequency", "mutant"]:
                np.testing.assert_array_equal(
                    getattr(batch, name).transformed.variance,
                    np.stack([getattr(s, name).transformed.variance for s in separate]),
                )
        a, b = statistics.median(bt), statistics.median(st)
        records.append(
            dict(experiments=size, alleles=8, batch_seconds=a, scalar_seconds=b, speedup=b / a)
        )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope="Three-run medians; same-Python summary batching, not native Fortran speedup",
        cases=records,
    )
    Path("docs/sppcr-frequencies-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
