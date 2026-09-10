"""Compare vectorized SPPCR intervals with the same scalar Python API."""

import json
import platform
import statistics
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from mdanderson_stats import sppcr_intervals


def main():
    records = []
    options = dict(
        progenitor=(0, 1),
        calibration_sd=0.1,
        frequency_transformed_sd=0.03,
        mutant_transformed_sd=0.02,
    )
    for size in [32, 1024]:
        mu = (np.arange(size * 4).reshape(size, 4) % 13 + 1) / 10
        bt, st = [], []
        for _ in range(3):
            start = time.perf_counter()
            batch = sppcr_intervals(mu, **options)
            bt.append(time.perf_counter() - start)
            start = time.perf_counter()
            separate = [sppcr_intervals(row, **options) for row in mu]
            st.append(time.perf_counter() - start)
            for name in ["calibration", "inverse_calibration", "frequency", "mutant"]:
                actual = getattr(batch, name)
                for field in fields(actual):
                    np.testing.assert_array_equal(
                        getattr(actual, field.name),
                        np.stack([getattr(getattr(s, name), field.name) for s in separate]),
                    )
        a, b = statistics.median(bt), statistics.median(st)
        records.append(
            dict(experiments=size, alleles=4, batch_seconds=a, scalar_seconds=b, speedup=b / a)
        )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope=(
            "Three-run medians; identical intervals and diagnostics; "
            "same Python, not native Fortran"
        ),
        cases=records,
    )
    Path("docs/sppcr-intervals-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
