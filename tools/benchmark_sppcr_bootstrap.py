"""Benchmark SPPCR bootstrap fitting and summaries on identical replicate counts."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import sppcr_bootstrap_summary, sppcr_fit_means, sppcr_generate


def main():
    records = []
    for size in [32, 1024]:
        samples = sppcr_generate(
            [[0.2, 0.4, 0.1], [0.35, 0.65, 0.2]],
            [40, 80],
            rng=np.random.default_rng(915),
            replicates=size,
        )
        bt, st = [], []
        for _ in range(3):
            start = time.perf_counter()
            fit = sppcr_fit_means([1, 2], samples.seen, samples.wells, saturation="half")
            result = sppcr_bootstrap_summary(fit.mu, progenitor=(0, 1))
            bt.append(time.perf_counter() - start)
            start = time.perf_counter()
            means = np.stack(
                [
                    sppcr_fit_means([1, 2], row, samples.wells, saturation="half").mu
                    for row in samples.seen
                ]
            )
            separate = sppcr_bootstrap_summary(means, progenitor=(0, 1))
            st.append(time.perf_counter() - start)
            np.testing.assert_array_equal(fit.mu, means)
            for name in [
                "mu",
                "calibration",
                "frequency",
                "transformed_frequency",
                "mutant",
                "transformed_mutant",
            ]:
                for field in ["values", "mean", "variance", "standard_deviation"]:
                    np.testing.assert_array_equal(
                        getattr(getattr(result, name), field),
                        getattr(getattr(separate, name), field),
                    )
        a, b = statistics.median(bt), statistics.median(st)
        records.append(
            dict(
                replicates=size,
                levels=2,
                alleles=3,
                batch_seconds=a,
                scalar_seconds=b,
                speedup=b / a,
            )
        )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope=(
            "Three-run medians; identical counts, fits and summaries; "
            "same Python, not native Fortran"
        ),
        cases=records,
    )
    Path("docs/sppcr-bootstrap-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
