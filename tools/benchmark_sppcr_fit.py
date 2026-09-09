"""Compare PCR fitting batches with repeated single-experiment Python fits."""

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mdanderson_stats import sppcr_fit_means


def main():
    records = []
    for size in [32, 1024]:
        dna = [1, 2, 3]
        seen = np.arange(size * 3 * 4).reshape(size, 3, 4) % 90 + 1
        batch_times, scalar_times = [], []
        for _ in range(3):
            start = time.perf_counter()
            batch = sppcr_fit_means(dna, seen, 100)
            batch_times.append(time.perf_counter() - start)
            start = time.perf_counter()
            separate = [sppcr_fit_means(dna, s, 100) for s in seen]
            scalar_times.append(time.perf_counter() - start)
            for name in ["mu", "variance", "log_likelihood", "mu_lower", "mu_upper"]:
                np.testing.assert_array_equal(
                    getattr(batch, name), np.stack([getattr(s, name) for s in separate])
                )
        a, b = statistics.median(batch_times), statistics.median(scalar_times)
        records.append(
            dict(
                experiments=size,
                dna_levels=3,
                alleles=4,
                batch_seconds=a,
                scalar_seconds=b,
                speedup=b / a,
            )
        )
    report = dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        scope="Three-run medians; same-Python batching, not native Fortran speedup",
        cases=records,
    )
    Path("docs/sppcr-fit-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
