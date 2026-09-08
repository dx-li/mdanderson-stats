"""Measure joint alternatives/levels against repeated calls to the same Python API."""

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy

from mdanderson_stats import multinomial_power


def main():
    workloads = []
    for n, null in ((60, [0.2, 0.3, 0.5]), (60, [0.1, 0.2, 0.3, 0.4]), (1000, [0.3, 0.7])):
        alternatives = [null, null[::-1], [0.9] + [0.1 / (len(null) - 1)] * (len(null) - 1)]
        levels = [0.001, 0.01, 0.05, 0.2, 0.8]
        batch = multinomial_power(n, null, alternatives, levels)
        for i, alternative in enumerate(alternatives):
            for j, alpha in enumerate(levels):
                single = multinomial_power(n, null, alternative, alpha)
                np.testing.assert_allclose(single.power[:, 0, 0], batch.power[:, i, j], atol=1e-14)
                np.testing.assert_allclose(
                    single.actual_size[:, 0], batch.actual_size[:, j], atol=1e-14
                )
        timing = {}
        for mode in ("joint", "repeated"):
            elapsed = []
            for _ in range(5):
                start = perf_counter()
                if mode == "joint":
                    multinomial_power(n, null, alternatives, levels)
                else:
                    for alternative in alternatives:
                        for alpha in levels:
                            multinomial_power(n, null, alternative, alpha)
                elapsed.append(perf_counter() - start)
            timing[mode] = median(elapsed)
        workloads.append(
            {
                "n": n,
                "null": null,
                "alternatives": alternatives,
                "alpha": levels,
                "points": batch.sample_space_size,
                "seconds": timing,
                "speedup": timing["repeated"] / timing["joint"],
            }
        )
    result = {
        "comparison": "Joint versus 15 repeated Python API calls; both statistics. "
        "Not a comparison with native Fortran. Median of five repeats after correctness checks.",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "workloads": workloads,
    }
    Path("docs/multinompow-benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in workloads:
        print(row["n"], row["points"], row["seconds"], row["speedup"])


if __name__ == "__main__":
    main()
